"""Generic bounded Z3 templates with UNSAT/optimality checks; CPU-only."""
from __future__ import annotations
from typing import Any


def _bounded_int(value: Any, lo: int, hi: int, name: str) -> int:
    if type(value) is not int or not lo <= value <= hi: raise ValueError(name)
    return value


def solve_constraints(spec: dict, timeout_ms: int = 3000) -> dict:
    import z3
    if not isinstance(spec, dict) or len(str(spec)) > 8000: raise ValueError("invalid specification")
    _bounded_int(timeout_ms, 100, 10000, "timeout")
    template = spec.get("template")
    if template == "schedule":
        tasks = spec.get("tasks", [])
        slots = _bounded_int(spec.get("slots"), 1, 24, "slots")
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= 12: raise ValueError("tasks")
        names = [t["id"] for t in tasks]
        if len(set(names)) != len(names) or any(type(n) is not str or not 1 <= len(n) <= 30 for n in names):
            raise ValueError("task ids")
        starts = {n: z3.Int("start_" + str(i)) for i,n in enumerate(names)}
        hard = []
        soft = []
        for t in tasks:
            dur = _bounded_int(t.get("duration", 1), 1, slots, "duration")
            window = t.get("window", [0, slots-1])
            if (not isinstance(window, list) or len(window) != 2): raise ValueError("window")
            earliest = _bounded_int(window[0], 0, slots-1, "window")
            latest = _bounded_int(window[1], earliest, slots-1, "window")
            hard.extend([starts[t["id"]] >= earliest, starts[t["id"]] <= min(latest, slots-dur)])
            pref = t.get("preferred")
            if pref is not None:
                p = _bounded_int(pref, 0, slots-1, "preferred")
                weight = _bounded_int(t.get("penalty", 1), 1, 100, "penalty")
                soft.append(z3.If(starts[t["id"]] == p, 0, weight))
        for i,a in enumerate(tasks):
            for b in tasks[i+1:]:
                if a.get("resource") is not None and a.get("resource") == b.get("resource"):
                    hard.append(z3.Or(starts[a["id"]] + a.get("duration",1) <= starts[b["id"]], starts[b["id"]] + b.get("duration",1) <= starts[a["id"]]))
            for pred in a.get("after", []):
                if pred not in starts: raise ValueError("invalid predecessor")
                p = tasks[names.index(pred)]
                hard.append(starts[a["id"]] >= starts[pred] + p.get("duration",1))
        variables = starts
        objective = z3.Sum(soft) if soft else z3.IntVal(0)
    elif template == "resources":
        demands = spec.get("demands", [])
        capacity = _bounded_int(spec.get("capacity"), 0, 500, "capacity")
        if not isinstance(demands, list) or not 1 <= len(demands) <= 12: raise ValueError("demands")
        variables = {str(i): z3.Int(f"allocated_{i}") for i in range(len(demands))}
        hard = [z3.Sum(list(variables.values())) <= capacity]
        soft = []
        for i,item in enumerate(demands):
            need = _bounded_int(item["need"], 0, 100, "need")
            minimum = _bounded_int(item.get("minimum",0), 0, need, "minimum")
            weight = _bounded_int(item.get("weight",1), 1, 100, "weight")
            v = variables[str(i)]
            hard.extend([v >= minimum, v <= need])
            soft.append((need-v)*weight)
        objective = z3.Sum(soft)
    elif template == "integer_puzzle":
        variables_raw = spec.get("variables", [])
        if not isinstance(variables_raw, list) or not 1 <= len(variables_raw) <= 8 or len(set(variables_raw)) != len(variables_raw):
            raise ValueError("variables")
        if any(type(v) is not str or not v.isidentifier() or len(v)>20 for v in variables_raw): raise ValueError("variable name")
        variables = {n:z3.Int(n) for n in variables_raw}
        lower,upper = _bounded_int(spec.get("lower",0),-1000,1000,"lower"),_bounded_int(spec.get("upper",100),-1000,1000,"upper")
        if lower>upper: raise ValueError("bounds")
        hard=[z3.And(v>=lower,v<=upper) for v in variables.values()]
        clauses=spec.get("constraints",[])
        if not isinstance(clauses,list) or not 1<=len(clauses)<=32: raise ValueError("constraints")
        for cl in clauses:
            if not isinstance(cl,dict): raise ValueError("clause")
            typ=cl.get("type")
            names=cl.get("vars",[])
            if not isinstance(names,list) or any(n not in variables for n in names): raise ValueError("unknown variable")
            vals=[variables[n] for n in names]
            val=_bounded_int(cl.get("value",0),-10000,10000,"value")
            if typ=="sum_eq": hard.append(z3.Sum(vals)==val)
            elif typ=="sum_le": hard.append(z3.Sum(vals)<=val)
            elif typ=="all_different": hard.append(z3.Distinct(*vals))
            elif typ=="difference_eq" and len(vals)==2: hard.append(vals[0]-vals[1]==val)
            elif typ=="mod_eq" and len(vals)==1:
                mod=_bounded_int(cl.get("modulus"),1,100,"modulus")
                hard.append(vals[0]%mod==val)
            elif typ=="digit_sum_eq" and len(vals)==1 and lower>=0:
                v=vals[0]
                hard.append(z3.Sum([(v/(10**p))%10 for p in range(5)])==val)
            else: raise ValueError("clause type")
        objective=None
    else:
        raise ValueError("unknown template")
    opt=z3.Optimize()
    opt.set(timeout=timeout_ms)
    opt.add(*hard)
    if objective is not None:
        opt.minimize(objective)
        for variable in variables.values(): opt.minimize(variable)
    result=opt.check()
    if result==z3.unsat: return {"verified":True,"status":"unsat","result":None,"method":"z3_unsat"}
    if result!=z3.sat: return {"verified":False,"status":"unknown","result":None}
    model=opt.model()
    assignment={name:model.eval(v,model_completion=True).as_long() for name,v in variables.items()}
    evaluator=z3.Solver(); evaluator.set(timeout=timeout_ms); evaluator.add(*hard)
    evaluator.add(z3.Or(*[v!=assignment[n] for n,v in variables.items()]))
    if objective is None:
        uniqueness=(evaluator.check()==z3.unsat)
        return {"verified":True,"status":"sat","result":assignment,"unique":uniqueness,"method":"z3_witness"}
    score=model.eval(objective, model_completion=True).as_long()
    evaluator=z3.Solver(); evaluator.set(timeout=timeout_ms); evaluator.add(*hard,objective<score)
    check=evaluator.check()
    return {"verified":check==z3.unsat,"status":"optimal" if check==z3.unsat else "unproven","result":assignment,"cost":score,"method":"z3_unsat_better"}
