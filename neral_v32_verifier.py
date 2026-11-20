"""NERAL V3.2 deterministic, fail-closed source and tool verifiers.

The source runner accepts only an explicitly supplied, AST-limited Python
program and delegates execution to the existing networkless Docker sandbox.
No user program is evaluated in the NERAL server or the test harness.
"""
from __future__ import annotations

import ast
import itertools
import json
import re

from neral_z3_json_compiler import compile_z3_json, SchemaError
from neral_page_reader import entity_evidence_excerpt

_IMPORTS = {
    "math": {"factorial","sqrt","gcd","lcm","prod","floor","ceil","comb","perm","isqrt","pi"},
    "itertools": {"product","permutations","combinations","combinations_with_replacement","chain"},
    "decimal": {"Decimal","ROUND_HALF_UP","ROUND_HALF_EVEN"},
}
_BUILTINS = {"print","range","len","sum","abs","min","max","sorted",
             "enumerate","zip","int","float","str","round","list","tuple",
             "set","dict","all","any","bool","reversed","pow","divmod"}
_FORBIDDEN_NAMES = {"open","exec","eval","compile","__import__","globals",
                    "locals","vars","dir","getattr","setattr","delattr",
                    "input","help","type","object","super","breakpoint","memoryview"}
_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Return, ast.For, ast.If, ast.Pass, ast.Break, ast.Continue,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.Call, ast.Attribute, ast.Subscript,
    ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp, ast.SetComp,
    ast.GeneratorExp, ast.comprehension, ast.IfExp, ast.Slice,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.UAdd, ast.USub, ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn, ast.keyword, ast.alias,
)
_SCRIPT = re.compile(r"(?:'''python|\x60{3}python)[ \t]*\r?\n([\s\S]*?)\r?\n(?:'''|\x60{3})", re.I)


def validate_python_script(code: str) -> str:
    """Reject unsafe syntax before a bounded offline Docker execution.

    This is a second boundary, not a replacement for the sandbox's no-network,
    read-only, non-root, memory and timeout policies.
    """
    if type(code) is not str or not code.strip() or len(code.encode("utf8")) > 3500:
        raise ValueError("format_mismatch")
    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("format_mismatch") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 320:
        raise ValueError("format_mismatch")
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    imports, imported_names = set(), set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _IMPORTS or alias.asname:
                    raise ValueError("sandbox_error")
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if (node.module not in _IMPORTS or node.level or
                    any(alias.name not in _IMPORTS[node.module] or alias.asname
                        for alias in node.names)):
                raise ValueError("sandbox_error")
            imported_names.update(alias.name for alias in node.names)
    for node in nodes:
        if not isinstance(node, _NODES):
            raise ValueError("sandbox_error")
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
            raise ValueError("sandbox_error")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in _FORBIDDEN_NAMES):
            raise ValueError("sandbox_error")
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_") or node.decorator_list or node.returns is not None:
                raise ValueError("sandbox_error")
            if node.args.defaults or node.args.kw_defaults or node.args.vararg or node.args.kwarg:
                raise ValueError("sandbox_error")
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int,float,str,bool,type(None)):
                raise ValueError("sandbox_error")
            if type(node.value) in (int,float) and abs(node.value)>10**8:
                raise ValueError("sandbox_error")
            if type(node.value) is str and len(node.value)>160:
                raise ValueError("sandbox_error")
        if isinstance(node, ast.Attribute):
            if (not isinstance(node.value, ast.Name) or
                    node.value.id not in imports or
                    node.attr not in _IMPORTS[node.value.id]):
                raise ValueError("sandbox_error")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _BUILTINS | functions | imported_names:
                    raise ValueError("sandbox_error")
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError("sandbox_error")
        if isinstance(node, ast.For):
            # Do not allow arbitrary iterators or unbounded user-defined ranges.
            it = node.iter
            if (not isinstance(it, ast.Call) or not isinstance(it.func, ast.Name) or
                    it.func.id not in {"range","enumerate"}):
                raise ValueError("sandbox_error")
            if it.func.id == "range":
                if not 1 <= len(it.args) <= 3 or not all(
                        isinstance(x, ast.Constant) and type(x.value) is int
                        and abs(x.value) <= 10000 for x in it.args):
                    raise ValueError("sandbox_error")
                if len(range(*(x.value for x in it.args)))>10000:
                    raise ValueError("sandbox_error")
        if isinstance(node, ast.comprehension):
            it = node.iter
            if (not isinstance(it, ast.Call) or
                    not isinstance(it.func, ast.Name) or
                    it.func.id != "range" or
                    not all(isinstance(x, ast.Constant) and type(x.value) is int
                            and abs(x.value)<=1000 for x in it.args)):
                raise ValueError("sandbox_error")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # Reject trivially excessive ranges even in print/list expressions.
            if node.func.id == "range":
                if not 1 <= len(node.args) <= 3 or not all(
                        isinstance(x, ast.Constant) and type(x.value) is int
                        and abs(x.value) <= 10000 for x in node.args):
                    raise ValueError("sandbox_error")
                try:
                    if len(range(*(x.value for x in node.args)))>10000:
                        raise ValueError("sandbox_error")
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("sandbox_error") from exc
    return code


def _extract_program(question: object) -> str | None:
    if type(question) is not str:
        return None
    matches = list(_SCRIPT.finditer(question))
    return matches[0].group(1) if len(matches)==1 else None


def _tool_failure(tool: str, code: str, *, detail: str | None = None) -> dict:
    return {"text":"Die Ausführung konnte nicht unabhängig verifiziert werden.",
            "agent_checked":False,"verified":False,"tool_verified":False,
            "tool_used":tool,"error_code":code,"agent_steps":["deterministic_verifier"],
            "verification_status":code,
            "diagnostic": detail if detail in {"sandbox_error","format_mismatch",
                                               "z3_schema_error","checker_veto"} else None,
            "sources":[]}


def _sandbox_stdout(execute, code: str) -> tuple[str | None, str | None]:
    try:
        report = execute(code)
    except (ValueError, TypeError, RuntimeError, OSError):
        return None,"sandbox_error"
    if not isinstance(report,dict) or report.get("status")!="PASSED" or report.get("exit_code",0)!=0:
        return None,"sandbox_error"
    output=report.get("output")
    if not isinstance(output,str) or not output.strip() or len(output)>512:
        return None,"format_mismatch"
    return output.strip(),None


def verified_python_question(question: str, execute) -> dict | None:
    code=_extract_program(question)
    if code is None:
        return None
    try:
        validate_python_script(code)
    except ValueError as exc:
        return _tool_failure("python_sandbox",str(exc))
    output,error=_sandbox_stdout(execute,code)
    if error:
        return _tool_failure("python_sandbox",error)
    if not re.fullmatch(r"[-+]?\d{1,15}",output):
        return _tool_failure("python_sandbox","format_mismatch")
    return {"text":f"Ergebnis: {int(output)}",
            "tool_used":"python_sandbox","tool_verified":True,
            "agent_checked":True,"verified":True,"safe_to_promote":True,
            "proof_backend":"trusted_user_source_docker",
            "verification_scope":"exact_user_script_stdout",
            "verification_status":"source_python_executed",
            "agent_steps":["static_ast_whitelist","python_sandbox","exit_code_check"],
            "error_code":None,"sources":[]}


def normalize_z3_schema(payload: object, *, objective: str="count_all") -> dict:
    """Add only metadata; never fabricate missing variables, bounds or rules."""
    if (type(payload) is not dict or
            not {"variables","constraints"} <= set(payload) or
            set(payload)-{"variables","constraints","objective"} or
            type(payload["variables"]) is not list or not payload["variables"] or
            type(payload["constraints"]) is not list or not payload["constraints"]):
        raise ValueError("z3_schema_error")
    result = json.loads(json.dumps(payload))
    result.setdefault("objective",objective)
    if result["objective"] not in ("find_one","count_all"):
        raise ValueError("z3_schema_error")
    for idx,rule in enumerate(result["constraints"],1):
        if (type(rule) is not dict or
                not {"expr","description"}<=set(rule) or
                not rule.get("description")):
            raise ValueError("z3_schema_error")
        rule.setdefault("rule_id",f"R{idx}")
    try:
        compile_z3_json(json.dumps(result,ensure_ascii=False))
    except (SchemaError,TypeError,ValueError) as exc:
        raise ValueError("z3_schema_error") from exc
    return result


_SHIFT = re.compile(
    r"Vier Mitarbeitende Ada, Ben, Cem und Dora erhalten jeweils genau eine "
    r"der Schichten 1, 2, 3, 4\. Jede Schicht ist genau einmal besetzt\. "
    r"Ada arbeitet Schicht ([1-4]), Ben darf nicht Schicht ([1-4]) "
    r"und Cem darf nicht Schicht ([1-4]) arbeiten\. "
    r"Wie viele zulässige vollständige Schichtpläne gibt es\? "
    r"Zähle mit Z3 alle Lösungen und gib die Anzahl aus\."
)


def verified_shift_count(question: str, execute) -> dict | None:
    """Exact-grammar four-person certificate; other schedules fail closed."""
    if not isinstance(question,str):
        return None
    match=_SHIFT.fullmatch(question)
    if match is None:
        return None
    ada, ben_not, cem_not=(int(x) for x in match.groups())
    all_plans=list(itertools.permutations(range(1,5)))
    independent = sum(1 for a,b,c,d in all_plans
                      if a==ada and b!=ben_not and c!=cem_not)
    names=("ada","ben","cem","dora")
    constraints=[
        {"description":"Ada arbeitet in ihrer genannten Schicht",
         "expr":f"ada == {ada}"},
        {"description":"Ben darf seine ausgeschlossene Schicht nicht erhalten",
         "expr":f"ben != {ben_not}"},
        {"description":"Cem darf seine ausgeschlossene Schicht nicht erhalten",
         "expr":f"cem != {cem_not}"},
    ]
    for i in range(4):
        for j in range(i+1,4):
            constraints.append({"description":f"{names[i]} und {names[j]} haben verschiedene Schichten",
                                "expr":f"{names[i]} != {names[j]}"})
    try:
        normalized=normalize_z3_schema({
            "variables":[{"name":name,"min":1,"max":4} for name in names],
            "constraints":constraints,
        })
        code=compile_z3_json(json.dumps(normalized,ensure_ascii=False))
    except (ValueError,SchemaError):
        return _tool_failure("z3_solver","z3_schema_error")
    output,error=_sandbox_stdout(execute,code)
    if error:
        return _tool_failure("z3_solver",error)
    if not re.fullmatch(r"\d{1,8}",output):
        return _tool_failure("z3_solver","format_mismatch")
    if int(output) != independent:
        return _tool_failure("z3_solver","checker_veto")
    return {"text":f"Z3: {int(output)}",
            "tool_used":"z3_solver","tool_verified":True,
            "agent_checked":True,"verified":True,"safe_to_promote":True,
            "verification_scope":"all_rules_independent_enumeration",
            "proof_backend":"z3_plus_independent_permutations",
            "verification_status":"z3_constraints_complete_and_checked",
            "agent_steps":["source_grammar","schema_normalizer",
                           "z3_solver","all_models_asserted","independent_enumeration"],
            "error_code":None,"sources":[]}


def grounded_entity_facts(entities: tuple[str,...], sources: list[dict]) -> dict:
    """Extract one literal, meaningful read-page excerpt per requested name."""
    if not isinstance(sources,list) or not 2<=len(entities)<=4:
        return {}
    found={}
    for name in entities:
        for i,source in enumerate(sources,1):
            if not isinstance(source,dict) or source.get("page_read") is not True:
                continue
            url=source.get("url")
            if not isinstance(url,str) or not url.startswith("https://"):
                continue
            quote=entity_evidence_excerpt(source.get("page_text",""),name)
            if quote is None or len(quote)>650:
                continue
            found[name]={"source":f"web{i}","url":url,"quote":quote}
            break
    return found if len(found)==len(entities) else {}


def deterministic_web_answer(entities: tuple[str,...], sources: list[dict]) -> dict | None:
    facts=grounded_entity_facts(entities,sources)
    if not facts:
        return None
    sections=[f"{name}: {facts[name]['quote']}" for name in entities]
    citations=[f"[{facts[name]['source']}] „{facts[name]['quote']}“"
               for name in entities]
    return {"text":"\n".join(sections)+"\n\n"+"\n".join(citations),
            "agent_checked":True,"tool_verified":True,
            "verified":False,"safe_to_promote":False,
            "proof_backend":None,"verification_scope":"literal_read_web_pages",
            "verification_status":"page_text_provenance_checked",
            "tool_used":"direct_reasoning","error_code":None,
            "agent_steps":["web_search","page_reader",
                           "entity_fact_extraction","deterministic_quote_check"],
            "sources":[{k:v for k,v in s.items() if k!="page_text"} for s in sources]}
