"""Independent RTX-host 1.2 SymPy/NetworkX/Z3 and Python Docker AST smoke.

Distinct new examples; never makes a 24-case historical score claim.
"""
from __future__ import annotations
import argparse,ast,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("Existing check must not be overwritten")
    from neral_cpu_foundation.symbolic import verify_symbolic
    from neral_cpu_foundation.graphs import verify_graph
    from neral_cpu_foundation.constraints import solve_constraints
    symbolic=verify_symbolic({"operation":"equivalent",
           "left":"(x+7)**2","right":"x**2+14*x+49"})
    if symbolic.get("verified") is not True:raise RuntimeError("Symbolic proof failed")
    graph=verify_graph({"operation":"shortest_path","directed":True,
        "nodes":["P","Q","R"],"edges":[["P","Q",3],["Q","R",4],["P","R",12]],
        "source":"P","target":"R"})
    if graph.get("verified") is not True or graph.get("result",{}).get("length")!=7:
        raise RuntimeError("NetworkX independent shortest path failed")
    z3=solve_constraints({"template":"integer_puzzle",
        "variables":["a","b"],"lower":0,"upper":9,
        "constraints":[{"type":"sum_eq","vars":["a","b"],"value":11},
                       {"type":"difference_eq","vars":["a","b"],"value":3}]})
    if z3.get("verified") is not True or z3.get("result")!={"a":7,"b":4}:
        raise RuntimeError("Z3 exact witness failed")
    broken="total=0\nfor x in range(4)\n    total+=x*x\nprint(total)"
    correct=broken.replace("for x in range(4)","for x in range(4):")
    try:ast.parse(broken)
    except SyntaxError:pass
    else:raise RuntimeError("Negative AST test not broken")
    ast.parse(correct)
    # Real Docker execution of the one-colon corrected program, no network.
    pin="python:3.12.8-alpine3.21"
    cmd=["docker","run","--rm","--network","none","--read-only",
        "--cap-drop","ALL","--security-opt","no-new-privileges",
        "--memory","192m","--cpus","1","--pids-limit","64",
        pin,"python","-B","-c",correct]
    run=subprocess.run(cmd,text=True,capture_output=True,timeout=55)
    if run.returncode!=0 or run.stdout.strip()!="14" or run.stderr.strip():
        raise RuntimeError("Docker AST one-step correction not independently executed")
    report={"status":"new_source_tool_regression_pass",
      "sympy":True,"networkx":True,"z3":True,
      "python_docker_one_step_ast":True,"docker_isolated_network":True,
      "does_not_substitute_for_24_original":True,"no_training":True}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NERAL_V2_TOOLS_REGRESSION "+json.dumps(report,sort_keys=True),flush=True)

if __name__=="__main__":main()
