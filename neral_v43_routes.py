"""Source-derived V4.3 verification routes; no model-authored gold or tests."""
import ast
import json
import re
import threading
import time
from neral_v43_bounded import Verdict, exact_integer_verdict
from scripts.neral_v43_obsidian_math import exact_algebra, falsify_claim, boundary_fuzz
from scripts.neral_v43_obsidian_codex import run_test_repair_loop
_WEB_SLOT=threading.BoundedSemaphore(1)

def source_plan(question):
    if not isinstance(question,str) or len(question)>12000: return None
    # Explicit structured constraints and tests belong to the user, not the model.
    for tag,kind in (("NERAL_MATH_SPEC:","logic"),("NERAL_TEST_SPEC:","code")):
        if "\n"+tag in question:
            try: spec=json.loads(question.rsplit("\n"+tag,1)[1].strip())
            except (ValueError,RecursionError): return None
            if isinstance(spec,dict): return kind,spec
    m=re.fullmatch(r"(?i)\s*(?:berechne|rechne|calculate)\s+([0-9 ()+*/%-]{1,100})[.?!]?\s*",question)
    if m:
        expr=m[1].strip()
        # Bound intermediate symbolic work; do not expand powers.
        if "**" not in expr and "%" not in expr: return "arithmetic",expr
    # Generic explicit Python assert examples, parsed without executing them.
    cases=[]
    for line in question.splitlines():
        if not line.strip().startswith("assert "): continue
        try:
            tree=ast.parse(line.strip())
            if len(tree.body)!=1: continue
            node=tree.body[0]
            if not isinstance(node,ast.Assert): continue
            test=node.test
            if not (isinstance(test,ast.Compare) and len(test.ops)==1 and
                    isinstance(test.ops[0],ast.Eq) and isinstance(test.left,ast.Call)
                    and isinstance(test.left.func,ast.Name) and not test.left.keywords): continue
            cases.append({"function":test.left.func.id,"args":[ast.literal_eval(a) for a in test.left.args],
                          "expected":ast.literal_eval(test.comparators[0])})
        except (SyntaxError,ValueError,TypeError,RecursionError): continue
    if cases: return "code",{"tests":cases,"properties":[]}
    return None

def _web(question,query,search,deadline):
    if search is None or not _WEB_SLOT.acquire(blocking=False): return Verdict("unknown")
    result=[]; done=threading.Event()
    def work():
        try:
            from neral_page_reader import read_top_pages
            found=search.search(query)
            pages=read_top_pages(found.get("results",[]),query)
            result.extend(p for p in pages if isinstance(p,dict) and p.get("page_read") is True)
        except Exception:
            pass
        finally:
            _WEB_SLOT.release(); done.set()
    try: threading.Thread(target=work,name="v43-web-read",daemon=True).start()
    except RuntimeError:
        _WEB_SLOT.release(); return Verdict("unknown")
    if not done.wait(max(0,deadline-time.monotonic())): return Verdict("timeout")
    # Retrieval is not semantic confirmation; never invent a source or cite a
    # snippet as proof. Keep the draft and expose only actual read-page metadata.
    sources=tuple({"url":p.get("url"),"title":p.get("title",""),"page_read":True}
                  for p in result[:3] if isinstance(p.get("url"),str))
    return Verdict("unknown",sources=sources)

def verify_answer(question,candidate,*,deadline,web_query=None,search=None,
                  backend="auto",trusted_certificate=None):
    if time.monotonic()>=deadline: return Verdict("timeout")
    if web_query:
        return _web(question,web_query,search,deadline)
    plan=source_plan(question)
    if plan is None:
        if trusted_certificate is not None:
            cert=trusted_certificate()
            if cert and cert.get("verified") is True and time.monotonic()<deadline:
                return exact_integer_verdict(candidate,cert)
        return Verdict("unsupported")
    kind,spec=plan
    text=candidate["text"]
    if kind=="arithmetic":
        try: expected=exact_algebra(spec)
        except (SyntaxError,ValueError,ArithmeticError): return Verdict("unsupported")
        if time.monotonic()>=deadline: return Verdict("timeout")
        return exact_integer_verdict(candidate,{"text":expected})
    if kind=="logic":
        try:
            check=falsify_claim(spec,deadline=deadline)
            if time.monotonic()>=deadline: return Verdict("timeout")
            fuzz=boundary_fuzz(spec,max_probes=64,deadline=deadline)
        except (SyntaxError,ValueError,TypeError,KeyError,ArithmeticError):
            return Verdict("unsupported")
        if time.monotonic()>=deadline: return Verdict("timeout")
        status=check["status"]
        if status not in ("counterexample","no_counterexample"): return Verdict("unknown")
        if status=="no_counterexample" and fuzz["status"]=="counterexample": return Verdict("unknown")
        val=text.strip().rstrip(".!").casefold()
        if val not in ("ja","nein","yes","no"): return Verdict("unsupported")
        good=(val in ("ja","yes"))==(status=="no_counterexample")
        return Verdict("verified") if good else Verdict("counterexample",
            "Independent bounded Z3 result: "+status+"; "+json.dumps(check["assignment"]))
    if kind=="code":
        if set(spec)!={"tests","properties"} or not (spec["tests"] or spec["properties"]):
            return Verdict("unsupported")
        code=text.strip()
        m=re.fullmatch(r"\x60\x60\x60(?:python)?\s*\n(.*?)\n\x60\x60\x60",code,re.S|re.I)
        if m: code=m[1]
        if not re.match(r"(?:def\s|import\s|from\s)",code): return Verdict("unsupported")
        try: ast.parse(code)
        except SyntaxError as exc:
            return Verdict("counterexample","Python SyntaxError at line "+str(exc.lineno))
        r=run_test_repair_loop(code,tests=spec["tests"],properties=spec["properties"],
                               max_repairs=0,deadline=deadline,backend=backend)
        status=r["status"]
        if status=="passed": return Verdict("verified")
        if status in ("tests_failed","property_counterexample") and r.get("evidence"):
            return Verdict("counterexample",r["evidence"])
        return Verdict("unknown" if status!="timeout" else "timeout")
    return Verdict("unsupported")
