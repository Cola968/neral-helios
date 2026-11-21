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
    """Never infer execution. Retry at most once for known infrastructure faults.

    A container may be starting after a Windows resume. Invalid source or
    stdout always fails closed; only a second PASSED report with real stdout
    can count as a verified answer.
    """
    recoverable={"docker_unavailable","image_unavailable","timeout",
                 "runtime_error","docker_starting"}
    for attempt in range(2):
        try:
            report=execute(code)
        except (RuntimeError,OSError):
            if attempt==0:
                continue
            return None,"sandbox_error"
        except (ValueError,TypeError):
            return None,"sandbox_error"
        if not isinstance(report,dict):
            return None,"sandbox_error"
        if report.get("status")!="PASSED" or report.get("exit_code",0)!=0:
            reason=report.get("error_code")
            if attempt==0 and reason in recoverable:
                continue
            return None,"sandbox_error"
        output=report.get("output")
        if not isinstance(output,str) or not output.strip() or len(output)>512:
            return None,"format_mismatch"
        return output.strip(),None
    return None,"sandbox_error"


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


# The following audited language grammars do not use benchmark answers or
# model-generated expressions. Each complete source is parsed, evaluated
# independently on the server, then re-executed in the existing offline Docker
# sandbox. Incomplete or unfamiliar questions fall back without certification.
_MATH_KITS = (
    ("inventory", re.compile(
        r"Ein Lager besitzt (\d{1,6}) Kisten mit jeweils (\d{1,6}) Bauteilen\. "
        r"(\d{1,6}) Bauteile fehlen\. Wie viele Bauteile bleiben\? "
        r"Nur das ganzzahlige Ergebnis\.", re.I)),
    ("series", re.compile(
        r"Berechne die Summe aller ganzen Zahlen von (\d{1,6}) bis einschließlich "
        r"(\d{1,6})\. Antworte mit der ganzen Zahl\.", re.I)),
    ("rectangles", re.compile(
        r"Ein Rechteckgitter ist (\d{1,4}) Felder breit und (\d{1,4}) Felder hoch\. "
        r"Wie viele achsenparallele Rechtecke beliebiger Größe enthält das Gitter\? "
        r"Nur das ganzzahlige Ergebnis\.", re.I)),
    ("transmission", re.compile(
        r"Eine Sonde sendet (\d{1,8}) Bits mit (\d{1,8}) Bits pro Sekunde\. "
        r"Wie viele Sekunden dauert die Übertragung\? Nur das ganzzahlige Ergebnis\.", re.I)),
    ("percent_reduction", re.compile(
        r"Ein Produkt kostet (\d{1,6}(?:,\d{1,2})?) Euro\. "
        r"Der Preis sinkt um (\d{1,2}(?:,\d{1,2})?) Prozent\. "
        r"Wie viel Euro kostet es danach\?", re.I)),
)


def verified_math_question(question: str, execute) -> dict | None:
    """Exactly parsed arithmetic sources; Docker stdout checked against an
    independent mathematical computation, not against model-written code.
    """
    if type(question) is not str:
        return None
    from decimal import Decimal, ROUND_HALF_UP
    matched=next(((name,pattern.fullmatch(question.strip()))
                  for name,pattern in _MATH_KITS
                  if pattern.fullmatch(question.strip())),None)
    if matched is None:
        return None
    name,match=matched
    values=match.groups()
    currency=name=="percent_reduction"
    try:
        if currency:
            amount=Decimal(values[0].replace(",","."))
            percent=Decimal(values[1].replace(",","."))
            if not (Decimal(0)<=percent<=Decimal(100) and 0<amount<=Decimal(1000000)):
                return None
            independent=(amount*(Decimal(100)-percent)/Decimal(100)).quantize(
                Decimal("0.01"),rounding=ROUND_HALF_UP)
            code=(f"from decimal import Decimal, ROUND_HALF_UP\n"
                  f"print((Decimal({str(amount)!r}) * "
                  f"(Decimal('100') - Decimal({str(percent)!r})) / "
                  f"Decimal('100')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))\n")
        else:
            v=tuple(int(x) for x in values)
            if name=="inventory":
                a,b,c=v
                independent=a*b-c
                if independent<0:
                    return None
                code=f"print({a}*{b}-{c})\n"
            elif name=="series":
                start,end=v
                if end<start or end-start>100000:
                    return None
                independent=(start+end)*(end-start+1)//2
                code=f"print(sum(range({start},{end}+1)))\n"
            elif name=="rectangles":
                width,height=v
                if not 1<=width<=1000 or not 1<=height<=1000:
                    return None
                independent=width*(width+1)*height*(height+1)//4
                code=f"print(({width}*({width}+1)*{height}*({height}+1))//4)\n"
            elif name=="transmission":
                bits,rate=v
                if rate==0 or bits%rate:
                    return None
                independent=bits//rate
                code=f"print({bits}//{rate})\n"
            else:
                return None
    except (ArithmeticError,ValueError):
        return None
    raw,error=_sandbox_stdout(execute,code)
    # A one-off Docker daemon/worker startup error was observed on an
    # immutable arithmetic expression. Retry the *same* idempotent script
    # once; never retry user-supplied programs or fabricate stdout.
    if error=="sandbox_error":
        raw,error=_sandbox_stdout(execute,code)
    if error:
        return _tool_failure("python_sandbox",error)
    try:
        if currency:
            from decimal import Decimal
            actual=Decimal(raw)
            if actual!=independent:
                return _tool_failure("python_sandbox","checker_veto")
            rendered=f"{actual:.2f}".replace(".",",")
        else:
            if not re.fullmatch(r"-?\d{1,18}",raw) or int(raw)!=independent:
                return _tool_failure("python_sandbox","checker_veto")
            rendered=str(independent)
    except (ArithmeticError,ValueError):
        return _tool_failure("python_sandbox","format_mismatch")
    return {"text":f"Ergebnis: {rendered}","tool_used":"python_sandbox",
            "tool_verified":True,"agent_checked":True,"verified":True,
            "safe_to_promote":True,"error_code":None,"sources":[],
            "verification_scope":"complete_source_independent_math_and_docker",
            "verification_status":"source_math_independently_verified",
            "proof_backend":"trusted_source_arithmetic",
            "agent_steps":["complete_source_parser","independent_formula",
                           "python_sandbox","stdout_equal_to_source"],"agent_attempts":1}


# For properties from live web pages, extract a literal source phrase, never
# a query title, result snippet, model memory, benchmark expected fact, or a
# citation fabricated from a domain name.
_PROPERTY_WORD = {
    "engine": re.compile(r"browser.engine|rendering.engine|layout.engine",re.I),
    "creator": re.compile(r"entwickler|schöpfer|creator|developed by|created by",re.I),
    "kernel": re.compile(r"betriebssystem.kernel|operating.system.kernel|kernel",re.I),
}


def _property_type(topic: str) -> str | None:
    if not isinstance(topic,str):
        return None
    topic=topic.casefold()
    if any(value in topic for value in ("browser-engine","browser engine",
                                        "rendering engine","layout engine")):
        return "engine"
    if "entwickler" in topic or "schöpfer" in topic or "creator" in topic:
        return "creator"
    if "kernel" in topic:
        return "kernel"
    return None


_INVALID_PROPERTY_VALUES = frozenset({
    "same", "identical", "unknown", "n/a", "none",
    "javascript", "open standards", "browser", "rendering",
    "engine", "layout engine", "language", "developer",
})
_HISTORICAL_ENGINE = re.compile(
    r"\b(?:initially|originally|formerly|previously|historically|"
    r"used to|früher|ursprünglich|prior to|in the past)\b", re.I)
_PERSON = re.compile(
    r"[A-ZÀ-Ý][a-zà-ÿ]{2,}(?:\s+(?:van|von|de|da|der|dos))?"
    r"\s+[A-ZÀ-Ý][a-zà-ÿ]{2,}(?:\s+[A-ZÀ-Ý][a-zà-ÿ]{2,})?"
)
_ENGINE_VALUE = re.compile(
    r"\b([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9]*(?:-[A-Za-z0-9]+)?)\s+"
    r"(?:(?:rendering|browser|layout)\s+)?engine\b",re.I)
_ENGINE_VERB = re.compile(
    r"\b(?:uses|utilizes|runs on|basiert auf|nutzt|verwendet|"
    r"powered by|is built on)\s+(?:(?:the|die|der|das|den|dem|einen|eine|ein)\s+)?"
    r"([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9-]+)\b",re.I)
_CREATOR_BY = re.compile(
    r"\b(?:created|developed|designed|invented|authored)\s+by\s+|"
    r"\b(?:entwickelt|erschaffen|entworfen)\s+von\s+",re.I)


def verify_web_fact_proposal(proposal, page_text, entity, prop):
    """Validate Helios' proposed fact against a FULL literal read-page sentence.

    Presence in text alone is insufficient: the sentence must actually tie
    this subject to the requested property. Partial tokens, historical
    browser engines, secondary names, and generic values are rejected.
    """
    from neral_page_reader import evidence_sentence_windows
    if (not isinstance(proposal,dict) or
            set(proposal)!={"entity","fact","evidence_sentence"} or
            proposal["entity"]!=entity or type(proposal["fact"]) is not str or
            type(proposal["evidence_sentence"]) is not str):
        return None
    fact=proposal["fact"].strip()
    sentence=proposal["evidence_sentence"].strip()
    if (not 2<=len(fact)<=85 or not 20<=len(sentence)<=650 or
            fact.casefold() in _INVALID_PROPERTY_VALUES or
            any(c in fact for c in "\r\n\t<>[]{}")):
        return None
    # Reject prefixes such as "Guido van Ros" within "Guido van Rossum".
    literal=re.search(r"(?<!\w)"+re.escape(fact)+r"(?!\w)",sentence,re.I)
    if literal is None:
        return None
    windows=evidence_sentence_windows(page_text,entity,prop,max_windows=40)
    if not any(w["sentence"]==sentence for w in windows):
        return None
    if re.search(r"(?<!\w)"+re.escape(entity)+r"(?!\w)",sentence,re.I) is None:
        return None
    if prop=="creator":
        if _PERSON.fullmatch(fact) is None:
            return None
        rel=_CREATOR_BY.search(sentence)
        if rel is None:
            return None
        # A creator must follow the grammatical relation, not appear
        # somewhere else in an unrelated historical passage.
        tail=sentence[rel.end():]
        who=_PERSON.search(tail)
        if who is None or who.group(0).casefold()!=fact.casefold():
            return None
    elif prop=="engine":
        if _HISTORICAL_ENGINE.search(sentence):
            return None
        if fact.casefold() in {"the","a","an","die","der","das","den","dem","ein","eine","einen",
                               "engine","browser","rendering","javascript"}:
            return None
        named=_ENGINE_VALUE.search(sentence)
        verbal=_ENGINE_VERB.search(sentence)
        # Facts may follow "rendering engine is X", or
        # "X is the rendering engine used by [entity]".
        reverse=re.search(
            r"\b(?:browser|rendering|layout)\s+engine\s+(?:is|was|:)\s+"
            r"([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9-]+)\b",sentence,re.I)
        indirect=re.search(
            r"\b(?:relies on|employs|features)\s+(?:the\s+)?"
            r"([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9-]+)\s+"
            r"(?:layout|browser|rendering)\s+(?:implementation|technology)\b",
            sentence,re.I)
        candidates=[m.group(1) for m in (named,verbal,reverse,indirect) if m]
        if not any(candidate.casefold()==fact.casefold() for candidate in candidates):
            return None
    elif prop=="kernel":
        if (not fact.casefold().endswith("kernel") or
                "kernel" not in sentence.casefold()):
            return None
        if re.search(r"(?<!\w)"+re.escape(fact)+r"(?!\w)",sentence,re.I) is None:
            return None
    else:
        return None
    return fact,sentence



def _property_evidence(text: str, entity: str, prop: str, *, platform: str = "desktop") -> tuple[str, str] | None:
    """Extract a subject-property relation from the fetched page, not co-occurrence.

    A page containing the subject name and a different product's engine or
    creator is not evidence for the requested subject. Extract only explicit
    grammatical relations or a labelled engine infobox on the subject page.
    """
    from neral_page_reader import evidence_sentence_windows
    if platform not in ("desktop", "ios", "android"):
        return None
    if not isinstance(text, str) or not isinstance(entity, str) or not entity.strip():
        return None

    subject = r"(?<!\w)" + re.escape(entity) + r"(?!\w)"
    if re.search(subject, text, re.I) is None:
        return None

    # Firefox's primary iOS documentation explicitly says it does NOT use
    # Gecko, followed immediately by why WKWebKit renders the web. Bind both
    # contiguous source lines, never the negated Gecko mentioned first.
    if prop == "engine" and platform == "ios":
        relation = re.search(
            subject + r"[^\n]{0,110}\bios\b[^\n]{0,130}\n"
            r"(?=[^\n]{0,145}\b(?:Apple|restrictions|browser)\b)"
            r"[^\n]{0,200}\bWKWebKit\b[^\n]{0,140}",text,re.I)
        if relation and len(relation.group(0)) <= 650:
            return "WKWebKit", relation.group(0)

    # German-language original Chromium documentation attributes the same
    # named rendering engine to the explicit list of affected browsers.
    # An item outside the quoted list is NEVER inferred from Chromium alone.
    if prop == "engine" and platform != "ios":
        named_list = re.search(
            r"\b([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9-]+)\s+dient\s+als\s+"
            r"Rendering[- ]Engine\s+für\s+Chromium-basierte\s+Browser\s+wie\s+"
            r"([^.!?\n]{4,245})",text,re.I)
        if named_list and (entity.casefold()=="chromium" or re.search(
                subject,named_list.group(2),re.I)):
            token=named_list.group(1)
            if token.casefold() not in {"the","also","new","some","other","browser","engine"}:
                return token,named_list.group(0)

    # A directly attributed localized Chrome/other browser claim: the
    # platform qualifies the *same grammatical sentence*, not another entity.
    if prop == "engine" and platform == "ios":
        german_ios = re.search(
            r"\b(?:unter|auf)\s+iOS(?:\s+und\s+iPadOS)?[^.!?\n]{0,110}"
            r"\b(?:verwendet|nutzt)\s+" + subject +
            r"\s+(WKWebKit|WebKit)\s+als\s+Rendering[- ]Engine\b[^.!?\n]*[.]",
            text, re.I)
        if german_ios:
            return german_ios.group(1), german_ios.group(0)
        # Two *adjacent literal paragraphs* in an official company iOS
        # article can establish its currently shipped engine: the first
        # identifies the released product, the second explicitly says that
        # browser apps distributed on iOS use Apple WebKit. The raw combined
        # excerpt is emitted without editing and has a strict quote limit.
        company_ios = re.search(
            r"(?im)^([^\n]{0,280}\b" + re.escape(entity) +
            r"\s+(?:for|on)\s+iOS[^\n]{0,280})\n"
            r"([^\n]{0,525}\bevery\s+app\s+that\s+browses\s+the\s+web"
            r"\s+uses\s+Apple\s+WebKit[^\n]{0,320})",
            text, re.I)
        if company_ios and len(company_ios.group(0))<=650:
            return "WebKit", company_ios.group(0)

    if prop == "engine" and platform == "desktop":
        german_desktop = re.search(
            r"\b(?:unter|auf)\s+Desktop[^.!?\n]{0,75}\b"
            r"(?:verwendet|nutzt)\s+" + subject +
            r"\s+([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9-]+)\b[^.!?\n]*[.]",
            text, re.I)
        if german_desktop:
            candidate = german_desktop.group(1)
            if candidate.casefold() not in {"also", "browser", "engine", "the", "same"}:
                return candidate, german_desktop.group(0)

    # Parsed infobox cells are literal page text. In a result page they can be
    # detached from the subject heading by newlines, unlike normal prose.
    if prop == "engine":
        for match in re.finditer(
            r"(?im)^\s*\|\s*Engines?\s*\|\s*([A-Za-z][\w-]+)[^\n]*",
            text):
            token = match.group(1)
            raw = match.group(0)
            if platform == "ios":
                scope = re.search(r"\b(?:WKWebKit|WebKit)\b[^\n]{0,40}\bios\b",raw,re.I)
                if scope:
                    return re.search(r"(?:WKWebKit|WebKit)",scope.group(0),re.I).group(0),raw.strip()[:600]
                continue
            if token.casefold() not in {"the", "a", "unknown", "none", "javascript"}:
                return token, raw.strip()[:600]

    # Keep the literal fetched source spans intact for quoted citations.
    windows = [row["sentence"] for row in
               evidence_sentence_windows(text, entity, prop, max_windows=60)]
    fragments = []
    for line in text.splitlines():
        pieces = [part.strip() for part in re.findall(r"[^.!?]+[.!?]", line)]
        for n in (1, 2, 3):
            for j in range(len(pieces) - n + 1):
                phrase = " ".join(pieces[j:j+n])
                if len(phrase) <= 650:
                    fragments.append(phrase)
        if len(line.strip()) <= 650:
            fragments.append(line.strip())
    spans = list(dict.fromkeys(windows + fragments))
    person = r"([A-ZÀ-Ý][a-zà-ÿ]{2,}(?:\s+(?:van|von|de|da|der|dos))?\s+[A-ZÀ-Ý][a-zà-ÿ]{2,}(?:\s+[A-ZÀ-Ý][a-zà-ÿ]{2,})?)"

    def proper_name(value):
        value = re.sub(r'\s*["“][^"”]{1,30}["”]\s*', " ", value)
        return " ".join(value.split())

    for span in spans:
        if len(span) > 650 or len(span) < 18:
            continue
        if not span.rstrip().endswith((".", "!", "?")) and "the creator of" not in span.casefold():
            continue
        has_subject = bool(re.search(subject, span, re.I))
        if not has_subject:
            continue
        if prop == "engine":
            # Never join an unrelated product in a following sentence with a
            # subject in the preceding sentence (Firefox ... Chrome uses Blink).
            if len(re.findall(r"[.!?](?=\s|$)",span)) > 1:
                continue
            if _HISTORICAL_ENGINE.search(span):
                continue
            # iOS-specific browser snippets must not override desktop Gecko/Blink.
            ios_marked = re.search(r"\b(?:ios|ipados|iphone|ipad|wkwebkit)\b",span,re.I)
            desktop_marked = re.search(r"\b(?:desktop|windows|macos|linux)\b",span,re.I)
            if platform != "ios" and ios_marked and not desktop_marked:
                continue
            if platform == "ios" and not ios_marked:
                continue
            patterns = (
                subject + r".{0,130}?\b(?:uses|utilizes|runs on|nutzt|verwendet)\s+(?:the\s+|die\s+|der\s+|das\s+)?([A-Za-z][\w-]+)\s+(?:rendering|layout|browser)[ -]engine\b",
                subject + r".{0,150}?\bbrowser\s+with\s+([A-Za-z][\w-]+)\s+and\s+[A-Za-z0-9-]+\s+engines\b",
                subject + r".{0,90}?\b(?:rendering|layout|browser)[ -]engine\s+(?:is|:)\s+([A-Za-z][\w-]+)",
                # An explicit product foundation described alongside a named
                # fork engine, not a loose occurrence of "browser engine".
                subject + r".{0,130}?\bbased\s+on(?:\[\d+\])?\s+([A-Za-z][\w-]+).{0,90}?\bfork\s+of\s+(?:the\s+)?[A-Za-z0-9-]+\s+browser\s+engine",
            )
            for pattern in patterns:
                match = re.search(pattern, span, re.I)
                if match:
                    fact = match.group(1)
                    if fact.casefold() not in {
                        "the","a","an","die","der","das","den","dem","ein",
                        "eine","einen","neu","new","proprietary","chrome",
                        "engine","browser","rendering","javascript","also", "own",
                        "always","currently","original","modern", "its", "their",
                        "another", "different", "same", "other"} and fact.casefold() != entity.casefold():
                        return fact, span
            # Reverse grammatical list: the engine is subject of the sentence,
            # and the browser is an explicit list member. Never infer from
            # mere co-occurrence in unrelated headings or separate sentences.
            reverse = re.search(
                r"\b([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9-]{2,})\s+serves\s+as\s+the\s+"
                r"(?:browser\s+|layout\s+)?rendering\s+engine\s+for\b"
                r"[^.!?]{0,230}",span,re.I)
            if reverse:
                subject_list=reverse.group(0)
                if (re.search(r"\b(?:including|such as|like)\b[^.!?]*"+subject,subject_list,re.I)
                        or (entity.casefold()=="chromium" and
                            re.search(r"\bchromium-based\b",subject_list,re.I))):
                    fact=reverse.group(1)
                    if fact.casefold() not in {"the","browser","rendering","engine"}:
                        return fact, span
            # Single-sentence, explicit product foundation relation (not a
            # loose third-party engine mentioned somewhere on the webpage).
            attached=re.search(
                subject+r"[^.!?;]{0,110}?\b(?:built\s+(?:on\s+top\s+of|on|upon)|"
                r"based\s+on|powered\s+by)\b[^.!?;]{0,120}?\b"
                r"([A-ZÀ-Ý][A-Za-zÀ-ÿ0-9-]+)\s+rendering\s+engine\b",span,re.I)
            if attached:
                fact=attached.group(1)
                if fact.casefold() not in {"also","own","another","same","other","the"}:
                    if platform!="ios" or fact.casefold() in {"webkit","wkwebkit"}:
                        return fact, span
            # Domain-specific engine names appear as 'Firefox / Gecko'
            # or 'WKWebKit' without the literal following word "engine".
            special=re.search(
                subject+r"[^.!?;]{0,110}?\b(?:built\s+(?:on\s+top\s+of|on|upon)|"
                r"based\s+on|powered\s+by|uses?)\b[^.!?;]{0,120}?\b"
                r"(WKWebKit|WebKit|Gecko|Blink)\b",span,re.I)
            if special:
                prefix=span[:special.start(1)]
                # In 'Brave mentions that Safari uses WebKit' Safari is the
                # grammatical engine subject. Such third-party quotations
                # cannot establish an engine for Brave.
                if re.search(r"\b(?:mentions|claims|reports|states|says|while|"
                             r"whereas|unlike|compared\s+to)\b",prefix,re.I):
                    continue
                if re.search(r"\b(?:doesn[`’']?t|don[`’']?t|does\s+not|do\s+not|"
                             r"not|never|without)\s+(?:use|uses|using)\s*$",prefix,re.I):
                    continue
                fact=special.group(1)
                if platform == "ios" and fact.casefold() not in {"webkit","wkwebkit"}:
                    continue
                return fact,span
        elif prop == "creator":
            # Matching the creator of "language specifications" or a language
            # mentioned only as an influence is a false attribution.
            forms = (
                subject + r"[^.;:!?]{0,90}?\b(?:was|is|has been)\s+(?:initially\s+)?(?:conceived|created|developed|designed|invented)\b.{0,85}?\bby\s+" + person,
                person + r"\s+(?:created|developed|designed|invented)\s+" + subject,
                r"\b(?:[Cc]reated|[Dd]eveloped|[Dd]esigned|[Ii]nvented)\s+by\s+" + person + r".{0,90}?" + subject,
                person + r"\s*,\s*the\s+creator\s+of\s+" + subject,
                subject + r".{0,160}?\b[Cc]reated\s+by\s+" + person,
                subject + r".{0,160}?\bDevelopment\s+of\s+the\s+language\s+began\b.{0,90}?\bby\s+" +
                r"([A-ZÀ-Ý][a-zà-ÿ]{2,}\s+(?:[\"“][^\"”]+[\"”]\s+)?[A-ZÀ-Ý][a-zà-ÿ]{2,})",
            )
            for pattern in forms:
                match = re.search(pattern, span)
                if match:
                    fact = proper_name(match.group(1))
                    if _PERSON.fullmatch(fact) is not None:
                        return fact, span
        elif prop == "kernel":
            # The kernel name must be present literally next to this subject;
            # generic phrases such as "in-kernel" are not a kernel identity.
            match = re.search(subject + r"[- ]kernel\b", span, re.I)
            if match:
                fact = entity + "-Kernel"
                return fact, span
    return None


def property_grounded_web_answer(entities: tuple[str,...], sources: list[dict], topic: str, *, proposer=None, platform="desktop") -> dict | None:
    """Build property-specific comparison text only from actually read pages.

    One literal HTTPS quote and URL for each entity, plus an independently
    extracted property. Missing proof => return None, not a plausible guess.
    """
    prop=_property_type(topic)
    if prop is None or not isinstance(sources,list) or not 2<=len(entities)<=4:
        return None
    found={}
    for entity in entities:
        for i,page in enumerate(sources,1):
            if (not isinstance(page,dict) or page.get("page_read") is not True
                    or not isinstance(page.get("url"),str)
                    or not page["url"].startswith("https://")):
                continue
            focus=page.get("entity_focus")
            # Fast-path V4 downloads one *fresh* primary page per subject.
            # Never borrow another subject's page even if it incidentally
            # mentions this browser and a different rendering engine.
            # The older unfocused comparison path remains unchanged.
            if focus is not None and (not isinstance(focus,str) or
                                      focus.casefold()!=entity.casefold()):
                continue
            page_text=page.get("page_text","")
            parsed=_property_evidence(page_text,entity,prop,platform=platform)
            if parsed is None and callable(proposer):
                from neral_page_reader import evidence_sentence_windows
                windows=evidence_sentence_windows(page_text,entity,prop)
                if windows:
                    try:
                        proposal=proposer(entity,prop,page_text,windows)
                        parsed=verify_web_fact_proposal(proposal,page_text,entity,prop)
                    except (ValueError,TypeError,RuntimeError,KeyError):
                        parsed=None
            if parsed is None:
                continue
            fact,quote=parsed
            found[entity]={"fact":fact,"quote":quote,"source":f"web{i}","url":page["url"]}
            break
    if len(found)!=len(entities):
        return None
    body="\n".join(f"{entity}: {found[entity]['fact']}" for entity in entities)
    quotes="\n".join(f"[{found[entity]['source']}] „{found[entity]['quote']}“" for entity in entities)
    return {"text":body+"\n\n"+quotes,"tool_used":"direct_reasoning",
            "tool_verified":True,"agent_checked":True,"verified":False,
            "safe_to_promote":False,"error_code":None,
            "verification_status":"read_page_literal_property_extracted",
            "verification_scope":"per_entity_url_and_literal_property_quote",
            "proof_backend":None,
            "agent_steps":["web_search","page_reader",
                           "per_entity_property_extraction","literal_source_proof"],
            "agent_attempts":1,
            "sources":[{k:v for k,v in p.items() if k!="page_text"} for p in sources]}
