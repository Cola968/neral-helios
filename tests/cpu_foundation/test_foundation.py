"""Offline CPU contract tests: deterministic witnesses, safety and export isolation."""
import json
from pathlib import Path
import pytest
from neral_cpu_foundation.symbolic import verify_symbolic
from neral_cpu_foundation.graphs import verify_graph
from neral_cpu_foundation.constraints import solve_constraints
from neral_cpu_foundation.memory import LocalMemory
from neral_cpu_foundation.rag import LocalRAG, chunk_text
from neral_cpu_foundation.web import SearchClient, clean_html, safe_public_url
from neral_cpu_foundation.router import choose_route, escalate
from neral_cpu_foundation.training import export_verified, canonical_hash
from neral_cpu_foundation.generate import generate_examples


@pytest.mark.parametrize(("task","result"),[
    ({"operation":"equivalent","left":"(x+1)**2","right":"x**2+2*x+1"}, "0"),
    ({"operation":"simplify","expression":"(x+2)**2-x**2"},"4*x + 4"),
    ({"operation":"differentiate","expression":"x**3+4*x"},"3*x**2 + 4"),
    ({"operation":"integrate","expression":"2*x"},"x**2"),
    ({"operation":"combination","n":8,"k":3},"56"),
])
def test_sympy_ops(task,result):
    actual=verify_symbolic(task)
    assert actual["ok"] and actual["verified"]
    assert actual["result"]==result

def test_sympy_equation_and_system():
    a=verify_symbolic({"operation":"equation","equation":"x+3=10","variables":["x"]})
    b=verify_symbolic({"operation":"system","equations":["x+y=7","x-y=1"],"variables":["x","y"]})
    assert a["verified"] and a["result"]==[{"x":"7"}]
    assert b["verified"] and b["result"]==[{"x":"4","y":"3"}]

@pytest.mark.parametrize("source",["__import__('os').system('id')","x.__class__","[i for i in range(1)]","x**100","lambda x:x","foo(x)"])
def test_sympy_rejects_unsafe_expressions(source):
    assert verify_symbolic({"operation":"simplify","expression":source})["verified"] is False

def test_sympy_detects_non_equivalence():
    assert verify_symbolic({"operation":"equivalent","left":"x+1","right":"x+2"})["verified"] is False

def test_sympy_timeout_validation():
    with pytest.raises(ValueError): verify_symbolic({"operation":"simplify","expression":"x"},timeout=12)

def test_graph_shortest_with_independent_bellman_ford():
    r=verify_graph({"operation":"shortest_path","directed":True,"nodes":["a","b","c"],
       "edges":[["a","b",2],["b","c",2],["a","c",9]],"source":"a","target":"c"})
    assert r["verified"] and r["result"]=={"path":["a","b","c"],"length":4}

def test_graph_coloring_and_bipartite():
    spec={"nodes":[1,2,3],"edges":[[1,2],[2,3]]}
    assert verify_graph({**spec,"operation":"coloring","colors":{"1":0,"2":1,"3":0}})["verified"]
    assert not verify_graph({**spec,"operation":"coloring","colors":{"1":0,"2":0,"3":0}})["verified"]
    assert verify_graph({**spec,"operation":"bipartite"})["result"]

def test_graph_euler_hamilton_cycles():
    spec={"nodes":[0,1,2],"edges":[[0,1],[1,2],[2,0]]}
    for operation in ("cycle","euler","hamilton"):
        res=verify_graph({**spec,"operation":operation})
        assert res["verified"] and res["result"] is True

def test_graph_flow_cut_certificate():
    res=verify_graph({"operation":"max_flow","directed":True,"nodes":["s","a","t"],
       "edges":[["s","a",5],["a","t",3],["s","t",1]],"source":"s","target":"t"})
    assert res["verified"] and res["result"]=={"flow":4,"min_cut":4}

@pytest.mark.parametrize("edges",[[[1,2,-1]],[[1,99,2]],[[1,2,1],[1,2,2]]])
def test_graph_rejects_bad_edges(edges):
    with pytest.raises(ValueError): verify_graph({"operation":"cycle","nodes":[1,2], "edges":edges})

def test_graph_hamilton_limit_is_unproven():
    assert not verify_graph({"operation":"hamilton","nodes":list(range(11)),"edges":[]})["verified"]

def test_z3_schedule_optimal():
    r=solve_constraints({"template":"schedule","slots":4,"tasks":[
       {"id":"a","resource":"lab","preferred":0},
       {"id":"b","resource":"lab","preferred":0}]})
    assert r["verified"] and r["status"]=="optimal" and r["cost"]==1
    assert r["result"]["a"]!=r["result"]["b"]

def test_z3_schedule_unsat():
    r=solve_constraints({"template":"schedule","slots":1,"tasks":[{"id":"a","resource":"R"},{"id":"b","resource":"R"}]})
    assert r["verified"] and r["status"]=="unsat"

def test_z3_resources():
    r=solve_constraints({"template":"resources","capacity":5,"demands":[
      {"need":4,"weight":2},{"need":4,"weight":1}]})
    assert r["verified"] and sum(r["result"].values())==5

def test_z3_integer_unique_and_multistage():
    r=solve_constraints({"template":"integer_puzzle","variables":["a","b"],"lower":0,"upper":9,
      "constraints":[{"type":"sum_eq","vars":["a","b"],"value":7},
        {"type":"difference_eq","vars":["a","b"],"value":3}]})
    assert r["verified"] and r["unique"] and r["result"]=={"a":5,"b":2}

def test_z3_modulo_digit_sum():
    r=solve_constraints({"template":"integer_puzzle","variables":["code"],"lower":10,"upper":99,
       "constraints":[{"type":"mod_eq","vars":["code"],"modulus":10,"value":2},
       {"type":"digit_sum_eq","vars":["code"],"value":7}]})
    assert r["verified"] and r["unique"] and r["result"]=={"code":52}

def test_z3_rejects_unknown_clause():
    with pytest.raises(ValueError):
        solve_constraints({"template":"integer_puzzle","variables":["x"],"constraints":[{"type":"execute","vars":["x"]}]})

def test_memory_consent_and_conflict(tmp_path):
    db=LocalMemory(tmp_path/"facts.db")
    with pytest.raises(PermissionError): db.upsert("user","theme","dark")
    assert db.upsert("user","theme","dark",consent=True,timestamp="2026-11-15T10:00:00+00:00")
    assert not db.upsert("user","theme","light",consent=True,timestamp="2026-11-14T10:00:00+00:00")
    assert db.list()[0]["value"]=="dark"
    assert "dark" in db.context("theme",max_tokens=60)
    assert db.forget("user","theme") and db.list()==[]
    db.close()

def test_memory_project_isolation_and_budget(tmp_path):
    db=LocalMemory(tmp_path/"facts.db")
    db.upsert("project-A","plan","alpha",kind="project",consent=True)
    db.upsert("project-B","plan","beta",kind="project",consent=True)
    assert "beta" not in db.context("plan",scope="project-A")
    assert len(db.context("plan",scope="project-A",max_tokens=10))<=30
    db.close()

def test_rag_hybrid_and_citations(tmp_path):
    rag=LocalRAG(tmp_path/"docs.db")
    rag.add("doc-a","Das Magnetfeld zeigt eine Lorentzkraft auf bewegte Ladungen. "*8,
            source="file:///lehre/physik.md",title="Physik")
    result=rag.search("Lorentzkraft")
    assert result and result[0]["citation"].startswith("[doc-a#c")
    assert result[0]["source"]=="file:///lehre/physik.md"
    rag.remove("doc-a")
    assert rag.search("Lorentzkraft")==[]
    rag.close()

def test_rag_vector_interface(tmp_path):
    fn=lambda value: [1.,0.] if "physik" in value.lower() else [0.,1.]
    rag=LocalRAG(tmp_path/"rag.db",embedding_fn=fn)
    rag.add("d1","Physik "*120,source="doc://1")
    assert rag.search("physik")[0]["doc_id"]=="d1"
    rag.close()

def test_rag_chunk_bounds():
    chunks=chunk_text("abcde"*300,size=180,overlap=30)
    assert len(chunks)>2 and all(0<len(v)<=180 for v in chunks)

def test_web_offline_fixtures():
    fixture={"results":[{"title":"<b>Wissenschaft</b>","url":"https://example.org/a","content":"<script>x</script><p>Forschung</p>"},
       {"title":"Private","url":"http://localhost/admin","content":"Bad"}]}
    client=SearchClient("searxng","https://search.example.org/search",transport=lambda url,body,timeout: fixture)
    results=client.search("test")
    assert len(results)==1 and results[0].title=="Wissenschaft" and results[0].snippet=="Forschung"
    assert results[0].citation=="[S1]"

@pytest.mark.parametrize(("provider","fixture"),[
   ("tavily",{"results":[{"title":"Hallo","url":"https://example.com/a","content":"Test"}]}),
   ("duckduckgo",{"RelatedTopics":[{"Text":"Hallo","FirstURL":"https://example.com/a"}]})
])
def test_web_provider_adapters(provider,fixture):
    client=SearchClient(provider,"https://example.com/api",api_key="mock",transport=lambda *args: fixture)
    assert client.search("Frage")[0].url=="https://example.com/a"

@pytest.mark.parametrize("url",["http://x.de","https://127.0.0.1/a","https://user:pass@a.de","https://192.168.0.1","https://localhost"])
def test_web_rejects_unsafe_source(url):
    assert not safe_public_url(url)

def test_html_sanitization():
    assert clean_html("<style>secret</style><p>Hallo <b>Welt</b></p>")=="Hallo Welt"

@pytest.mark.parametrize(("prompt","tier"),[
   ("Hallo, guten Tag","Luna"),
   ("Was steht in meiner PDF-Datei?","Terra"),
   ("Was ist heute in Berlin aktuell?","Terra"),
   ("Beweise die optimale Lösung mit Z3","Helios"),
])
def test_routing(prompt,tier):
    assert choose_route(prompt).tier==tier

def test_router_escalates_uncertainty_and_tool_failure():
    start=choose_route("Hallo")
    assert escalate(start,uncertain=True).tier=="Terra"
    assert escalate(start,tool_failed=True).tier=="Helios"
    assert choose_route("Hallo",force="Luna",uncertain=True).tier=="Terra"

def test_export_reverifies_ignores_self_attested_and_ast_repair(tmp_path):
    good={"prompt":"Wieviele Kombinationen?","task":{"tool":"sympy","spec":{"operation":"combination","n":8,"k":2}},
          "answer":"28","verified":True}
    evil={**good,"answer":"29","verified":True}
    fake={"prompt":"Skript","task":{"tool":"python_sandbox","spec":{"code":"print(1)"}},
          "answer":1,"verified":True,"ast_repaired":True,"exit_code":0}
    result=export_verified([good,good,evil,fake],tmp_path)
    assert result["verified"]==1 and result["rejected"]==3
    assert sum(result["splits"].values())==1
    assert (tmp_path/"helios_terra_v2.jsonl").exists() and (tmp_path/"luna_distill_v1.jsonl").exists()

def test_hash_is_order_independent():
    a={"tool":"sympy","spec":{"n":8,"k":2}}
    b={"spec":{"k":2,"n":8},"tool":"sympy"}
    assert canonical_hash(a)==canonical_hash(b)

def test_generated_corpus_minimum():
    rows=generate_examples()
    assert len(rows)>=250
    assert len({canonical_hash(r["task"]) for r in rows})==len(rows)
    assert all(r["prompt"].strip() and r["answer"] is not None for r in rows)
