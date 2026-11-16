"""Read-only 1.2 live RTX4070 smoke. No training, writes or web queries."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from neral_preview_server import create_preview_app
from neral_agent_autonomy import OllamaLocalBackend
from scripts.neral_nl_schema_probe import old_case_question


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists():raise SystemExit("EXISTING_REPORT_REJECTED")
    head=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,timeout=12,check=True).stdout.strip()
    if head!=os.environ.get("NERAL_INTEGRATION_SOURCE_SHA"):
        raise SystemExit("PINNED_SHA_MISMATCH")
    from scripts.neral_1_1_prospective_blind100 import require_gpu
    hardware=require_gpu()  # NVIDIA RTX4070, Ollama models, trusted Docker
    backend=OllamaLocalBackend("qwen3.5:9b",timeout=90,
      inference_profile="measured_nonthinking_2048")
    app=create_preview_app(backend,secret_key=os.urandom(32))
    client=app.test_client()
    base="http://127.0.0.1:9690"
    assert client.get("/",base_url=base).status_code==200
    csrf=client.get("/api/neral/preview/session",base_url=base).json["csrf"]
    headers={"X-Neral-CSRF":csrf}
    started=time.monotonic()
    verification=[
      {"tool":"sympy","spec":{"operation":"combination","n":8,"k":3}},
      {"tool":"networkx","spec":{"operation":"bipartite","nodes":[1,2,3],
                                 "edges":[[1,2],[2,3]]}},
      {"tool":"z3","spec":{"template":"integer_puzzle",
            "variables":["a","b"],"lower":0,"upper":9,"constraints":[
            {"type":"sum_eq","vars":["a","b"],"value":7},
            {"type":"difference_eq","vars":["a","b"],"value":3}]}},
    ]
    checked=[]
    for item in verification:
        response=client.post("/api/neral/cpu/verify",json=item,headers=headers,base_url=base)
        if response.status_code!=200 or response.json.get("verified") is not True:
            raise RuntimeError("CPU_TOOL_VERIFY_BLOCKED")
        checked.append(item["tool"])
    # Full-source 1.1 grammar only, never model-authored schema.
    source=old_case_question("logic-02")
    proven=client.post("/api/neral/chat/generate",json={"prompt":source},
      headers=headers,base_url=base)
    if (proven.status_code!=200 or proven.json.get("verified") is not True or
        proven.json.get("automatic_verified_routing") is not True or
        proven.json.get("proof_backend")!="z3_unsat" or proven.json.get("text")!="6"):
        raise RuntimeError("AUTO_Z3_SOURCE_VERIFICATION_FAILED")
    # Exercise the actual local Qwen backend with a non-certified question.
    qwen=client.post("/api/neral/chat/generate",
      json={"prompt":"Erkläre in einem Satz, was eine Galaxie ist."},
      headers=headers,base_url=base)
    if (qwen.status_code!=200 or
        not isinstance(qwen.json.get("text"),str) or
        not qwen.json["text"].strip() or
        qwen.json.get("model_generated") is not True):
        raise RuntimeError("ORDINARY_QWEN_FALLBACK_FAILED")
    summary={"source_sha":head,"status":"PASS",
      "independently_verified_tools":checked,"full_source_z3_verified":True,
      "model_fallback_completed":True,"no_training":True,"no_web":True,
      "no_main_write":True,"elapsed_seconds":round(time.monotonic()-started,2),
      "gpu_type_contains_RTX4070": "RTX 4070" in hardware["gpu"]}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NERAL_1_2_GPU_SMOKE "+json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__":main()
