"""Live NERAL 1.0 release smoke against a REAL installed local Ollama model.

No server is bound, no user files are changed, no remote inference, no
training or model downloads. Exercises actual Flask HTTP routes in memory
with the real Ollama-backed code. Public report is scalar only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from neral_agent_autonomy import OllamaLocalBackend
from neral_preview_server import create_preview_app
from scripts.neral_v7_foundation_tournament import OllamaLocal

HOST="http://127.0.0.1:9690"
NAME="qwen3:8b"


def test_local_release(output: Path):
    output=Path(output)
    if (output.exists() or output.is_symlink() or not output.parent.is_dir() or
            output.parent.is_symlink()):
        raise ValueError("Use a fresh local scalar output file")
    installed=OllamaLocal().inventory()
    digest=installed.get(NAME)
    if not isinstance(digest,str) or not digest.startswith("sha256:"):
        raise ValueError("Pinned local Qwen model not present")
    backend=OllamaLocalBackend(NAME,timeout=90)
    app=create_preview_app(backend)
    client=app.test_client()
    initial=client.get("/",base_url=HOST)
    if initial.status_code!=200:
        raise ValueError("Local preview home route did not respond")
    sess=client.get("/api/neral/preview/session",base_url=HOST)
    if (sess.status_code!=200 or
            sess.json.get("model")!=NAME or
            sess.json.get("release_validated") is not False):
        raise ValueError("Preview session or provenance unsafe")
    hdr={"X-Neral-CSRF":sess.json["csrf"]}
    invalid=client.post("/api/neral/chat/generate",
                       json={"prompt":"test"},base_url=HOST)
    hostile=client.post("/api/neral/chat/generate",
                       json={"prompt":"test"},base_url=HOST,
                       headers={**hdr,"Origin":"https://example.invalid"})
    if invalid.status_code!=403 or hostile.status_code!=403:
        raise ValueError("Local CSRF/Origin guards failed")

    # A direct identity question is answered from real backend metadata, not
    # sampled from Qwen; this must not pretend that a NERAL-LoRA is loaded.
    identity=client.post("/api/neral/chat/generate",base_url=HOST,headers=hdr,
        json={"prompt":"Wer bist du?"})
    if (identity.status_code!=200 or
            identity.json.get("model")!=NAME or
            identity.json.get("model_generated") is not False or
            identity.json.get("tool_used")!="neral_model_provenance" or
            identity.json.get("adapter_loaded") is not False or
            "kein NERAL-LoRA-Adapter" not in identity.json.get("text","")):
        raise ValueError("REAL local foundation identity/provenance smoke failed")

    fast=client.post("/api/neral/chat/generate",base_url=HOST,headers=hdr,
       json={"prompt":"Wie heißt die Hauptstadt von Deutschland? Antworte kurz."})
    if (fast.status_code!=200 or
            not isinstance(fast.json.get("text"),str) or
            "berlin" not in fast.json["text"].lower() or
            fast.json.get("model_origin")!="local_foundation_not_neral_adapter" or
            fast.json.get("adapter_loaded") is not False or
            fast.json.get("quality_improvement_proven") is not False):
        raise ValueError("REAL Qwen FAST local answer/provenance smoke failed")

    deep=client.post("/api/neral/chat/think",base_url=HOST,headers=hdr,
       json={"prompt":"Berechne schrittweise: 6*7"})
    if (deep.status_code!=200 or
            deep.json.get("mode")!="HIGH" or
            deep.json.get("verified") is not True or
            "42" not in str(deep.json.get("text","")) or
            deep.json.get("model_origin")!="local_foundation_not_neral_adapter" or
            deep.json.get("adapter_loaded") is not False):
        raise ValueError("REAL Qwen THINK/exact-local-tool smoke failed")
    # HIGH must really run two independent local thinking generations for an
    # open question. This is a system-depth check, not a quality score.
    reviewed=client.post("/api/neral/chat/think",base_url=HOST,headers=hdr,
       json={"prompt":"Erkläre den Unterschied zwischen einem Stern und einem Planeten."})
    if (reviewed.status_code!=200 or
            reviewed.json.get("mode")!="HIGH" or
            reviewed.json.get("calls_used")!=2 or
            reviewed.json.get("review_passes")!=1 or
            reviewed.json.get("review_status")!="completed_unverified" or
            reviewed.json.get("verified") is not False or
            not isinstance(reviewed.json.get("text"),str) or
            len(reviewed.json["text"])<15 or
            reviewed.json.get("adapter_loaded") is not False or
            reviewed.json.get("model_origin")!="local_foundation_not_neral_adapter"):
        raise ValueError("REAL local HIGH two-pass inference failed")

    if OllamaLocal().inventory().get(NAME)!=digest:
        raise ValueError("Local model SHA drift during live smoke")
    report={
        "schema":1,
        "status":"NERAL_1_0_LOCAL_PC_SMOKE_PASSED",
        "model":NAME,
        "model_digest":digest,
        "local_windows":True,
        "fast_actual_inference":True,
        "think_actual_inference":True,
        "high_two_pass_actual_inference":True,
        "exact_numeric_result_checked":True,
        "csrf_and_origin_blocked":True,
        "foundation_provenance_preserved":True,
        "model_identity_runtime_verified":True,
        "legacy_50m_required":False,
        "raw_prompts_or_responses_included":False,
        "independent_model_intelligence_proven":False,
        "weights_promoted":False,
        "release_validated":False,
    }
    with output.open("x",encoding="utf8") as stream:
        stream.write(json.dumps(report,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    test_local_release(args.out)


if __name__=="__main__":
    main()
