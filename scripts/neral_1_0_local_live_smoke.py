"""Live NERAL 1.0 release smoke against a REAL installed local Ollama model.

No server is bound, no user files are changed, no remote inference, no
training or model downloads. Exercises actual Flask HTTP routes in memory
with the real Ollama-backed code. Public report is scalar only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path

from neral_agent_autonomy import OllamaLocalBackend
from neral_preview_server import create_preview_app
from scripts.neral_v7_foundation_tournament import OllamaLocal

HOST="http://127.0.0.1:9690"
NAME="qwen3:8b"


def valid_high_smoke(status_code: int, payload: object) -> bool:
    """Verify the current HIGH API contract without printing private replies."""
    if status_code != 200 or not isinstance(payload, dict):
        return False
    answer = payload.get("text")
    return (
        payload.get("mode") == "HIGH"
        and payload.get("verified") is True
        and isinstance(answer, str)
        and "42" in answer
        and payload.get("model_origin") == "local_foundation_not_neral_adapter"
        and payload.get("adapter_loaded") is False
    )


def verify_live_text_attachment(client, csrf: str) -> bool:
    """Exercise actual file extraction AND local model reading it, never HTTP alone.

    The codeword occurs only in the uploaded BytesIO file, not in the
    question. Responses/prompts are never written to reports or logs.
    """
    reply = client.post(
        "/api/neral/preview/attachment-chat",
        base_url=HOST,
        headers={"X-Neral-CSRF": csrf},
        content_type="multipart/form-data",
        data={
            "prompt": "Welches Prüfwort steht in der beigefügten Datei? Gib nur dieses Prüfwort an.",
            "mode": "generate",
            "history": "[]",
            "files": [(BytesIO(b"Pruefwort: RAUTE731"), "neral-check.txt")],
        },
    )
    data = reply.get_json(silent=True)
    if (
        reply.status_code != 200
        or not isinstance(data, dict)
        or not isinstance(data.get("text"), str)
        or "RAUTE731" not in data["text"].upper()
        or data.get("mode") != "LOW"
        or data.get("model_generated") is not True
        or data.get("model_origin") != "local_foundation_not_neral_adapter"
        or data.get("adapter_loaded") is not False
        or data.get("file_bytes_persisted") is not False
        or data.get("vision_models") != []
        or not isinstance(data.get("attachments"), list)
        or len(data["attachments"]) != 1
        or data["attachments"][0].get("name") != "neral-check.txt"
        or data["attachments"][0].get("kind") != "document"
    ):
        raise ValueError("REAL local attachment context/model acceptance failed")
    return True


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
    if not valid_high_smoke(deep.status_code, deep.get_json(silent=True)):
        raise ValueError("REAL Qwen HIGH/exact-local-tool smoke failed")
    attachment_actual_inference = verify_live_text_attachment(client, hdr["X-Neral-CSRF"])
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
        "attachment_actual_inference":attachment_actual_inference,
        "exact_numeric_result_checked":True,
        "csrf_and_origin_blocked":True,
        "foundation_provenance_preserved":True,
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
