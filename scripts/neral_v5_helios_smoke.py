"""Read-only real NERAL-v5 Helios adapter smoke on the owner's Windows GPU.

No downloads, training, project reads, browser access or raw-response upload.
The same HFAdapterLocalBackend is used by Helios/agent. A single fixed,
non-personal prompt is run; only scalar evidence leaves the Windows runner.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import time

from neral_hf_identity import BASE_REVISION


def make_report(backend, response: str, duration: float) -> dict:
    if (getattr(backend, "_model", None) is None or
            not isinstance(getattr(backend._model, "peft_config", None), dict) or
            not backend._model.peft_config):
        raise ValueError("A real PEFT model was not confirmed loaded")
    digest = getattr(backend, "adapter_sha256", None)
    if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}",digest) is None:
        raise ValueError("Unverified local adapter hash")
    if (not isinstance(response, str) or not response.strip() or
            len(response) > 1024 or type(duration) not in (int,float) or
            not 0 <= duration <= 2400):
        raise ValueError("No valid bounded local response")
    return {
        "schema": 1,
        "status": "NERAL_V5_HELIOS_LOCAL_ADAPTER_LOADED",
        "foundation": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "base_revision": BASE_REVISION,
        "adapter_sha256": digest,
        "adapter_loaded": True,
        "model_origin": "sealed_local_neral_lora_experimental",
        "response_nonempty": True,
        "response_length": len(response),
        "elapsed_seconds": round(duration, 2),
        "release_validated": False,
        "general_improvement_proven": False,
        "training_performed": False,
        "gpu_inference_performed": True,
    }


def write_report(path: Path, result: dict) -> None:
    path = Path(path)
    if (path.is_symlink() or path.exists() or path.parent.is_symlink() or
            not path.parent.is_dir()):
        raise ValueError("Refuse to overwrite existing or linked report")
    with path.open("x",encoding="utf8") as dest:
        json.dump(result,dest,indent=2,sort_keys=True,allow_nan=False)
        dest.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--base",required=True,type=Path)
    parser.add_argument("--adapter",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args = parser.parse_args(argv)
    try:
        from neral_agent_hf import HFAdapterLocalBackend
        backend = HFAdapterLocalBackend(args.base,args.adapter,max_new_tokens=64)
        started = time.monotonic()
        reply = backend.respond([
            {"role":"system","content":"Du bist NERAL. Antworte in einem kurzen Satz."},
            {"role":"user","content":"Schreibe einen sehr kurzen deutschen Begrüßungssatz."},
        ])
        elapsed = time.monotonic()-started
        result = make_report(backend,reply,elapsed)
        write_report(args.out,result)
        print("NERAL_V5_HELIOS_LOCAL_ADAPTER_LOADED; only scalar proof saved",flush=True)
        return 0
    except Exception as exc:
        # Never print raw prompts/response/model paths in runner logs.
        print("NERAL_V5_HELIOS_SMOKE_FAILED="+type(exc).__name__,flush=True)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
