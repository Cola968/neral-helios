"""Read-only actual local RTX 4070 LOW -> HIGH -> LOW opt-in acceptance.

Every request travels through the same Flask and Ollama HTTP handlers used by
NERAL 1.0. Nothing in this script downloads or trains a model or uploads
private prompts/responses. The only generated file is a new scalar report.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from scripts.neral_foundation_2048_retest import SampledVram, _read_gpu_memory
from neral_preview_server import create_preview_app, build_experimental_high
from neral_agent_autonomy import OllamaLocalBackend

HOST = "http://127.0.0.1:9690"
STATUS = "NERAL_HIGH_LIVE_RTX4070_OPT_IN_SMOKE_ONLY"


class SmokeError(ValueError):
    """Fail closed; never include private model output in errors."""


def evaluate_response_metadata(mode, model, packet, latency) -> dict:
    if (mode not in ("LOW","HIGH") or
        model not in ("qwen3:8b","qwen3.5:9b") or
        type(packet) is not dict or
        packet.get("mode") != mode or
        packet.get("model") != model or
        packet.get("inference_profile") != (
            "legacy" if mode=="LOW" else "measured_nonthinking_2048") or
        packet.get("model_origin") != "local_foundation_not_neral_adapter" or
        packet.get("adapter_loaded") is not False or
        type(packet.get("model_generated")) is not bool or
        not isinstance(packet.get("text"),str) or
        not packet["text"].strip() or
        type(latency) not in (int,float) or not math.isfinite(latency) or
        not 0 < latency <= 360):
        raise SmokeError("Wrong local foundation, profile, completion or runtime provenance")
    return {
        "mode":mode,"model":model,
        "inference_profile":packet["inference_profile"],
        "generated":packet["model_generated"],
        "seconds":round(latency,3)
    }


def make_report(rows:list[dict],gpu:dict) -> dict:
    if (type(rows) is not list or len(rows)!=4 or
        [(x.get("mode"),x.get("model")) for x in rows] !=
          [("LOW","qwen3:8b"),("HIGH","qwen3.5:9b"),
           ("LOW","qwen3:8b"),("HIGH","qwen3.5:9b")] or
        rows[3].get("generated") is not False or
        any(set(x)!={"mode","model","inference_profile","generated","seconds"}
            for x in rows) or
        type(gpu) is not dict or
        set(gpu)!={"peak_used_mib","total_mib","sample_count"} or
        any(type(gpu[k]) is not int for k in gpu) or
        not 0<gpu["peak_used_mib"]<=gpu["total_mib"] or
        not 10000<=gpu["total_mib"]<=17000 or
        gpu["sample_count"]<3):
        raise SmokeError("Missing model switch, direct-tool or actual GPU telemetry")
    return {
        "schema":1,"status":STATUS,
        "routes":["LOW_generate","HIGH_think","LOW_generate","HIGH_direct_tool"],
        "requests_completed":4,
        "models":["qwen3:8b","qwen3.5:9b"],
        "high_profile":"measured_nonthinking_2048",
        "low_profile":"legacy",
        "ordered_latency_seconds":[x["seconds"] for x in rows],
        "sampled_peak_vram_mib":gpu["peak_used_mib"],
        "sampled_peak_vram_gib":round(gpu["peak_used_mib"]/1024,3),
        "gpu_total_vram_mib":gpu["total_mib"],
        "vram_observations":gpu["sample_count"],
        "gpu_memory_scope":"sampled_gpu_wide_not_process_specific",
        "raw_prompts_or_answers_uploaded":False,
        "training_performed":False,
        "neural_improvement_proven":False,
        "model_promotion_authorized":False,
        "release_validated":False,
        "limitations":"Four real app smoke requests verify model switching and "
                      "provenance, not accuracy or a general quality holdout."
    }


def run(out:Path):
    out=Path(out)
    if (out.exists() or out.is_symlink() or out.suffix!=".json" or
        not out.parent.is_dir() or out.parent.is_symlink() or
        out.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise SmokeError("New private scalar report outside source checkout required")
    used,total=_read_gpu_memory()
    if total-used<7000:
        raise SmokeError("RTX 4070 too busy; do not run model smoke")
    low=OllamaLocalBackend("qwen3:8b",inference_profile="legacy",timeout=90)
    high=build_experimental_high("qwen3:8b",True)
    app=create_preview_app(low,experimental_high_backend=high)
    client=app.test_client()
    if client.get("/",base_url=HOST).status_code!=200:
        raise SmokeError("Local NERAL app cannot initialize")
    session=client.get("/api/neral/preview/session",base_url=HOST)
    if session.status_code!=200 or type(session.json) is not dict:
        raise SmokeError("Local session failed")
    if (session.json.get("model")!="qwen3:8b" or
        session.json.get("experimental_high_model")!="qwen3.5:9b" or
        session.json.get("experimental_high_enabled") is not True):
        raise SmokeError("Session omitted experimental HIGH provenance")
    headers={"X-Neral-CSRF":session.json["csrf"]}
    samples=[
        ("LOW","qwen3:8b","/api/neral/chat/generate",
         "Antworte mit einem Satz: Was ist ein Stern?"),
        ("HIGH","qwen3.5:9b","/api/neral/chat/think",
         "Erkläre kurz, wie sich Sterne von Planeten unterscheiden."),
        ("LOW","qwen3:8b","/api/neral/chat/generate",
         "Antworte mit einem Satz: Wozu dienen Satelliten?"),
        ("HIGH","qwen3.5:9b","/api/neral/chat/think",
         "Calculate 17*13"),
    ]
    rows=[]
    try:
        with SampledVram() as sampler:
            for mode,model,url,question in samples:
                start=time.monotonic()
                r=client.post(url,json={"prompt":question},
                              headers=headers,base_url=HOST)
                elapsed=time.monotonic()-start
                if r.status_code != 200:
                    raise SmokeError(f"Real local {mode} API failed with HTTP {r.status_code}")
                packet=r.get_json(silent=True)
                rows.append(evaluate_response_metadata(mode,model,packet,elapsed))
                print(f"LOCAL_HIGH_SMOKE_{len(rows)}_OF_4_{mode}_OK",flush=True)
        summary=make_report(rows,sampler.report())
    finally:
        # Always release model allocations; no model or chat deletion.
        high.unload()
        low.unload()
    with out.open("x",encoding="utf8") as fd:
        json.dump(summary,fd,indent=2,sort_keys=True)
        fd.write("\n")
    print(json.dumps(summary,sort_keys=True),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args().out)


if __name__=="__main__":
    main()
