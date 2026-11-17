"""Compare true Luna-v1 PEFT HF nonthinking fast profile against frozen Helios.

This measures only generated-token latency on the same 20 synthetic RAG and
dialogue probes. No training, no raw prompts/answers in the persisted report.
The 96-token fast profile is an explicitly separate inference condition from
earlier 256-token tests; score comparisons must preserve that distinction.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, statistics, time
from collections.abc import Mapping
from pathlib import Path

from neral_cpu_foundation.router import choose_route, generation_profile
from scripts.neral_v2_dialogue_rag_sample import casepack, score
from scripts.neral_v2_final_eval40 import validate_frozen
from scripts.neral_v2_eval_checkpoint import model_sha


def fast_luna_profile():
    route = choose_route("Hallo")
    settings = generation_profile(route)
    if (route.tier != "Luna" or route.think is not False
        or route.adapter_id != "luna-v1-4b-peft"
        or route.max_new_tokens != 96
        or settings.get("think") is not False
        or settings["options"]["num_predict"] != 96):
        raise RuntimeError("Luna router is not in a guaranteed nonthinking 96-token mode")
    return settings


def evaluate(label: str, base_path: Path, adapter_path: Path | None,
             probes: list[dict], max_new: int) -> dict:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(str(base_path), local_files_only=True, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = Qwen3_5ForCausalLM.from_pretrained(
        str(base_path), local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
        dtype=dtype, device_map={"":0}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False) if adapter_path else base
    model.eval()
    times=[]; generated=[]; passed=0; by_kind={}
    for i, probe in enumerate(probes):
        messages=[{"role":"system","content":probe["system"]},
                  {"role":"user","content":probe["prompt"]}]
        ids=tok.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if isinstance(ids,Mapping):ids=ids["input_ids"]
        if not isinstance(ids,list) or len(ids)>3800:
            raise RuntimeError("Unbounded probe prompt")
        inp=torch.tensor([ids],dtype=torch.long,device="cuda")
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            out=model.generate(input_ids=inp,do_sample=False,
                max_new_tokens=max_new, use_cache=True,
                pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize();duration=time.perf_counter()-started
        answer=tok.decode(out[0,len(ids):],skip_special_tokens=True)
        ok=score(probe,answer)
        passed+=int(ok);times.append(duration)
        generated.append(int(out.shape[1])-len(ids))
        cell=by_kind.setdefault(probe["kind"],{"passed":0,"total":0})
        cell["passed"]+=int(ok);cell["total"]+=1
        print("NERAL_FAST20_PROGRESS "+label+" "+str(i+1)+"/20",flush=True)
        del inp,out
    report={"model":label,"n":20,"passed_automatic_proxy":passed,
            "by_kind":by_kind,
            "median_seconds":round(statistics.median(times),3),
            "median_generated_tokens":statistics.median(generated),
            "generation_max_new_tokens":max_new,
            "thinking_enabled":False,
            "fast_profile_loaded_in_real_hf_inference":True,
            "peak_gpu_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
            "does_not_include_loading_time":True,
            "human_quality_judgement":False}
    print("NERAL_FAST20_SCALAR "+json.dumps(report,sort_keys=True),flush=True)
    del model,base
    gc.collect();torch.cuda.empty_cache()
    return report


def main() -> None:
    ap=argparse.ArgumentParser()
    for key in ("base9","full9","base4","luna4","v1","out"):
        ap.add_argument("--"+key,required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Never overwrite fast-mode comparison")
    validate_frozen(args.full9,args.v1,args.out)
    luna_manifest=json.loads((args.luna4/"manifest.json").read_text("utf8"))
    if (luna_manifest.get("status")!="frozen_adapter_ready_for_heldout_comparison"
        or luna_manifest.get("selected_epoch")!=2
        or luna_manifest.get("rank")!=8
        or luna_manifest.get("train_rows")!=480
        or luna_manifest.get("val_rows")!=75
        or luna_manifest.get("test_rows_opened")!=0
        or luna_manifest.get("frozen_adapter_sha256")!=model_sha(args.luna4/"adapter")):
        raise RuntimeError("Real frozen Luna PEFT adapter not verified")
    from transformers import AutoTokenizer
    from pathlib import Path
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Real RTX4070 is mandatory")
    fast=fast_luna_profile()
    probes=casepack()
    scores={
      "qwen4_base_fast96": evaluate("Qwen4B-base-nonthinking-96",args.base4,None,probes,96),
      "luna4_peft_fast96": evaluate("Luna-v1-4B-PEFT-nonthinking-96",
                                       args.base4,args.luna4/"adapter",probes,96),
      "helios9_peft_fast96": evaluate("Helios-v2-full-9B-PEFT-nonthinking-96",
                                       args.base9,args.full9/"adapter",probes,96),
      "helios9_peft_256": evaluate("Helios-v2-full-9B-PEFT-nonthinking-256",
                                       args.base9,args.full9/"adapter",probes,256),
    }
    report={"status":"real_gpu_fast_20_complete",
            "routing_profile":{"tier":"Luna","base":fast["model"],"think":False,
                              "max_new_tokens":96,"adapter_id":fast["adapter_id"]},
            "same_20_synthetic_probes":True,
            "same_decode_policy":True,
            "same_96_token_ceiling_for_fair_speed_comparison":True,
            "additional_helios_standard_256_only_context":True,
            "no_training":True,"no_sealed_40_read":True,
            "samples_are_synthetic":True,"models":scores}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n","utf8")
    print("NERAL_FAST20_FINAL "+json.dumps({
        k:{"passed":v["passed_automatic_proxy"],"median_s":v["median_seconds"]}
        for k,v in scores.items()},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
