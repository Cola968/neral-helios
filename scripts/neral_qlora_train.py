"""Opt-in local QLoRA on Qwen3.5-9B; never trains on curated holdout.

Only run manually after CPU dataset checks. RTX 4070 ~12 GiB: rank 8,
4-bit, batch 1, gradient accumulation 8, 2048-token strict cap.
The real Hugging Face tokenizer produces Qwen3.5's native chat template.
Loss is applied ONLY to the assistant target tokens, not to user prompts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.neral_qlora_holdout_eval import load_holdout, assert_disjoint


def validated_training_rows(train_path:Path,holdout_path:Path)->list[dict]:
    if train_path.resolve()==holdout_path.resolve():
        raise ValueError("Training and holdout paths must differ")
    holdout=load_holdout(holdout_path)
    assert_disjoint(holdout,train_path)
    rows=[]
    seen=set()
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        if row["meta"]["split"]!="train" or row["meta"]["source"]!="synthetic_oracle":
            raise ValueError("Refusing non-training or non-oracle records")
        if row["meta"]["mode"] not in ("turn0","turn1"):
            raise ValueError("Invalid training turn")
        messages=row["messages"]
        if (len(messages)!=3 or [m["role"] for m in messages] !=
                ["system","user","assistant"]):
            raise ValueError("Invalid conversation roles")
        answer=json.loads(messages[-1]["content"])
        if answer.get("kind")!="formal" or answer.get("complete") is not True:
            raise ValueError("Invalid assistant schema target")
        key=(row["meta"]["question_sha256"],row["meta"]["mode"])
        if key in seen:
            raise ValueError("Duplicate training mode/question")
        seen.add(key)
        rows.append(row)
    if not 500<=len(rows)<=1000 or len(rows)%2:
        raise ValueError("Expected 500-1000 paired records")
    counts={mode:sum(r["meta"]["mode"]==mode for r in rows)
            for mode in ("turn0","turn1")}
    if counts["turn0"]!=counts["turn1"] or len(seen)!=len(rows):
        raise ValueError("Unbalanced training turns")
    return rows


def encode_answer_only(rows:list[dict],tokenizer, max_seq_length:int)->list[dict]:
    """Always use native Qwen3.5 template; fail rather than silently truncate."""
    encoded=[]
    for row in rows:
        turns=row["messages"]
        try:
            prefix=tokenizer.apply_chat_template(
                turns[:-1],tokenize=True,add_generation_prompt=True,
                enable_thinking=False)
            complete=tokenizer.apply_chat_template(
                turns,tokenize=True,add_generation_prompt=False,
                enable_thinking=False)
        except TypeError as exc:
            raise RuntimeError("Installed Qwen3.5 tokenizer lacks required template flags") from exc
        if not isinstance(prefix,list) or not isinstance(complete,list):
            raise RuntimeError("Tokenizer returned non-token chat structure")
        # Some tokenizers return {"input_ids":...}; require actual list here.
        if complete[:len(prefix)] != prefix:
            raise RuntimeError("Native chat template assistant-prefix mismatch")
        if len(complete)>max_seq_length or len(complete)<=len(prefix):
            raise RuntimeError("Chat sequence exceeds cap or target is missing")
        labels=[-100]*len(prefix)+complete[len(prefix):]
        if not any(x>=0 for x in labels):
            raise RuntimeError("No assistant-only training tokens")
        encoded.append({"input_ids":complete,
                        "attention_mask":[1]*len(complete),
                        "labels":labels})
    return encoded


class FixedTokenDataset:
    def __init__(self,examples:list[dict]):
        self.examples=examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self,index):
        return self.examples[index]


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True,type=Path)
    p.add_argument("--holdout",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--model",default="unsloth/Qwen3.5-9B")
    p.add_argument("--max-steps",type=int,default=60)
    p.add_argument("--max-seq-length",type=int,default=2048)
    p.add_argument("--check-data-only",action="store_true")
    args=p.parse_args()
    rows=validated_training_rows(args.train,args.holdout)
    if args.check_data_only:
        print(json.dumps({"status":"train_data_checked_no_gpu",
                          "rows":len(rows),"holdout_used_for_training":False}))
        return 0
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Refuse overwrite of existing adapter checkpoint")
    if args.max_steps<1 or not 1024<=args.max_seq_length<=2048:
        raise ValueError("Refusing unconstrained training configuration")
    # Import model/trainer only AFTER validating data, labels and holdout.
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Training requires one local CUDA GPU")
    free,total=torch.cuda.mem_get_info()
    if total<11*1024**3 or free<9*1024**3:
        raise RuntimeError("Insufficient free RTX 4070 VRAM; unload local Ollama first")
    # Unsloth's FastModel handles Qwen3.5 vision-capable architecture;
    # text-only data never fine-tunes the vision layers.
    from unsloth import FastModel
    from transformers import Trainer, TrainingArguments
    model,processor=FastModel.from_pretrained(
        model_name=args.model,max_seq_length=args.max_seq_length,
        dtype=None,load_in_4bit=True,full_finetuning=False)
    model=FastModel.get_peft_model(
        model,r=8,lora_alpha=16,lora_dropout=0,bias="none",
        finetune_vision_layers=False,finetune_language_layers=True,
        finetune_attention_modules=True,finetune_mlp_modules=True,
        use_gradient_checkpointing="unsloth",random_state=TRAINING_SEED)
    tokenizer=getattr(processor,"tokenizer",processor)
    encoded=encode_answer_only(rows,tokenizer,args.max_seq_length)

    def collate(batch):
        pad=tokenizer.pad_token_id
        if pad is None:
            raise RuntimeError("Tokenizer has no padding token")
        n=max(len(v["input_ids"]) for v in batch)
        return {
            "input_ids":torch.tensor([v["input_ids"]+[pad]*(n-len(v["input_ids"]))
                                     for v in batch],dtype=torch.long),
            "attention_mask":torch.tensor([v["attention_mask"]+[0]*(n-len(v["attention_mask"]))
                                           for v in batch],dtype=torch.long),
            "labels":torch.tensor([v["labels"]+[-100]*(n-len(v["labels"]))
                                   for v in batch],dtype=torch.long),
        }
    args.out.mkdir(parents=True,exist_ok=False)
    cfg=TrainingArguments(
        output_dir=str(args.out/"trainer"),
        per_device_train_batch_size=1,gradient_accumulation_steps=8,
        max_steps=args.max_steps,learning_rate=1e-4,warmup_steps=5,
        lr_scheduler_type="cosine",weight_decay=0.01,
        logging_steps=10,save_strategy="no",
        bf16=bool(torch.cuda.is_bf16_supported()),fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",gradient_checkpointing=True,
        report_to="none",seed=TRAINING_SEED,
        remove_unused_columns=False)
    trainer=Trainer(model=model,args=cfg,
                    train_dataset=FixedTokenDataset(encoded),
                    data_collator=collate)
    trainer.train()
    model.save_pretrained(str(args.out/"adapters"))
    processor.save_pretrained(str(args.out/"adapters"))
    (args.out/"training_manifest.json").write_text(
        json.dumps({"status":"adapters_saved_not_evaluated",
                    "base_model":args.model,"rows":len(rows),
                    "seed":TRAINING_SEED,"max_steps":args.max_steps,
                    "rank":8,"quantization":"bnb_nf4_4bit",
                    "assistant_only_loss":True,
                    "holdout_used_for_training":False},indent=2)+"\n",
        encoding="utf-8")
    print("QLORA_ADAPTER_SAVED_NO_PERFORMANCE_CLAIM")
    return 0

TRAINING_SEED=2026111201
if __name__=="__main__":
    raise SystemExit(main())
