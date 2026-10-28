# Arctus 350M — Claude Code Review · Teil A: Tokens, Maskierung, Training

**Source:** private `Cola968/neral-helios`, pinned commit `bf9f638f6c45311c175c3f154d7c62a7952b8465` (`arctus350-chat-cloud-prep`). Generated 2026-10-28. **These are excerpts of the ACTUAL ORIGINAL SOURCE FILES, not invented substitutes.** All lines below have added `L### |` markers for review; remove markers before executing code. No private checkpoints, credentials, tokenizer assets or actual training corpus included.

## Audit request

Review why a historical 1,200-step chat SFT pilot reduced validation-answer NLL from **3.163 to 1.736** while the frozen 30-case capability probe declined from **3/30 to 2/30**. Do **not** conclude a bug merely from one-case change: this small suite is noisy. Look for prompt/response format mismatches; masked loss; per-token vs per-batch scaling; EOS targets; label shift; answer truncation; sampler/data-split artifacts; replay/interference; checkpoint selection. Provide filename and line references. No simulated GPU runs, no training and no claims of results. **Arctus is our own scratch-trained ~350M model (over 2 billion historical training tokens), not Qwen.**

## Source overview

| Review priority | File | Role |
|---|---|---|
| P0 | `chat350_contract.py` | Tokenizer lock and model checkpoint identities |
| P0 | `chat350_sft_format.py` | User/assistant serialization, EOS, training labels |
| P0 | `chat_prompt350.py` | Inference history and rendered prompt format |
| P0 | `chat350_data.py` | Data schema and heldout overlap gates |
| P1 | `sft_batching350.py`, `sft_stratified350.py` | Padding, bucketing, human-weighted sampling |
| P0 | `train_chat350.py` | Actual SFT loss/collator/optimizer/validation/selection |
| P1 | `model350.py` | Model loss/forward, tied head, 350M backbone |

---

## 1. `frontier350/chat350_contract.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/chat350_contract.py  
Git blob SHA: `531ebcd3f4f1a4c307ff259d9af895c63e3895d9` · 22 lines.

```text
L001 | from __future__ import annotations
L002 | ARCHITECTURE="frontier350-v1"
L003 | EXPECTED_PARAMS=349_825_152
L004 | EXPECTED_TOKENIZER_SHA256="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
L005 | EXPECTED_SPECIALS={"<|bos|>":0,"<|eos|>":1,"<|sep|>":2,"<|pad|>":3}
L006 | EXPECTED_CONFIG={"vocab_size":16384,"d_model":1152,"n_layers":24,"n_heads":18,"n_kv_heads":2,"ffn_hidden":3136,"max_seq_len":8192,"rope_theta":250000.0,"rms_eps":1e-5,"qk_rms_eps":1e-6,"dropout":0.0,"initializer_range":0.02}
L007 | def validate_tokenizer_lock(lock):
L008 |     if lock.get("architecture")!=ARCHITECTURE: raise ValueError("tokenizer lock architecture mismatch")
L009 |     if int(lock.get("vocab_size",-1))!=16384: raise ValueError("tokenizer vocabulary mismatch")
L010 |     if lock.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256: raise ValueError("tokenizer digest mismatch")
L011 |     if lock.get("special_token_ids")!=EXPECTED_SPECIALS: raise ValueError("special-token mapping mismatch")
L012 |     if lock.get("foreign_vocab_or_merges_loaded") is not False: raise ValueError("foreign tokenizer components are not allowed")
L013 | def checkpoint_processed_tokens(meta):
L014 |     return int(meta.get("processed_tokens",meta.get("source_processed_tokens",meta.get("tokens",0))))
L015 | def validate_checkpoint_metadata(meta,min_processed_tokens=1_000_000_000):
L016 |     if meta.get("config")!=EXPECTED_CONFIG: raise ValueError("checkpoint architecture/config mismatch")
L017 |     if meta.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256: raise ValueError("checkpoint/tokenizer digest mismatch")
L018 |     n=checkpoint_processed_tokens(meta)
L019 |     if n<min_processed_tokens: raise ValueError(f"checkpoint is too early for chat recovery: {n:,} tokens")
L020 |     if "model" not in meta: raise ValueError("checkpoint has no model state")
L021 | def release_identity():
L022 |     return {"architecture":ARCHITECTURE,"parameters":EXPECTED_PARAMS,"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,"special_token_ids":dict(EXPECTED_SPECIALS),"config":dict(EXPECTED_CONFIG),"posttraining_optimizer_policy":"fresh_optimizer_from_immutable_foundation_checkpoint"}
```

## 2. `frontier350/chat350_sft_format.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/chat350_sft_format.py  
Git blob SHA: `93b61092a6e5519270f931d37ac5bcdea3f0b1d1` · 85 lines.

```text
L001 | from __future__ import annotations
L002 | from dataclasses import dataclass
L003 | 
L004 | @dataclass(frozen=True)
L005 | class EncodedChatExample:
L006 |     input_ids: list[int]
L007 |     targets: list[int]
L008 |     assistant_tokens: int
L009 | 
L010 | def _id(tokenizer, token: str) -> int:
L011 |     value = tokenizer.token_to_id(token)
L012 |     if value is None:
L013 |         raise ValueError(f"missing required special token: {token}")
L014 |     return int(value)
L015 | 
L016 | def _encode(tokenizer, text: str) -> list[int]:
L017 |     return list(tokenizer.encode(text).ids)
L018 | 
L019 | def encode_messages(tokenizer, messages: list[dict[str, str]], *, max_seq_len: int) -> EncodedChatExample:
L020 |     if max_seq_len <= 0:
L021 |         raise ValueError("max_seq_len must be positive")
L022 |     if not messages or messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
L023 |         raise ValueError("conversation must start with user and end with assistant")
L024 |     bos = _id(tokenizer, "<|bos|>")
L025 |     eos = _id(tokenizer, "<|eos|>")
L026 |     ids = [bos]
L027 |     supervised = [False]
L028 |     expected = "user"
L029 |     for i, msg in enumerate(messages):
L030 |         role = msg.get("role")
L031 |         content = str(msg.get("content", "")).strip()
L032 |         if role != expected or not content:
L033 |             raise ValueError(f"invalid {expected} turn")
L034 |         if role == "user":
L035 |             segment = _encode(tokenizer, f"User: {content}\nAssistant:")
L036 |             ids.extend(segment)
L037 |             supervised.extend([False] * len(segment))
L038 |             expected = "assistant"
L039 |         else:
L040 |             segment = _encode(tokenizer, f" {content}\n")
L041 |             ids.extend(segment)
L042 |             supervised.extend([True] * len(segment))
L043 |             expected = "user"
L044 |             if i != len(messages) - 1:
L045 |                 spacer = _encode(tokenizer, "User:")
L046 |                 # Keep the next literal User label as unsupervised context. The following
L047 |                 # user turn content is appended by replacing the duplicate label below.
L048 |                 # To avoid duplicated labels, remove it immediately; this branch exists only
L049 |                 # to make the alternation intent explicit.
L050 |                 if spacer:
L051 |                     pass
L052 |     ids.append(eos)
L053 |     supervised.append(True)
L054 |     # The generic loop above emits a complete User/Assistant pair each time. For multi-turn
L055 |     # chats the next user segment already includes its own "User:" label.
L056 |     if len(ids) - 1 > max_seq_len:
L057 |         raise ValueError(f"encoded example exceeds max_seq_len={max_seq_len}: {len(ids)-1}")
L058 |     target_stream = [tok if sup else -100 for tok, sup in zip(ids, supervised)]
L059 |     x = ids[:-1]
L060 |     y = target_stream[1:]
L061 |     assistant_tokens = sum(v != -100 for v in y)
L062 |     if len(x) != len(y) or assistant_tokens <= 0 or y[-1] != eos:
L063 |         raise AssertionError("assistant-only causal alignment failure")
L064 |     return EncodedChatExample(x, y, assistant_tokens)
L065 | 
L066 | def encode_prompt_answer(tokenizer, prompt: str, answer: str, *, max_seq_len: int) -> EncodedChatExample:
L067 |     return encode_messages(
L068 |         tokenizer,
L069 |         [{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": answer.strip()}],
L070 |         max_seq_len=max_seq_len,
L071 |     )
L072 | 
L073 | def render_multiturn(messages: list[dict[str, str]]) -> str:
L074 |     parts = []
L075 |     expected = "user"
L076 |     for msg in messages:
L077 |         role = msg.get("role")
L078 |         content = str(msg.get("content", "")).strip()
L079 |         if role != expected or not content:
L080 |             raise ValueError(f"expected non-empty {expected} turn")
L081 |         parts.append(("User: " if role == "user" else "Assistant: ") + content)
L082 |         expected = "assistant" if role == "user" else "user"
L083 |     if not parts or messages[-1].get("role") != "assistant":
L084 |         raise ValueError("conversation must end with assistant")
L085 |     return "\n".join(parts)
```

## 3. `frontier350/chat_prompt350.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/chat_prompt350.py  
Git blob SHA: `cd7b29bc8c6774031bbfaca4014376413c20e8dd` · 48 lines.

```text
L001 | from __future__ import annotations
L002 | 
L003 | def render_history(history):
L004 |     parts=[]
L005 |     for role,text in history:
L006 |         text=str(text).strip()
L007 |         if role=="user":
L008 |             parts.append("User: "+text+"\nAssistant:")
L009 |         elif role=="assistant":
L010 |             parts.append(" "+text+"\n")
L011 |         else:
L012 |             raise ValueError("role must be user/assistant")
L013 |     return "".join(parts)
L014 | 
L015 | def prompt_ids(tokenizer,history):
L016 |     bos=tokenizer.token_to_id("<|bos|>")
L017 |     if bos is None:
L018 |         raise ValueError("tokenizer missing BOS")
L019 |     # SFT/DPO encode each role segment independently. Encoding the whole
L020 |     # concatenated transcript here can produce different BPE merges across
L021 |     # the Assistant:/answer and answer/User: boundaries.
L022 |     # Preserve exactly the same token boundaries at inference time.
L023 |     ids = [int(bos)]
L024 |     for role, text in history:
L025 |         content = str(text).strip()
L026 |         if role == "user":
L027 |             segment = f"User: {content}\nAssistant:"
L028 |         elif role == "assistant":
L029 |             segment = f" {content}\n"
L030 |         else:
L031 |             raise ValueError("role must be user/assistant")
L032 |         ids.extend(tokenizer.encode(segment).ids)
L033 |     return ids
L034 | 
L035 | def trim_history(tokenizer,history,max_prompt_tokens):
L036 |     if max_prompt_tokens<32:
L037 |         raise ValueError("max_prompt_tokens too small")
L038 |     h=list(history)
L039 |     while h and len(prompt_ids(tokenizer,h))>max_prompt_tokens:
L040 |         if len(h)>=3 and h[0][0]=="user" and h[1][0]=="assistant":
L041 |             h=h[2:]
L042 |         else:
L043 |             h=h[1:]
L044 |     if not h or h[-1][0]!="user":
L045 |         raise ValueError("latest user prompt cannot fit context")
L046 |     if len(prompt_ids(tokenizer,h))>max_prompt_tokens:
L047 |         raise ValueError("latest user prompt too long")
L048 |     return h
```

## 4. `frontier350/chat350_data.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/chat350_data.py  
Git blob SHA: `367b0da7dfbff0201e2405bf2b90083998560104` · 173 lines.

```text
L001 | from __future__ import annotations
L002 | import argparse
L003 | import collections
L004 | import hashlib
L005 | import json
L006 | import unicodedata
L007 | from pathlib import Path
L008 | 
L009 | REQUIRED_META = ("id", "language", "domain", "source", "license")
L010 | SPLITS = ("train", "val", "test")
L011 | ROLES = {"user", "assistant"}
L012 | 
L013 | def _norm(text: str) -> str:
L014 |     return unicodedata.normalize("NFC", str(text)).strip()
L015 | 
L016 | def normalize_row(row: dict) -> dict:
L017 |     if not isinstance(row, dict):
L018 |         raise ValueError("row must be an object")
L019 |     for key in REQUIRED_META:
L020 |         if not _norm(row.get(key, "")):
L021 |             raise ValueError(f"missing/empty field: {key}")
L022 |     if "messages" in row:
L023 |         raw = row["messages"]
L024 |         if not isinstance(raw, list) or len(raw) < 2 or len(raw) > 32:
L025 |             raise ValueError("messages must contain 2..32 turns")
L026 |         messages = []
L027 |         expected = "user"
L028 |         for item in raw:
L029 |             if not isinstance(item, dict):
L030 |                 raise ValueError("message must be an object")
L031 |             role = item.get("role")
L032 |             content = _norm(item.get("content", ""))
L033 |             if role not in ROLES or role != expected:
L034 |                 raise ValueError(f"expected {expected} turn")
L035 |             if not content:
L036 |                 raise ValueError("empty message content")
L037 |             if len(content) > 16_384:
L038 |                 raise ValueError("single turn too long")
L039 |             messages.append({"role": role, "content": content})
L040 |             expected = "assistant" if role == "user" else "user"
L041 |         if messages[-1]["role"] != "assistant":
L042 |             raise ValueError("conversation must end with assistant")
L043 |     else:
L044 |         prompt = _norm(row.get("prompt", ""))
L045 |         answer = _norm(row.get("answer", ""))
L046 |         if not prompt or not answer:
L047 |             raise ValueError("row needs messages or non-empty prompt/answer")
L048 |         messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
L049 |     total_chars = sum(len(x["content"]) for x in messages)
L050 |     if total_chars > 65_536:
L051 |         raise ValueError("conversation too long")
L052 |     return {
L053 |         "id": _norm(row["id"]),
L054 |         "language": _norm(row["language"]).lower(),
L055 |         "domain": _norm(row["domain"]).lower(),
L056 |         "source": _norm(row["source"]),
L057 |         "license": _norm(row["license"]),
L058 |         "messages": messages,
L059 |     }
L060 | 
L061 | def conversation_sha256(row: dict) -> str:
L062 |     payload = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
L063 |     return hashlib.sha256(payload.encode("utf8")).hexdigest()
L064 | 
L065 | def validate_prompt_holdouts(split_rows: dict[str, list[dict]]) -> dict:
L066 |     """Fail closed on exact normalized initial-user prompts shared by data splits.
L067 | 
L068 |     Different answers to the same root prompt still constitute holdout
L069 |     contamination. Only hashes, never prompt contents, appear in exceptions.
L070 |     This complements (not replaces) exact conversation and semantic audits.
L071 |     """
L072 |     seen: dict[str, str] = {}
L073 |     overlaps: set[tuple[str, str, str]] = set()
L074 |     for split in SPLITS:
L075 |         for row in split_rows[split]:
L076 |             first = row["messages"][0]["content"]
L077 |             canonical = " ".join(unicodedata.normalize("NFKC", first).casefold().split())
L078 |             digest = hashlib.sha256(canonical.encode("utf8")).hexdigest()
L079 |             previous = seen.setdefault(digest, split)
L080 |             if previous != split:
L081 |                 overlaps.add((digest, previous, split))
L082 |     if overlaps:
L083 |         sample = sorted(overlaps)[0]
L084 |         raise ValueError(
L085 |             f"first-user prompt leakage across train/val/test: "
L086 |             f"{len(overlaps)} cross-split overlaps (first sha256={sample[0]}, "
L087 |             f"{sample[1]} vs {sample[2]})"
L088 |         )
L089 |     return {"first_user_prompt_overlap": 0, "normalization": "NFKC-casefold-whitespace"}
L090 | 
L091 | 
L092 | def file_sha256(path: Path) -> str:
L093 |     h = hashlib.sha256()
L094 |     with path.open("rb") as f:
L095 |         for chunk in iter(lambda: f.read(1024 * 1024), b""):
L096 |             h.update(chunk)
L097 |     return h.hexdigest()
L098 | 
L099 | def read_jsonl(path: Path) -> list[dict]:
L100 |     rows = []
L101 |     for line_no, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
L102 |         if not line.strip():
L103 |             continue
L104 |         try:
L105 |             rows.append(normalize_row(json.loads(line)))
L106 |         except Exception as exc:
L107 |             raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
L108 |     if not rows:
L109 |         raise ValueError(f"empty split: {path}")
L110 |     return rows
L111 | 
L112 | def validate_dataset_root(root: Path) -> tuple[dict, dict[str, list[dict]]]:
L113 |     root = Path(root)
L114 |     split_rows = {}
L115 |     seen_ids: dict[str, str] = {}
L116 |     seen_content: dict[str, str] = {}
L117 |     stats = {}
L118 |     for split in SPLITS:
L119 |         path = root / f"{split}.jsonl"
L120 |         if not path.is_file():
L121 |             raise FileNotFoundError(path)
L122 |         rows = read_jsonl(path)
L123 |         split_rows[split] = rows
L124 |         languages = collections.Counter()
L125 |         domains = collections.Counter()
L126 |         sources = collections.Counter()
L127 |         turns = collections.Counter()
L128 |         for row in rows:
L129 |             rid = row["id"]
L130 |             if rid in seen_ids:
L131 |                 raise ValueError(f"duplicate id across splits: {rid} ({seen_ids[rid]} vs {split})")
L132 |             seen_ids[rid] = split
L133 |             digest = conversation_sha256(row)
L134 |             if digest in seen_content:
L135 |                 raise ValueError(f"duplicate conversation across splits: {seen_content[digest]} vs {split}")
L136 |             seen_content[digest] = split
L137 |             languages[row["language"]] += 1
L138 |             domains[row["domain"]] += 1
L139 |             sources[row["source"]] += 1
L140 |             turns[len(row["messages"])] += 1
L141 |         stats[split] = {
L142 |             "rows": len(rows),
L143 |             "sha256": file_sha256(path),
L144 |             "languages": dict(sorted(languages.items())),
L145 |             "domains": dict(sorted(domains.items())),
L146 |             "sources": dict(sorted(sources.items())),
L147 |             "turn_counts": {str(k): v for k, v in sorted(turns.items())},
L148 |         }
L149 |     manifest = {
L150 |         "schema": 1,
L151 |         "format": "neraltalk350-jsonl-v1",
L152 |         "splits": stats,
L153 |         "total_rows": sum(x["rows"] for x in stats.values()),
L154 |         "global_unique_ids": len(seen_ids),
L155 |         "global_unique_conversations": len(seen_content),
L156 |         "assistant_only_training": True,
L157 |         "benchmark_policy": "train/val/test are development data only; frozen NIS2/public benchmark holdouts remain separate and forbidden from training",
L158 |     }
L159 |     canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf8")
L160 |     manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
L161 |     return manifest, split_rows
L162 | 
L163 | def main():
L164 |     ap = argparse.ArgumentParser()
L165 |     ap.add_argument("--root", required=True)
L166 |     ap.add_argument("--out", required=True)
L167 |     args = ap.parse_args()
L168 |     manifest, _ = validate_dataset_root(Path(args.root))
L169 |     Path(args.out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
L170 |     print(json.dumps(manifest, ensure_ascii=False))
L171 | 
L172 | if __name__ == "__main__":
L173 |     main()
```

## 5. `frontier350/sft_batching350.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/sft_batching350.py  
Git blob SHA: `c03783634cea3dabb2f4cbcce4280672dced1a7c` · 42 lines.

```text
L001 | from __future__ import annotations
L002 | 
L003 | def _length(row) -> int:
L004 |     ids=getattr(row,"input_ids",None)
L005 |     if ids is None:raise ValueError("row has no input_ids")
L006 |     n=len(ids)
L007 |     if n<=0:raise ValueError("empty encoded row")
L008 |     return n
L009 | 
L010 | def epoch_order(rows, rng, *, batch_size:int, bucket_size:int) -> list[int]:
L011 |     if batch_size<=0:raise ValueError("batch_size must be positive")
L012 |     if bucket_size<batch_size or bucket_size%batch_size:
L013 |         raise ValueError("bucket_size must be >= batch_size and divisible by batch_size")
L014 |     if not rows:raise ValueError("rows are empty")
L015 |     order=sorted(range(len(rows)),key=lambda i:(_length(rows[i]),i))
L016 |     buckets=[order[i:i+bucket_size] for i in range(0,len(order),bucket_size)]
L017 |     for bucket in buckets:rng.shuffle(bucket)
L018 |     rng.shuffle(buckets)
L019 |     flat=[i for bucket in buckets for i in bucket]
L020 |     if sorted(flat)!=list(range(len(rows))):raise AssertionError("epoch order lost or duplicated rows")
L021 |     return flat
L022 | 
L023 | def length_bucket_batches(rows,batch_size,rng,*,bucket_size:int=64):
L024 |     if len(rows)<batch_size:raise ValueError("dataset smaller than batch_size")
L025 |     queue=[]
L026 |     while True:
L027 |         if len(queue)<batch_size:
L028 |             queue.extend(epoch_order(rows,rng,batch_size=batch_size,bucket_size=bucket_size))
L029 |         picked=queue[:batch_size]
L030 |         del queue[:batch_size]
L031 |         yield [rows[i] for i in picked]
L032 | 
L033 | def padding_stats(batch) -> dict:
L034 |     lengths=[_length(x) for x in batch]
L035 |     width=max(lengths);real=sum(lengths);padded=width*len(lengths)
L036 |     return {
L037 |         "batch_width":width,
L038 |         "real_tokens":real,
L039 |         "padded_tokens":padded,
L040 |         "padding_tokens":padded-real,
L041 |         "padding_fraction":(padded-real)/padded if padded else 0.0,
L042 |     }
```

## 6. `frontier350/sft_stratified350.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/sft_stratified350.py  
Git blob SHA: `b7c3ecf7275d3bf3e5cee2b5984b468dca4ddcee` · 55 lines.

```text
L001 | from __future__ import annotations
L002 | from collections import deque
L003 | 
L004 | 
L005 | def _shuffled_length_order(rows, indices, rng, bucket_size):
L006 |     if not indices:
L007 |         raise ValueError("a sampling cohort is empty")
L008 |     ordered = sorted(indices, key=lambda i: (len(rows[i].input_ids), i))
L009 |     buckets = [ordered[i:i + bucket_size] for i in range(0, len(ordered), bucket_size)]
L010 |     for bucket in buckets:
L011 |         rng.shuffle(bucket)
L012 |     rng.shuffle(buckets)
L013 |     result = [i for bucket in buckets for i in bucket]
L014 |     if len(result) != len(indices) or len(set(result)) != len(indices):
L015 |         raise AssertionError("sampling epoch lost or duplicated examples")
L016 |     return result
L017 | 
L018 | 
L019 | def stratified_batches(rows, is_human, batch_size, rng, *,
L020 |                        human_per_batch=2, bucket_size=64):
L021 |     """Infinite deterministic batches with independently shuffled human/verified cohorts.
L022 | 
L023 |     Every cohort is fully traversed without replacement before an index from
L024 |     that cohort can be reused. No foreign model teacher data are introduced.
L025 |     The supplied index labels come from the *validated source metadata*, never
L026 |     from the response text.
L027 |     """
L028 |     if len(rows) != len(is_human):
L029 |         raise ValueError("cohort label length mismatch")
L030 |     if batch_size <= 1 or not 0 < human_per_batch < batch_size:
L031 |         raise ValueError("human_per_batch must be between 1 and batch_size-1")
L032 |     if bucket_size < batch_size:
L033 |         raise ValueError("bucket_size must be >= batch_size")
L034 |     groups = {
L035 |         True: [i for i, human in enumerate(is_human) if human],
L036 |         False: [i for i, human in enumerate(is_human) if not human],
L037 |     }
L038 |     if min(len(groups[True]), len(groups[False])) < batch_size:
L039 |         raise ValueError("both cohorts need at least batch_size examples")
L040 |     queues = {True: deque(), False: deque()}
L041 |     quotas = {True: human_per_batch, False: batch_size-human_per_batch}
L042 |     while True:
L043 |         picked = []
L044 |         for key in (True, False):
L045 |             while len(queues[key]) < quotas[key]:
L046 |                 queues[key].extend(_shuffled_length_order(rows, groups[key], rng, bucket_size))
L047 |             picked.extend(queues[key].popleft() for _ in range(quotas[key]))
L048 |         rng.shuffle(picked)
L049 |         yield [rows[i] for i in picked]
L050 | 
L051 | 
L052 | def cohort_summary(is_human):
L053 |     h = sum(bool(x) for x in is_human)
L054 |     return {"human_rows":h,"verified_rows":len(is_human)-h,
L055 |             "human_row_fraction":h/max(1,len(is_human))}
```

## 7. `frontier350/train_chat350.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/train_chat350.py  
Git blob SHA: `c2530477fc604d71313e0dadf95cb91f65a2bde2` · 252 lines.

```text
L001 | from __future__ import annotations
L002 | import argparse
L003 | import hashlib
L004 | import json
L005 | import math
L006 | import os
L007 | import random
L008 | import time
L009 | from pathlib import Path
L010 | 
L011 | import numpy as np
L012 | import torch
L013 | import torch.nn.functional as F
L014 | from tokenizers import Tokenizer
L015 | 
L016 | from chat350_contract import EXPECTED_TOKENIZER_SHA256, validate_checkpoint_metadata
L017 | from secure_checkpoint350 import load_checkpoint
L018 | from chat350_data import validate_dataset_root
L019 | from chat350_sft_format import encode_messages
L020 | from model350 import FRONTIER_350, Neral350M
L021 | from sft_batching350 import length_bucket_batches, padding_stats
L022 | from sft_stratified350 import stratified_batches, cohort_summary
L023 | 
L024 | def sha256_file(path: Path) -> str:
L025 |     h=hashlib.sha256()
L026 |     with path.open("rb") as f:
L027 |         for chunk in iter(lambda:f.read(4*1024*1024),b""): h.update(chunk)
L028 |     return h.hexdigest()
L029 | 
L030 | def atomic_save(obj, path: Path):
L031 |     path.parent.mkdir(parents=True, exist_ok=True)
L032 |     tmp=path.with_suffix(path.suffix+".tmp")
L033 |     torch.save(obj,tmp)
L034 |     os.replace(tmp,path)
L035 | 
L036 | def load_tokenizer(path: Path) -> Tokenizer:
L037 |     if sha256_file(path)!=EXPECTED_TOKENIZER_SHA256:
L038 |         raise ValueError("locked 16K tokenizer digest mismatch")
L039 |     tok=Tokenizer.from_file(str(path))
L040 |     if tok.get_vocab_size()!=FRONTIER_350.vocab_size:
L041 |         raise ValueError("tokenizer vocabulary mismatch")
L042 |     expected={"<|bos|>":0,"<|eos|>":1,"<|sep|>":2,"<|pad|>":3}
L043 |     got={k:tok.token_to_id(k) for k in expected}
L044 |     if got!=expected: raise ValueError(f"special-token mapping mismatch: {got}")
L045 |     return tok
L046 | 
L047 | def list_shards(root: Path) -> list[Path]:
L048 |     shards=sorted(Path(root).glob("*.bin"))
L049 |     if not shards: raise FileNotFoundError(f"no replay shards in {root}")
L050 |     return shards
L051 | 
L052 | def replay_batch(shards, batch, seq, device, rng, vocab):
L053 |     xs=[]; ys=[]
L054 |     for _ in range(batch):
L055 |         p=rng.choice(shards); a=np.memmap(p,dtype=np.uint16,mode="r")
L056 |         if len(a)<seq+1: raise ValueError(f"replay shard too short: {p}")
L057 |         s=rng.randrange(0,len(a)-seq)
L058 |         z=np.asarray(a[s:s+seq+1],dtype=np.int64)
L059 |         if z.max(initial=0)>=vocab: raise ValueError(f"replay token outside vocabulary: {p}")
L060 |         xs.append(torch.from_numpy(z[:-1].copy())); ys.append(torch.from_numpy(z[1:].copy()))
L061 |     return torch.stack(xs).to(device,non_blocking=True),torch.stack(ys).to(device,non_blocking=True)
L062 | 
L063 | def collate(encoded, pad_id, device):
L064 |     width=max(len(x.input_ids) for x in encoded)
L065 |     x=torch.full((len(encoded),width),pad_id,dtype=torch.long)
L066 |     y=torch.full((len(encoded),width),-100,dtype=torch.long)
L067 |     for i,row in enumerate(encoded):
L068 |         x[i,:len(row.input_ids)]=torch.tensor(row.input_ids,dtype=torch.long)
L069 |         y[i,:len(row.targets)]=torch.tensor(row.targets,dtype=torch.long)
L070 |     return x.to(device,non_blocking=True),y.to(device,non_blocking=True)
L071 | 
L072 | def assistant_loss(model,x,y,eos_id,eos_weight):
L073 |     logits,_=model(x)
L074 |     flat_y=y.reshape(-1); mask=flat_y!=-100
L075 |     if not bool(mask.any()): raise ValueError("batch has no assistant targets")
L076 |     selected=logits.reshape(-1,logits.size(-1))[mask].float()
L077 |     targets=flat_y[mask]
L078 |     per=F.cross_entropy(selected,targets,reduction="none")
L079 |     w=torch.ones_like(per); w[targets==eos_id]=eos_weight
L080 |     return (per*w).sum()/w.sum().clamp_min(1.0)
L081 | 
L082 | @torch.inference_mode()
L083 | def eval_sft(model,encoded,batch_size,pad_id,eos_id,device):
L084 |     model.eval(); nll=0.0; tokens=0; eos_ok=0; eos_total=0
L085 |     for i in range(0,len(encoded),batch_size):
L086 |         x,y=collate(encoded[i:i+batch_size],pad_id,device)
L087 |         logits,_=model(x); fy=y.reshape(-1); mask=fy!=-100
L088 |         loss=F.cross_entropy(logits.reshape(-1,logits.size(-1))[mask].float(),fy[mask],reduction="sum")
L089 |         nll+=float(loss); tokens+=int(mask.sum())
L090 |         em=y==eos_id
L091 |         if bool(em.any()):
L092 |             pred=logits.argmax(-1); eos_ok+=int((pred[em]==eos_id).sum()); eos_total+=int(em.sum())
L093 |     model.train()
L094 |     return {"answer_nll":nll/max(1,tokens),"answer_tokens":tokens,
L095 |             "terminal_eos_top1_pct":100.0*eos_ok/max(1,eos_total)}
L096 | 
L097 | @torch.inference_mode()
L098 | def eval_replay(model,shards,batch,seq,batches,device,seed):
L099 |     rng=random.Random(seed); model.eval(); vals=[]
L100 |     for _ in range(batches):
L101 |         x,y=replay_batch(shards,batch,seq,device,rng,model.cfg.vocab_size)
L102 |         _,loss=model(x,y); vals.append(float(loss))
L103 |     model.train(); return sum(vals)/len(vals)
L104 | 
L105 | def lr_at(step,total,peak,floor,warm_frac):
L106 |     warm=max(1,round(total*warm_frac))
L107 |     if step<=warm: return peak*step/warm
L108 |     p=min(1.0,max(0.0,(step-warm)/max(1,total-warm)))
L109 |     return floor+(peak-floor)*0.5*(1+math.cos(math.pi*p))
L110 | 
L111 | def shuffled_batches(rows,batch,rng):
L112 |     order=list(range(len(rows))); rng.shuffle(order); cursor=0
L113 |     while True:
L114 |         picked=[]
L115 |         while len(picked)<batch:
L116 |             if cursor>=len(order): rng.shuffle(order); cursor=0
L117 |             take=min(batch-len(picked),len(order)-cursor)
L118 |             picked.extend(rows[order[j]] for j in range(cursor,cursor+take)); cursor+=take
L119 |         yield picked
L120 | 
L121 | def main():
L122 |     ap=argparse.ArgumentParser()
L123 |     ap.add_argument("--init-from",required=True); ap.add_argument("--tokenizer",required=True)
L124 |     ap.add_argument("--data",required=True); ap.add_argument("--replay-data",required=True); ap.add_argument("--out",required=True)
L125 |     ap.add_argument("--steps",type=int,default=300); ap.add_argument("--seq",type=int,default=512)
L126 |     ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--replay-batch",type=int,default=2)
L127 |     ap.add_argument("--replay-seq",type=int,default=512); ap.add_argument("--replay-every",type=int,default=4)
L128 |     ap.add_argument("--sft-weight",type=float,default=.90); ap.add_argument("--eos-weight",type=float,default=1.5)
L129 |     ap.add_argument("--peak-lr",type=float,default=2e-5); ap.add_argument("--min-lr",type=float,default=2e-6)
L130 |     ap.add_argument("--warmup-fraction",type=float,default=.05); ap.add_argument("--eval-every",type=int,default=50)
L131 |     ap.add_argument("--save-every",type=int,default=50); ap.add_argument("--eval-replay-batches",type=int,default=8)
L132 |     ap.add_argument("--seed",type=int,default=350_726); ap.add_argument("--gradient-checkpointing",action="store_true"); ap.add_argument("--length-bucket-size",type=int,default=64)
L133 |     ap.add_argument("--sampling-mode",choices=["bucketed","human-balanced"],default="bucketed")
L134 |     ap.add_argument("--human-per-batch",type=int,default=2)
L135 |     args=ap.parse_args()
L136 |     if args.seq>FRONTIER_350.max_seq_len or args.replay_seq>FRONTIER_350.max_seq_len: raise ValueError("sequence exceeds architecture")
L137 |     if min(args.steps,args.batch_size,args.replay_batch,args.replay_every,args.eval_every,args.save_every)<=0: raise ValueError("positive budgets required")
L138 |     if args.length_bucket_size<args.batch_size or args.length_bucket_size%args.batch_size: raise ValueError("length-bucket-size must be >= batch-size and divisible by batch-size")
L139 |     if not 0.5<=args.sft_weight<=1.0: raise ValueError("sft-weight must be 0.5..1")
L140 |     if not 1.0<=args.eos_weight<=3.0: raise ValueError("eos-weight must be 1..3")
L141 |     if not 0<args.min_lr<=args.peak_lr: raise ValueError("invalid learning rates")
L142 |     if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 CUDA required")
L143 | 
L144 |     root=Path(args.data); manifest,splits=validate_dataset_root(root)
L145 |     # Exact conversation de-duplication alone misses identical prompts with
L146 |     # different answers across train and evaluation splits.
L147 |     from chat350_data import validate_prompt_holdouts
L148 |     prompt_holdout_audit = validate_prompt_holdouts(splits)
L149 |     tok=load_tokenizer(Path(args.tokenizer))
L150 |     train=[encode_messages(tok,r["messages"],max_seq_len=args.seq) for r in splits["train"]]
L151 |     train_is_human=[r["source"].startswith("oasst2-human") for r in splits["train"]]
L152 |     mixture=cohort_summary(train_is_human)
L153 |     if args.sampling_mode=="human-balanced" and not (0<args.human_per_batch<args.batch_size):
L154 |         raise ValueError("human-per-batch must be between 1 and batch-size-1")
L155 |     val=[encode_messages(tok,r["messages"],max_seq_len=args.seq) for r in splits["val"]]
L156 |     test=[encode_messages(tok,r["messages"],max_seq_len=args.seq) for r in splits["test"]]
L157 |     pad_id=int(tok.token_to_id("<|pad|>")); eos_id=int(tok.token_to_id("<|eos|>"))
L158 |     replay=list_shards(Path(args.replay_data))
L159 | 
L160 |     source_path=Path(args.init_from); source_sha=sha256_file(source_path)
L161 |     payload=load_checkpoint(source_path)
L162 |     validate_checkpoint_metadata(payload,min_processed_tokens=1_000_000_000)
L163 |     source_tokens=int(payload.get("processed_tokens",payload.get("source_processed_tokens",payload.get("tokens",0))))
L164 |     device=torch.device("cuda"); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
L165 |     random.seed(args.seed); np.random.seed(args.seed); torch.set_float32_matmul_precision("high")
L166 |     model=Neral350M().to(device=device,dtype=torch.bfloat16)
L167 |     model.load_state_dict(payload["model"],strict=True); del payload
L168 |     model.set_gradient_checkpointing(args.gradient_checkpointing); model.train()
L169 |     if model.parameter_report()["total"]!=349_825_152: raise RuntimeError("unexpected parameter count")
L170 | 
L171 |     baseline_sft=eval_sft(model,val,args.batch_size,pad_id,eos_id,device)
L172 |     baseline_replay=eval_replay(model,replay,args.replay_batch,args.replay_seq,args.eval_replay_batches,device,args.seed+91)
L173 |     opt=torch.optim.AdamW(model.parameters(),lr=args.peak_lr,betas=(.9,.95),eps=1e-8,weight_decay=.03,fused=True)
L174 |     rng=random.Random(args.seed+1)
L175 |     batches=(stratified_batches(train,train_is_human,args.batch_size,rng,
L176 |               human_per_batch=args.human_per_batch,bucket_size=args.length_bucket_size)
L177 |               if args.sampling_mode=="human-balanced"
L178 |               else length_bucket_batches(train,args.batch_size,rng,bucket_size=args.length_bucket_size))
L179 |     out=Path(args.out)
L180 |     if out.exists() and any(out.iterdir()): raise FileExistsError("chat output directory must be fresh")
L181 |     out.mkdir(parents=True,exist_ok=True)
L182 |     run={"stage":"Arctus 350M chat SFT pilot","source_checkpoint_sha256":source_sha,"source_processed_tokens":source_tokens,
L183 |          "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,"data_manifest":manifest,"baseline_sft":baseline_sft,
L184 |          "baseline_replay_loss":baseline_replay,"fresh_optimizer":True,"assistant_only":True,"args":vars(args),
L185 |          "gpu":torch.cuda.get_device_name(0),"torch":str(torch.__version__),
L186 |          "sampling":{**mixture,"mode":args.sampling_mode,"human_per_batch":args.human_per_batch}}
L187 |     (out/"run.json").write_text(json.dumps(run,indent=2,ensure_ascii=False),encoding="utf8")
L188 | 
L189 |     def model_payload(step, val_report=None):
L190 |         return {"model":model.state_dict(),"config":FRONTIER_350.to_dict(),"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
L191 |                 "source_checkpoint_sha256":source_sha,"source_processed_tokens":source_tokens,
L192 |                 "chat_step":step,"data_manifest_sha256":manifest["manifest_sha256"],"validation":val_report}
L193 | 
L194 |     best=baseline_sft["answer_nll"]; best_eos=baseline_sft["terminal_eos_top1_pct"]
L195 |     start=time.perf_counter(); measured_tokens=0; update_seconds=0.0; total_sft_real=0; total_sft_padded=0; sampled_human=0; sampled_verified=0; torch.cuda.reset_peak_memory_stats()
L196 |     metrics=out/"metrics.jsonl"
L197 |     for step in range(1,args.steps+1):
L198 |         lr=lr_at(step,args.steps,args.peak_lr,args.min_lr,args.warmup_fraction)
L199 |         for g in opt.param_groups:g["lr"]=lr
L200 |         batch_rows=next(batches); pad=padding_stats(batch_rows); x,y=collate(batch_rows,pad_id,device); do_replay=step%args.replay_every==0
L201 |         total_sft_real+=pad["real_tokens"]; total_sft_padded+=pad["padded_tokens"]
L202 |         if args.sampling_mode=="human-balanced":
L203 |             sampled_human+=args.human_per_batch
L204 |             sampled_verified+=args.batch_size-args.human_per_batch
L205 |         opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); tick=time.perf_counter()
L206 |         sft=assistant_loss(model,x,y,eos_id,args.eos_weight)
L207 |         (sft*(args.sft_weight if do_replay else 1.0)).backward()
L208 |         replay_loss=None; compute_tokens=x.numel()
L209 |         if do_replay:
L210 |             rx,ry=replay_batch(replay,args.replay_batch,args.replay_seq,device,rng,model.cfg.vocab_size)
L211 |             _,rl=model(rx,ry); replay_loss=rl
L212 |             (rl*(1.0-args.sft_weight)).backward(); compute_tokens+=rx.numel()
L213 |         gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
L214 |         opt.step(); torch.cuda.synchronize(); dt=time.perf_counter()-tick
L215 |         update_seconds+=dt; measured_tokens+=compute_tokens
L216 |         row={"step":step,"sft_loss":float(sft.detach()),"replay_loss":None if replay_loss is None else float(replay_loss.detach()),
L217 |              "lr":lr,"grad_norm":float(gn),"seconds":dt,"compute_tokens":compute_tokens,"tokens_per_second":compute_tokens/max(dt,1e-9),
L218 |              "sft_batch_width":pad["batch_width"],"sft_real_tokens":pad["real_tokens"],"sft_padded_tokens":pad["padded_tokens"],
L219 |              "sft_padding_tokens":pad["padding_tokens"],"sft_padding_fraction":pad["padding_fraction"],
L220 |              "sampling_mode":args.sampling_mode}
L221 |         if step%args.eval_every==0 or step==args.steps:
L222 |             vr=eval_sft(model,val,args.batch_size,pad_id,eos_id,device)
L223 |             rr=eval_replay(model,replay,args.replay_batch,args.replay_seq,args.eval_replay_batches,device,args.seed+91)
L224 |             replay_ok=rr<=baseline_replay*1.05
L225 |             improved=vr["answer_nll"]<best-0.002 and vr["terminal_eos_top1_pct"]>=best_eos and replay_ok
L226 |             row.update(validation=vr,replay_validation_loss=rr,replay_guard=replay_ok,selected=improved)
L227 |             if improved:
L228 |                 best=vr["answer_nll"]; best_eos=vr["terminal_eos_top1_pct"]
L229 |                 atomic_save(model_payload(step,vr),out/"best_chat.pt")
L230 |         if step%args.save_every==0 or step==args.steps:
L231 |             atomic_save({**model_payload(step,row.get("validation")),"optimizer":opt.state_dict(),
L232 |                          "rng_state":rng.getstate(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},
L233 |                         out/"latest_resume.pt")
L234 |         with metrics.open("a",encoding="utf8") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
L235 |         if step%10==0 or step==args.steps: print(json.dumps(row,ensure_ascii=False),flush=True)
L236 | 
L237 |     final_sft=eval_sft(model,val,args.batch_size,pad_id,eos_id,device)
L238 |     test_sft=eval_sft(model,test,args.batch_size,pad_id,eos_id,device)
L239 |     final_replay=eval_replay(model,replay,args.replay_batch,args.replay_seq,args.eval_replay_batches,device,args.seed+91)
L240 |     summary={"complete":True,"steps":args.steps,"source_checkpoint_sha256":source_sha,"source_processed_tokens":source_tokens,
L241 |              "baseline_sft":baseline_sft,"final_sft":final_sft,"test_sft":test_sft,"baseline_replay_loss":baseline_replay,
L242 |              "final_replay_loss":final_replay,"replay_regression_pct":100*(final_replay/baseline_replay-1),
L243 |              "best_answer_nll":best,"best_eos_top1_pct":best_eos,"optimizer_seconds":update_seconds,
L244 |              "mean_optimizer_tokens_per_second":measured_tokens/max(update_seconds,1e-9),
L245 |              "sft_real_tokens":total_sft_real,"sft_padded_tokens":total_sft_padded,"sft_padding_fraction":1.0-total_sft_real/max(1,total_sft_padded),
L246 |              "sampled_human_rows":sampled_human,"sampled_verified_rows":sampled_verified,"sampling_mode":args.sampling_mode,
L247 |              "elapsed_seconds":time.perf_counter()-start,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
L248 |              "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,"best_checkpoint_exists":(out/"best_chat.pt").is_file()}
L249 |     (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
L250 |     print(json.dumps(summary,ensure_ascii=False),flush=True)
L251 | 
L252 | if __name__=="__main__": main()
```

## 8. `frontier350/model350.py`

Original file: https://github.com/Cola968/neral-helios/blob/bf9f638f6c45311c175c3f154d7c62a7952b8465/frontier350/model350.py  
Git blob SHA: `e6184b0698ffdd28e5c2eff2110a4b146a58a84b` · 209 lines.

```text
L001 | from __future__ import annotations
L002 | 
L003 | """Neral 350M Frontier v1.
L004 | 
L005 | From-scratch decoder-only LM sized for ~350M parameters and optimized for RTX-class
L006 | training plus compact mobile inference. The release-candidate backbone uses Neral's own
L007 | 16K byte-level BPE tokenizer; no foreign weights, vocabulary, or merges are loaded.
L008 | """
L009 | 
L010 | from dataclasses import dataclass, asdict
L011 | import math
L012 | from typing import Optional
L013 | import torch
L014 | import torch.nn as nn
L015 | import torch.nn.functional as F
L016 | from torch.utils.checkpoint import checkpoint as activation_checkpoint
L017 | 
L018 | 
L019 | _ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
L020 | 
L021 | 
L022 | @dataclass(frozen=True)
L023 | class Neral350Config:
L024 |     vocab_size: int = 16_384
L025 |     d_model: int = 1152
L026 |     n_layers: int = 24
L027 |     n_heads: int = 18
L028 |     n_kv_heads: int = 2
L029 |     ffn_hidden: int = 3136
L030 |     max_seq_len: int = 8192
L031 |     rope_theta: float = 250_000.0
L032 |     rms_eps: float = 1e-5
L033 |     qk_rms_eps: float = 1e-6
L034 |     dropout: float = 0.0
L035 |     initializer_range: float = 0.02
L036 | 
L037 |     @property
L038 |     def head_dim(self) -> int:
L039 |         return self.d_model // self.n_heads
L040 | 
L041 |     @property
L042 |     def kv_dim(self) -> int:
L043 |         return self.n_kv_heads * self.head_dim
L044 | 
L045 |     @property
L046 |     def gqa_groups(self) -> int:
L047 |         return self.n_heads // self.n_kv_heads
L048 | 
L049 |     def __post_init__(self) -> None:
L050 |         if self.d_model % self.n_heads:
L051 |             raise ValueError("d_model must be divisible by n_heads")
L052 |         if self.n_heads % self.n_kv_heads:
L053 |             raise ValueError("n_heads must be divisible by n_kv_heads")
L054 |         if self.head_dim % 2:
L055 |             raise ValueError("RoPE requires even head dimension")
L056 |         if min(self.n_layers, self.ffn_hidden, self.max_seq_len) <= 0:
L057 |             raise ValueError("positive dimensions required")
L058 | 
L059 |     def to_dict(self) -> dict:
L060 |         return asdict(self)
L061 | 
L062 | 
L063 | # Hardware winner on the project's RTX 4070: 24 layers x 1152 width, 18 Q heads / 2 KV
L064 | # heads, 3136 SwiGLU hidden, 16K own BPE. Measured parameter count: 349,825,152.
L065 | FRONTIER_350 = Neral350Config()
L066 | 
L067 | 
L068 | class RMSNorm(nn.Module):
L069 |     def __init__(self, dim: int, eps: float):
L070 |         super().__init__()
L071 |         self.weight = nn.Parameter(torch.ones(dim))
L072 |         self.eps = eps
L073 | 
L074 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
L075 |         xf = x.float()
L076 |         y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
L077 |         return y.to(x.dtype) * self.weight
L078 | 
L079 | 
L080 | def rotate_half(x: torch.Tensor) -> torch.Tensor:
L081 |     a, b = x[..., ::2], x[..., 1::2]
L082 |     return torch.stack((-b, a), dim=-1).flatten(-2)
L083 | 
L084 | 
L085 | def _rope_table(max_seq_len: int, dim: int, theta: float, device: torch.device, dtype: torch.dtype):
L086 |     key = (device.type, device.index, str(dtype), int(max_seq_len), int(dim), float(theta))
L087 |     cached = _ROPE_CACHE.get(key)
L088 |     if cached is not None and not any(t.is_inference() for t in cached):
L089 |         return cached
L090 |     # Evaluation uses torch.inference_mode(). Caching tensors created inside
L091 |     # that context makes the first subsequent training backward fail because
L092 |     # autograd cannot save inference tensors (even if they require no grad).
L093 |     # Explicitly disable inference mode when constructing static RoPE tables.
L094 |     with torch.inference_mode(False), torch.no_grad():
L095 |         pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
L096 |         inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
L097 |         phase = torch.outer(pos, inv).repeat_interleave(2, -1)[None, None]
L098 |         result = phase.cos().to(dtype=dtype), phase.sin().to(dtype=dtype)
L099 |     _ROPE_CACHE[key] = result
L100 |     return result
L101 | 
L102 | 
L103 | def rope(q: torch.Tensor, k: torch.Tensor, theta: float, max_seq_len: int, offset: int = 0):
L104 |     t = q.size(-2)
L105 |     end = offset + t
L106 |     if end > max_seq_len:
L107 |         raise ValueError("RoPE position exceeds max_seq_len")
L108 |     c, s = _rope_table(max_seq_len, q.size(-1), theta, q.device, q.dtype)
L109 |     c = c[..., offset:end, :]
L110 |     s = s[..., offset:end, :]
L111 |     return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s
L112 | 
L113 | 
L114 | class GQA(nn.Module):
L115 |     def __init__(self, cfg: Neral350Config):
L116 |         super().__init__()
L117 |         self.cfg = cfg
L118 |         d = cfg.d_model
L119 |         self.q = nn.Linear(d, d, bias=False)
L120 |         self.k = nn.Linear(d, cfg.kv_dim, bias=False)
L121 |         self.v = nn.Linear(d, cfg.kv_dim, bias=False)
L122 |         self.o = nn.Linear(d, d, bias=False)
L123 |         self.qn = RMSNorm(cfg.head_dim, cfg.qk_rms_eps)
L124 |         self.kn = RMSNorm(cfg.head_dim, cfg.qk_rms_eps)
L125 | 
L126 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
L127 |         b, t, _ = x.shape
L128 |         q = self.q(x).view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
L129 |         k = self.k(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
L130 |         v = self.v(x).view(b, t, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
L131 |         q, k = self.qn(q), self.kn(k)
L132 |         q, k = rope(q, k, self.cfg.rope_theta, self.cfg.max_seq_len)
L133 |         try:
L134 |             y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.cfg.dropout if self.training else 0.0, enable_gqa=True)
L135 |         except (TypeError, RuntimeError):
L136 |             k = k.repeat_interleave(self.cfg.gqa_groups, 1)
L137 |             v = v.repeat_interleave(self.cfg.gqa_groups, 1)
L138 |             y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.cfg.dropout if self.training else 0.0)
L139 |         return self.o(y.transpose(1, 2).contiguous().view(b, t, self.cfg.d_model))
L140 | 
L141 | 
L142 | class SwiGLU(nn.Module):
L143 |     def __init__(self, cfg: Neral350Config):
L144 |         super().__init__()
L145 |         self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
L146 |         self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
L147 |         self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
L148 | 
L149 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
L150 |         return self.down(F.silu(self.gate(x)) * self.up(x))
L151 | 
L152 | 
L153 | class Block(nn.Module):
L154 |     def __init__(self, cfg: Neral350Config):
L155 |         super().__init__()
L156 |         self.n1 = RMSNorm(cfg.d_model, cfg.rms_eps)
L157 |         self.attn = GQA(cfg)
L158 |         self.n2 = RMSNorm(cfg.d_model, cfg.rms_eps)
L159 |         self.ffn = SwiGLU(cfg)
L160 | 
L161 |     def forward(self, x: torch.Tensor) -> torch.Tensor:
L162 |         x = x + self.attn(self.n1(x))
L163 |         return x + self.ffn(self.n2(x))
L164 | 
L165 | 
L166 | class Neral350M(nn.Module):
L167 |     def __init__(self, cfg: Optional[Neral350Config] = None):
L168 |         super().__init__()
L169 |         self.cfg = cfg or FRONTIER_350
L170 |         self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
L171 |         self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)])
L172 |         self.norm = RMSNorm(self.cfg.d_model, self.cfg.rms_eps)
L173 |         self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
L174 |         self.lm_head.weight = self.tok_emb.weight
L175 |         self.gradient_checkpointing = False
L176 |         self.apply(self._init)
L177 |         self._residual_scale()
L178 | 
L179 |     def _init(self, module: nn.Module) -> None:
L180 |         if isinstance(module, (nn.Linear, nn.Embedding)):
L181 |             nn.init.normal_(module.weight, 0.0, self.cfg.initializer_range)
L182 | 
L183 |     def _residual_scale(self) -> None:
L184 |         std = self.cfg.initializer_range / math.sqrt(2 * self.cfg.n_layers)
L185 |         for b in self.blocks:
L186 |             nn.init.normal_(b.attn.o.weight, 0.0, std)
L187 |             nn.init.normal_(b.ffn.down.weight, 0.0, std)
L188 | 
L189 |     def set_gradient_checkpointing(self, enabled: bool = True) -> None:
L190 |         self.gradient_checkpointing = bool(enabled)
L191 | 
L192 |     def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
L193 |         if idx.ndim != 2 or not 0 < idx.size(1) <= self.cfg.max_seq_len:
L194 |             raise ValueError("invalid input shape/context")
L195 |         x = self.tok_emb(idx)
L196 |         for b in self.blocks:
L197 |             x = activation_checkpoint(b, x, use_reentrant=False) if self.training and self.gradient_checkpointing else b(x)
L198 |         logits = self.lm_head(self.norm(x))
L199 |         loss = None
L200 |         if targets is not None:
L201 |             loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
L202 |         return logits, loss
L203 | 
L204 |     def parameter_report(self) -> dict:
L205 |         total = sum(p.numel() for p in self.parameters())
L206 |         return {"total": total, "millions": total / 1e6}
L207 | 
L208 |     def estimated_kv_cache_bytes(self, sequence_length: int, batch_size: int = 1, bytes_per_value: int = 2) -> int:
L209 |         return batch_size * self.cfg.n_layers * 2 * self.cfg.n_kv_heads * sequence_length * self.cfg.head_dim * bytes_per_value
```

**End of Part A.** Next, attach Part B (evaluation/score/checkpoints) to the same Claude conversation. Historical logs and local checkpoint bytes are not provided; request actual evidence if a hypothesis depends on them.
