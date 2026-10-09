# External evaluation-only holdouts

These datasets are not Neral training sources.

GSM8K
- Hugging Face repo: openai/gsm8k
- pinned revision: 740312add88f781978c0658806c59bc2815b9866
- test parquet: main/test-00000-of-00001.parquet
- parquet SHA-256: ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59
- rows: 1,319
- license: MIT
- original/crowdsourced benchmark

MBPP
- Hugging Face repo: google-research-datasets/mbpp
- pinned revision: 4bb6404fdc6cacfda99d4ac4205087b89d32030c
- sanitized test parquet: sanitized/test-00000-of-00001.parquet
- parquet SHA-256: e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850
- rows: 257
- license: CC-BY-4.0
- crowd/expert-generated benchmark

The preparation script verifies both SHA-256 hashes before converting them to local JSONL. Raw benchmark questions remain local and are never uploaded by the benchmark workflows.

GSM8K scoring uses exact final-number comparison on deterministic greedy generations. MBPP uses sanitized tests. Model-produced Python is treated as untrusted: AST validation blocks dangerous imports, dynamic execution/file APIs and dunder access, then a child Python process applies CPU, address-space and file-size limits. This is defense in depth, not a general-purpose secure sandbox.

External benchmark results are supplementary because absence of historical pretraining-corpus contamination is not yet certified. They do not replace the frozen Neral capability suite.
