# NERAL V4.2 CPU-only offline knowledge (experimental)

Owner-only Kiwix service was installed and verified on loopback 9691 in run 45005018292.
The feature is enabled by default after owner-approved local Kiwix health verification; disable with
NERAL_V42_OFFLINE_RAG=0. Explicit web requests remain Web V4. Amber, sandbox,
attachments, private/personal queries, ambiguous follow-ups and benchmark_isolated
are unaffected. Automatic web fallback is opted into by this environment setting.
It applies to short public, standalone knowledge questions, not arbitrary private chat.
Freshness and title matching are conservative heuristics, not a fact verifier.

## Setup after Blind200

1. Download official Windows Kiwix tools from https://download.kiwix.org/release/kiwix-tools/
   and verify the archive SHA-256 against a trusted value. Either extract it manually
   and pass the EXE path, or use -ToolsZip and -ToolsSha256.
2. Place ZIMs from https://download.kiwix.org/zim/ in D:\neral-knowledge\zim.
   Prefer wikipedia_de_all_mini and one English mini/top_nopic archive, plus
   optional DevDocs/technology ZIMs if they fit. Select full-text indexed editions.
   Dates and actual sizes vary; 15 GB decimal is a hard ZIM cap, not a promise
   that any particular combination fits. Allow extra space for tools/temp files.
   No auto-download or package update occurs.
3. Run from the repository in a separate PowerShell:
   .\scripts\neral_v42_kiwix.ps1 -KiwixServe 'D:\neral-knowledge\tools\kiwix-serve.exe'
   .\scripts\neral_v42_kiwix.ps1 -HealthOnly
4. Only after approval, start NERAL with the environment flag set. Port 9690 is
   never touched by the setup script. Kiwix listens only on 127.0.0.1:9691.

Retrieval uses the Kiwix XML search API, fixed loopback HTTP, bounded content,
a single worker and a 450 ms caller wait. A slow in-flight socket may finish
later; no new local worker starts while it is busy. Results are relevance-filtered
and at most three introductions/infoboxes are offered as untrusted context.
One good source is accepted; no sources or fresh facts invoke the existing Web V4.
No GPU embedding model or CUDA library is imported by retrieval. Normal answer
generation still uses the selected existing model and its usual context/KV cache.
ZIM snapshots can be stale and are never independent proof; do not ingest the
offline route into the verified training/strategy stores. No external links
are followed by retrieval. Source snapshot book and URL are exposed in the answer.
Wikipedia content remains subject to its original licensing and attribution.

## Tests (off-host CPU, after review)

python -m unittest discover -s tests -p test_neral_v42_offline_rag.py -v
python -m py_compile neral_local_wiki.py neral_three_tier_runtime.py

Mock tests are not a real ZIM interoperability or latency/VRAM measurement.
Before enabling, test a real German and English indexed archive on 9691,
record cold/warm retrieval latency and measure process VRAM, verify the existing
Web V4 route and both tiers, and recheck the full CPU regression suite.
