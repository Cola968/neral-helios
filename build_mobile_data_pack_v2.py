from __future__ import annotations

"""Hardened entrypoint for the hosted Neral mobile feeder.

Fixes two issues observed in the first real hosted run:
1. FineWeb2 is multilingual and the old substring fallback could map "eng" to a non-English
   config such as ``asm_Beng``. English general web data now comes from monolingual FineWeb.
2. datasets/pyarrow can crash during CPython finalization after a successful streaming run on
   the hosted image. All files are already closed by ``build_mobile_data_pack.main``; this
   entrypoint exits directly after flushing successful output to avoid third-party finalizers.
"""

import os
import sys

import build_mobile_data_pack as feeder
from datasets import get_dataset_config_names


feeder.SOURCES = (
    feeder.SourceSpec("fineweb2_de", "HuggingFaceFW/fineweb-2", "de", 0.40, ("deu_Latn", "deu"), "ODC-By-1.0"),
    feeder.SourceSpec("fineweb_en", "HuggingFaceFW/fineweb", "en", 0.20, ("sample-10BT", "default"), "ODC-By-1.0"),
    feeder.SourceSpec("fineweb_edu", "HuggingFaceFW/fineweb-edu", "en", 0.25, ("sample-10BT", "default"), "ODC-By-1.0"),
    feeder.SourceSpec("finemath", "HuggingFaceTB/finemath", "en", 0.15, ("finemath-3plus", "finemath-4plus", "default"), "ODC-By-1.0"),
)


def strict_resolve_config(repo: str, revision: str, preferences: tuple[str, ...]) -> str:
    names = get_dataset_config_names(repo, revision=revision)
    if not names:
        raise RuntimeError(f"no configs for {repo}@{revision}")
    for pref in preferences:
        if pref in names:
            return pref
    for pref in preferences:
        p = pref.lower()
        for name in names:
            low = name.lower()
            if low == p or low.startswith(p + "_") or low.startswith(p + "-"):
                return name
    # FineWeb2 is multilingual: never guess its language config.
    if repo == "HuggingFaceFW/fineweb-2":
        raise RuntimeError(f"no safe language config match for {repo}: {preferences}")
    # The remaining configured sources are English-only or single-domain English datasets.
    return names[0]


feeder.resolve_config = strict_resolve_config


if __name__ == "__main__":
    try:
        feeder.main()
    except BaseException:
        raise
    else:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
