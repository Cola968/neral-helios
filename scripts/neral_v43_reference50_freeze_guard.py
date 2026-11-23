"""CPU gate: the actual NERAL 1.0 runtime staged for reference50 is frozen.

The V4.3 approval/orchestration scripts may change, but every file deployed
by the frozen owner manifest must match the source-release SHA byte-for-byte.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import argparse

FILES=(
    "neral_v41_context.py","neral_verified_notes.py",
    "neral_evidence_agent.py","neral_page_reader.py","neral_web_search.py",
    "neral_quantity_plans.py","neral_z3_json_compiler.py",
    "neral_v32_verifier.py","neral_knowledge_verifier.py",
    "neral_obsidian_deep.py","neral_local_wiki.py",
    "neral_v42_deadline.py","neral_v42_telemetry.py","neral_v42_quality.py",
    "neral_three_tier_runtime.py","neral_preview_server.py",
    "templates/neral_preview.html",
    "static/neral-preview.js","static/neral-reference-ui.js",
)


def verify_paths(candidate:Path,frozen:Path):
    for item in FILES:
        left=candidate/item;right=frozen/item
        if (not left.is_file() or left.is_symlink()
                or not right.is_file() or right.is_symlink()):
            raise ValueError("Release-file missing or symlinked")
        if sha256(left.read_bytes()).digest()!=sha256(right.read_bytes()).digest():
            raise ValueError("Candidate runtime differs from frozen 1.0 source")
    return len(FILES)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--frozen",type=Path,required=True)
    args=parser.parse_args()
    n=verify_paths(args.candidate,args.frozen)
    print("NERAL_V43_FROZEN_RUNTIME_19_OF_19_SHA256_EQUAL "+str(n),flush=True)


if __name__=="__main__":
    main()
