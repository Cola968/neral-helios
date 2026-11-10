"""Reproducible, source-only NERAL 1.0 Local Preview release ZIP.

No local chats, state, model weights, adapters, secrets, cloud dependencies
or build-machine paths are included. The ZIP is a LOCAL PREVIEW, not a
verified general intelligence release or a self-contained offline model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

REQUIRED_FILES = (
    "START HIER.txt",
    "Start NERAL 1.0 Local.cmd",
    "Start NERAL 1.0 Stark.cmd",
    "Install NERAL Voice.cmd",
    "Install-Neral-Local-Voice.ps1",
    "requirements-neral-local-voice.txt",
    "scripts/neral_local_voice_install.py",
    "scripts/neral_local_voice_selftest.py",
    "neral_local_voice.py",
    "static/neral-local-voice.js",
    "static/neral-local-voice-worklet.js",
    "docs/NERAL_LOCAL_VOICE.md",
    "Start-Neral-1.0-Local.ps1",
    "neral_preview_server.py",
    "neral_auto_update.py",
    "neral_local_accounts.py",
    "neral_agent_autonomy.py",
    "neral_agent_sandbox.py",
    "solve.py",
    "neral_local_chat_api.py",
    "neral_preview_attachments.py",
    "neral_identity.py",
    "neral_agent_math.py",
    "neral_agent_python_trace.py",
    "neral_agent_quantitative.py",
    "neral_quantitative_oracles.py",
    "neral_logic_engine.py",
    "neral_logic_transfer.py",
    "neral_percentage_tool.py",
    "neral_think.py",
    "neral_numeric_final.py",
    "neral_high_verified_first.py",
    "neral_tir_experimental.py",
    "neral_formal_constraints.py",
    "neral_web_search.py",
    "neral_web_api.py",
    "agent_mode.py",
    "static/neral-preview.js",
    "static/neral-web-live.js",
    "static/neral-live-stream.js",
    "static/neral-accounts.js",
    "static/neral-accounts.css",
    "static/neral-preview.css",
    "static/neral-modern.css",
    "static/neral-modern.js",
    "static/neral-dynamic-ui.js",
    "static/neral-dynamic-ui.css",
    "static/neral-math.js",
    "static/neral-math.css",
    "static/neral-formulas.js",
    "static/neral-formulas.css",
    "static/vendor/katex/katex.min.js",
    "static/vendor/katex/katex.min.css",
    "static/vendor/katex/auto-render.min.js",
    "static/vendor/katex/mhchem.min.js",
    "static/vendor/katex/LICENSE.txt",
    "static/vendor/katex/VERSION.txt",
    "static/vendor/katex/fonts/KaTeX_AMS-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Caligraphic-Bold.woff2",
    "static/vendor/katex/fonts/KaTeX_Caligraphic-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Fraktur-Bold.woff2",
    "static/vendor/katex/fonts/KaTeX_Fraktur-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Main-Bold.woff2",
    "static/vendor/katex/fonts/KaTeX_Main-BoldItalic.woff2",
    "static/vendor/katex/fonts/KaTeX_Main-Italic.woff2",
    "static/vendor/katex/fonts/KaTeX_Main-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Math-BoldItalic.woff2",
    "static/vendor/katex/fonts/KaTeX_Math-Italic.woff2",
    "static/vendor/katex/fonts/KaTeX_SansSerif-Bold.woff2",
    "static/vendor/katex/fonts/KaTeX_SansSerif-Italic.woff2",
    "static/vendor/katex/fonts/KaTeX_SansSerif-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Script-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Size1-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Size2-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Size3-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Size4-Regular.woff2",
    "static/vendor/katex/fonts/KaTeX_Typewriter-Regular.woff2",
    "static/neral-reference-ui.css",
    "static/neral-reference-ui.js",
    "static/neral-clean-ui.css",
    "static/neral-clean-ui.js",
    "static/neral-motion.js",
    "static/neral-motion.css",
    "static/icon.svg",
    "templates/neral_preview.html",
    "requirements-neral-preview.txt",
    "docs/NERAL_1_0_LOCAL_PREVIEW_RC.md",
)
PREFIX="NERAL-1.0-Local"
TIMESTAMP=(2026,11,8,0,0,0)


def build_preview(root: Path, output: Path) -> dict:
    root=Path(root).resolve(strict=True)
    # Build-time dependency only. End-user app and release run fully offline.
    # The pinned, script-free npm package is converted into a reviewed allowlist.
    from scripts.vendor_katex import vendor
    vendor(root)
    output=Path(output)
    if (output.exists() or output.is_symlink() or not output.parent.is_dir() or
            output.parent.is_symlink() or output.suffix.lower()!=".zip"):
        raise ValueError("Create-only ZIP path in existing directory required")
    hashes={}
    # Snapshot precise reviewed allowlist only; fail on untrusted file links.
    files=[]
    for relative in REQUIRED_FILES:
        if (not re.fullmatch(r"[A-Za-z0-9_./ -]{1,160}",relative) or
                PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts):
            raise ValueError("Untrusted archive path")
        source=root.joinpath(*PurePosixPath(relative).parts)
        limit=800000 if relative=="static/vendor/katex/katex.min.js" else 200000
        if (source.is_symlink() or not source.is_file() or
                not source.resolve().is_relative_to(root) or
                not 0<source.stat().st_size<limit):
            raise ValueError("Missing, untrusted or oversized release source file")
        raw=source.read_bytes()
        # The only binary data allowed are the explicitly enumerated,
        # locally packaged WOFF2 fonts. No model weights or arbitrary binaries.
        font=relative.startswith("static/vendor/katex/fonts/") and relative.endswith(".woff2")
        if font:
            if not raw.startswith(b"wOF2"):
                raise ValueError("Unexpected KaTeX WOFF2 font format")
        elif b"\x00" in raw:
            raise ValueError("No binary assets or models in release ZIP")
        files.append((relative,raw))
        hashes[relative]=hashlib.sha256(raw).hexdigest()
    manifest={
        "schema":1,
        "name":"NERAL 1.0 Local Preview",
        "status":"RELEASE_CANDIDATE_NOT_PROMOTED",
        "model":"qwen3:8b",
        "model_origin":"local_foundation_not_neral_adapter",
        "model_weights_included":False,
        "release_validated":False,
        "weights_promoted":False,
        "files_sha256":hashes,
    }
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9,strict_timestamps=True) as bundle:
        for relative,raw in files+[
            ("RELEASE-MANIFEST.json",
             (json.dumps(manifest,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode("utf-8"))]:
            entry=zipfile.ZipInfo(f"{PREFIX}/{relative}",date_time=TIMESTAMP)
            entry.compress_type=zipfile.ZIP_DEFLATED
            entry.external_attr=0o100644 << 16
            entry.create_system=3
            bundle.writestr(entry,raw,compress_type=zipfile.ZIP_DEFLATED,
                            compresslevel=9)
    return {
        "status":"NERAL_1_0_PREVIEW_ARCHIVE_BUILT",
        "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes":output.stat().st_size,
        "source_files":len(REQUIRED_FILES),
        "model_weights_included":False,
        "release_validated":False,
        "weights_promoted":False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build_preview(args.root,args.out),sort_keys=True))


if __name__=="__main__":
    main()
