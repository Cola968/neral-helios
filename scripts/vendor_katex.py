"""Create the exact, offline KaTeX browser assets used in NERAL releases.

Run before local preview ZIP or Windows EXE packaging, not at inference time.
Uses the pinned npm package, ignores package scripts, only allows listed dist
files and WOFF2 fonts, and never copies an unrelated npm dependency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

VERSION = "0.19.1"
REL = Path("static/vendor/katex")
FONT_NAMES = (
    "KaTeX_AMS-Regular",
    "KaTeX_Caligraphic-Bold",
    "KaTeX_Caligraphic-Regular",
    "KaTeX_Fraktur-Bold",
    "KaTeX_Fraktur-Regular",
    "KaTeX_Main-Bold",
    "KaTeX_Main-BoldItalic",
    "KaTeX_Main-Italic",
    "KaTeX_Main-Regular",
    "KaTeX_Math-BoldItalic",
    "KaTeX_Math-Italic",
    "KaTeX_SansSerif-Bold",
    "KaTeX_SansSerif-Italic",
    "KaTeX_SansSerif-Regular",
    "KaTeX_Script-Regular",
    "KaTeX_Size1-Regular",
    "KaTeX_Size2-Regular",
    "KaTeX_Size3-Regular",
    "KaTeX_Size4-Regular",
    "KaTeX_Typewriter-Regular",
)
ASSETS = (
    "katex.min.js",
    "katex.min.css",
    "auto-render.min.js",
    "LICENSE.txt",
    "VERSION.txt",
) + tuple("fonts/" + name + ".woff2" for name in FONT_NAMES)


def check_assets(root: Path) -> None:
    destination = Path(root) / REL
    if (destination / "VERSION.txt").read_text("ascii").strip() != VERSION:
        raise RuntimeError("NERAL KaTeX version does not match pinned version")
    for name in ASSETS:
        file = destination / name
        size_limit = 800_000 if name == "katex.min.js" else 200_000
        if file.is_symlink() or not file.is_file() or not 0 < file.stat().st_size < size_limit:
            raise RuntimeError(f"Missing, unsafe or oversized local KaTeX asset: {name}")
    css = (destination / "katex.min.css").read_text("utf8")
    for font in FONT_NAMES:
        if font not in css:
            raise RuntimeError("Bundled KaTeX CSS lacks required local font: " + font)
    if len(list((destination / "fonts").glob("*.woff2"))) != len(FONT_NAMES):
        raise RuntimeError("Unexpected count of KaTeX WOFF2 font assets")


def vendor(root: Path) -> None:
    root = Path(root).resolve()
    destination = root / REL
    if destination.is_dir():
        try:
            check_assets(root)
            print("NERAL: locally bundled KaTeX " + VERSION + " already complete.")
            return
        except (RuntimeError, FileNotFoundError):
            pass
    command = shutil.which("npm.cmd") if shutil.which("npm.cmd") else shutil.which("npm")
    if not command:
        raise RuntimeError("Node.js/npm is required to vendor KaTeX while building (not at runtime).")
    with tempfile.TemporaryDirectory(prefix="neral-pinned-katex-") as tmp:
        tmp_path = Path(tmp)
        subprocess.run([
            command, "install", "--prefix", str(tmp_path),
            "--no-save", "--no-package-lock", "--ignore-scripts",
            "--no-audit", "--no-fund", "--no-progress",
            "katex@" + VERSION,
        ], check=True, timeout=180)
        base = tmp_path / "node_modules" / "katex"
        package = json.loads((base / "package.json").read_text("utf8"))
        if package.get("version") != VERSION:
            raise RuntimeError("Downloaded KaTeX version mismatch")
        sources = {
            "katex.min.js": base / "dist/katex.min.js",
            "katex.min.css": base / "dist/katex.min.css",
            "auto-render.min.js": base / "dist/contrib/auto-render.min.js",
            "LICENSE.txt": base / "LICENSE",
        }
        for name in FONT_NAMES:
            sources[f"fonts/{name}.woff2"] = base / f"dist/fonts/{name}.woff2"
        # Copy only explicit, audited paths. No JavaScript build hooks execute.
        destination.mkdir(parents=True, exist_ok=True)
        for name, src in sources.items():
            if src.is_symlink() or not src.is_file():
                raise RuntimeError("Missing pinned KaTeX asset: " + str(src))
            dest = destination / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        (destination / "VERSION.txt").write_text(VERSION + "\n", encoding="ascii")
    check_assets(root)
    print("NERAL: KaTeX " + VERSION + " with offline WOFF2 fonts bundled.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        check_assets(args.root)
        print("NERAL KaTeX offline assets verified.")
    else:
        vendor(args.root)


if __name__ == "__main__":
    main()
