"""Pinned integration CPU gate; no GPU, model inference or corpus generation."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='2de4b034d79521e4f5c9d432af7f98a70087ed70'
PROTECTED=(
  'neral_web_search.py','neral_web_api.py',
  'static/neral-preview.js','static/neral-web-search.js',
  'static/neral-web-live.js','static/neral-reference-ui.js',
  'templates/neral_preview.html',
)

def verify_main_integrity():
    for name in PROTECTED:
        original=subprocess.run(
          ['git','show',BASE+':'+name],cwd=ROOT,capture_output=True,
          timeout=15,check=True).stdout
        current=(ROOT/name).read_bytes()
        if original!=current:
            raise RuntimeError('CURRENT_MAIN_WEB_OR_UI_CHANGED: '+name)
    print('MAIN_1_0_WEB_UI_SECURITY_BLOBS_PRESERVED',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tests',action='store_true')
    args=parser.parse_args()
    verify_main_integrity()
    if args.tests:
        cases=(ROOT/'docs/NERAL_1_1_MAIN_MERGE_TESTS.txt').read_text(
          encoding='utf8').splitlines()
        if not 60<=len(cases)<=90 or len(cases)!=len(set(cases)):
            raise RuntimeError('Unreviewed integration test inventory')
        if any(not x.startswith('tests/') or not (ROOT/x).is_file()
               for x in cases):
            raise RuntimeError('Missing integration test contract')
        command=[sys.executable,'-B','-m','pytest','-q','--color=no',*cases]
        raise SystemExit(subprocess.run(command,cwd=ROOT,check=False).returncode)

if __name__=='__main__': main()
