"""Pinned integration CPU gate; no GPU, model inference or corpus generation."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='2de4b034d79521e4f5c9d432af7f98a70087ed70'
# Updated main from before this PR. Protect its newer UI exactly on merge commits.
CURRENT_MAIN='f7affa3ff06b018bd7460e2802be4b19cd67a265'
PROTECTED=(
  'neral_web_search.py','neral_web_api.py',
  'static/neral-preview.js','static/neral-web-search.js',
  'static/neral-web-live.js','static/neral-reference-ui.js',
  'templates/neral_preview.html',
)

def _protected_base():
    """Keep the original feature branch and the merged PR integrity-pinned.

    A feature checkout descends from BASE; a PR merge checkout descends from
    CURRENT_MAIN. Only the corresponding immutable source blobs are accepted.
    Never infer a baseline from mutable branch names or untrusted CI input.
    """
    for sha in (CURRENT_MAIN, BASE):
        result=subprocess.run(
            ['git','merge-base','--is-ancestor',sha,'HEAD'],
            cwd=ROOT,capture_output=True,timeout=15,check=False)
        if result.returncode==0:
            return sha
        if result.returncode!=1:
            raise RuntimeError('INTEGRITY_BASE_UNKNOWN: incomplete git history')
    raise RuntimeError('INTEGRITY_BASE_UNKNOWN: unrecognized ancestry')


def verify_main_integrity():
    base=_protected_base()
    for name in PROTECTED:
        # Compare *repository blobs*, not checked-out bytes: Windows Git may
        # materialize LF source as CRLF without changing the tracked content.
        # Also refuse real uncommitted changes in the checked-out sources.
        old=subprocess.run(
          ['git','rev-parse',base+':'+name],cwd=ROOT,text=True,
          capture_output=True,timeout=15,check=True).stdout.strip()
        current=subprocess.run(
          ['git','rev-parse','HEAD:'+name],cwd=ROOT,text=True,
          capture_output=True,timeout=15,check=True).stdout.strip()
        if old!=current or not old:
            raise RuntimeError('CURRENT_MAIN_WEB_OR_UI_CHANGED: '+name)
        clean=subprocess.run(
          ['git','diff','--quiet','HEAD','--',name],
          cwd=ROOT,capture_output=True,timeout=15,check=False)
        if clean.returncode!=0:
            raise RuntimeError('WORKTREE_DRIFT: '+name)
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
