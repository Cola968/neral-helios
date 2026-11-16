"""Pinned integration CPU gate; no GPU, model inference or corpus generation."""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
# Source of truth for approved web/UI blobs: the actual main merge base.
# Using a hard-coded historical SHA breaks CI as soon as approved main evolves.
PROTECTED=(
  'neral_web_search.py','neral_web_api.py',
  'static/neral-preview.js','static/neral-web-search.js',
  'static/neral-web-live.js','static/neral-reference-ui.js',
  'templates/neral_preview.html',
)

def _trusted_main_base():
    # In PR CI, HEAD is GitHub's synthetic merge commit. Its merge-base with
    # the locally fetched origin/main is the approved main state. In isolated
    # integration branches it is the point where the branch diverged from main.
    # Missing remote tracking data fails closed; never silently accept HEAD.
    completed=subprocess.run(
        ['git','merge-base','HEAD','refs/remotes/origin/main'],
        cwd=ROOT,text=True,capture_output=True,timeout=15,check=True)
    sha=completed.stdout.strip()
    if re.fullmatch(r'[0-9a-f]{40}',sha) is None:
        raise RuntimeError('APPROVED_MAIN_BASE_UNAVAILABLE')
    return sha


def verify_main_integrity():
    base=_trusted_main_base()
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
