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
        # Only exact audited PR558/PR561 blob SHAs are accepted for the
        # web-search and status-UI updates; no arbitrary protected-file drift.
        approved_blobs = {
            'neral_web_search.py': (
                'd9a34976abe4824b2ef7c7ae65613b7f38ba5d40',
                '08ae0fc8264fc9ace68f7e9a9a14a50a5b9c652c',
            ),
            'neral_web_api.py': 'a4d621b318666d8521a758e7699ee19f408756ee',
            'static/neral-web-live.js': 'a0e30ebf2dbca66a25556a07740b5ac93bb4c472',
            'static/neral-preview.js': (
                '666a9e4b2d2467f6397a16b107a709dcf34746c2',
                'ce535f84c2e779d6619a968ca1e9b2582d20edbf',
                '77e124c98e61c54f2427269e840a21de7feee03e',
                'b886736b5be3cccf2fb4657124ff8ca4fef2f1a2',
                '42dcc9338bf5701ed3d982848c3db5ebe240b0b5',
            ),
            'static/neral-reference-ui.js': 'a28912ebcc43892555c8a30b7a75663252c76f10',
        }
        approved = approved_blobs.get(name)
        allowed_pinned_patch = (base == CURRENT_MAIN and approved is not None
                         and current in (approved if isinstance(approved, tuple)
                                         else (approved,)))
        if (old!=current and not allowed_pinned_patch) or not old:
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
