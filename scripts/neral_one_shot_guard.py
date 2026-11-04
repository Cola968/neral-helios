"""Permanently claim one authorized NERAL QLoRA r32 smoke per Windows runner profile.

Pure stdlib, atomic creation, no model access or network. An interrupted run still
consumes the authorization. Recovery requires explicit human inspection, not retry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

MARKER = '.neral-r32-three-step-one-shot.lock'


class OneShotError(ValueError):
    pass


def claim(root: Path, run_id: str, attempt: str) -> Path:
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or root.resolve() != root.absolute():
        raise OneShotError('Invalid existing runner profile root')
    if not isinstance(run_id,str) or not re.fullmatch(r'[0-9]{6,24}',run_id):
        raise OneShotError('Invalid GitHub run id')
    if not isinstance(attempt,str) or not re.fullmatch(r'[1-9][0-9]{0,2}',attempt):
        raise OneShotError('Invalid GitHub run attempt')
    payload={'schema':1,'action':'r32_qlora_three_optimizer_steps',
             'run_id':run_id,'attempt':int(attempt)}
    marker=root/MARKER
    # O_CREAT|O_EXCL is atomic even for competing invocations; symlinks also fail.
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    if hasattr(os,'O_BINARY'):
        flags |= os.O_BINARY
    try:
        fd=os.open(marker,flags,0o600)
    except FileExistsError as exc:
        raise OneShotError('One-shot authorization already consumed or unsafe') from exc
    with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(payload,sort_keys=True)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    return marker


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--attempt',required=True)
    args=p.parse_args(argv)
    try:
        claim(args.root,args.run_id,args.attempt)
    except (OneShotError,OSError,ValueError) as exc:
        p.exit(2,f'NERAL pilot refused: {exc}\n')
    print('NERAL rank32 one-shot training authorization consumed; no automatic retries.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
