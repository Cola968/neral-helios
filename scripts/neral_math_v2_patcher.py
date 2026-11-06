"""Make a fail-closed, non-overwriting scratch trainer variant (lower LR)."""
import argparse
import hashlib
from pathlib import Path

EXPECTED_GIT_BLOB='f4b0f72440ce40653ca0ce2b171bec6721505e18'
OLD='learning_rate=2e-4, bf16=True'
NEW='learning_rate=5e-5, bf16=True'

def blob_sha(raw:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()

def patch(source:Path,destination:Path):
    if destination.exists() or source.is_symlink() or destination.is_symlink():raise ValueError('Unsafe scratch trainer destination')
    data=source.read_bytes()
    if blob_sha(data)!=EXPECTED_GIT_BLOB:raise ValueError('Pinned QLoRA trainer changed or CRLF-translated')
    text=data.decode('utf8')
    if text.count(OLD)!=1:raise ValueError('Learning rate anchor changed')
    patched=text.replace(OLD,NEW,1)
    destination.write_text(patched,encoding='utf8',newline='\n')
    return {'source_blob':EXPECTED_GIT_BLOB,'source_sha256':hashlib.sha256(data).hexdigest(),
            'patched_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'learning_rate':0.00005,'original_unmodified':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();print(patch(a.source,a.out))
