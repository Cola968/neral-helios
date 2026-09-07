"""A short-answer subset for a bounded diagnostic intervention, not main Stage D.

Uses only existing, contamination-screened SFT splits; never moves rows across splits.
Whole UTF-8 answers and the terminal EOS must fit. No synthetic model answers.
"""
import hashlib
import json
from pathlib import Path
from train_instruct50 import encode_sft


def main():
    source=Path('sft50_data');out=Path('reports/termination_probe_data');out.mkdir(exist_ok=True)
    manifest={'purpose':'Isolated short-answer/EOS intervention on B1; not promotion or main Stage D','sequence':256,'splits':{},'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()}
    for split in ('train','val','test','holdout'):
        rows=[]
        for line in (source/(split+'.jsonl')).read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            try:encode_sft(row,256)
            except ValueError:continue
            rows.append(row)
        path=out/(split+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
        manifest['splits'][split]={'examples':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps(manifest))


if __name__=='__main__':main()
