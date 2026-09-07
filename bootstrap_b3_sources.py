"""Stage a larger provenance-controlled public corpus for Neral-50M B3.

The source families intentionally match B2's broad DE/EN/education mix, but this script stages
substantially more upstream documents. prepare_pretrain50_b3.py then excludes every normalized
B2 document hash before selecting disjoint B3 phase-A/B documents.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

from bootstrap_b2_sources import ODC_LICENSE, ODC_LICENSE_URL, stream_public_source

SPECS=(
    {"id":"fineweb-en-general","repo":"HuggingFaceFW/fineweb","config":"sample-10BT","revision":"v1.4.0","language":"en","domain":"general","target_fraction":.40,"staging_bytes":300_000_000,"license":ODC_LICENSE,"license_url":ODC_LICENSE_URL,"provenance_url":"https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/v1.4.0"},
    {"id":"fineweb2-de-general","repo":"HuggingFaceFW/fineweb-2","config":"deu_Latn","revision":"v2.1.0","language":"de","domain":"general","target_fraction":.40,"staging_bytes":330_000_000,"license":ODC_LICENSE,"license_url":ODC_LICENSE_URL,"provenance_url":"https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/tree/v2.1.0"},
    {"id":"fineweb-edu-en","repo":"HuggingFaceFW/fineweb-edu","config":"sample-10BT","revision":"v1.4.0","language":"en","domain":"education","target_fraction":.20,"staging_bytes":180_000_000,"license":ODC_LICENSE,"license_url":ODC_LICENSE_URL,"provenance_url":"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/v1.4.0"},
)


def bootstrap(root: Path,scale=1.0):
    if not .5<=scale<=3.0:raise ValueError('scale must be 0.5..3.0')
    root.mkdir(parents=True,exist_ok=True);sources=[]
    for spec in SPECS:
        path=root/f"{spec['id']}.jsonl";report=stream_public_source(spec,path,round(spec['staging_bytes']*scale))
        sources.append({k:v for k,v in spec.items() if k not in {'repo','config','staging_bytes'}}|{'file':path.name,'sha256':report['sha256'],'dataset_repo':spec['repo'],'dataset_config':spec['config'],'staged_text_bytes':report['text_bytes'],'staged_documents':report['documents']})
        print(json.dumps({'source':spec['id'],**report},ensure_ascii=False),flush=True)
    if not math.isclose(sum(s['target_fraction'] for s in sources),1.0,abs_tol=1e-9):raise AssertionError('B3 fractions must sum to 1')
    m={'version':1,'bootstrap_version':1,'purpose':'Neral-50M B3 disjoint public-data staging','sources':sources,
       'notes':['Raw source files remain local.','prepare_pretrain50_b3.py excludes all normalized B2 document hashes before phase selection.','FineWeb-family datasets are ODC-By-1.0 and derived from Common Crawl; preserve source attribution and applicable Common Crawl terms.']}
    (root/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False),encoding='utf8');return m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='b3_sources');ap.add_argument('--scale',type=float,default=1.0);a=ap.parse_args();bootstrap(Path(a.out),a.scale)

if __name__=='__main__':main()
