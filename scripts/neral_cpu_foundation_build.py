"""Build isolated verified CPU corpus and train-only files; no GPU/model/network use."""
import argparse
import json
from pathlib import Path
from neral_cpu_foundation.generate import save_generated
from neral_cpu_foundation.training import export_log_file

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="datasets/neral_cpu_foundation")
    parser.add_argument("--count",type=int,default=256)
    options=parser.parse_args()
    path=Path(options.output)
    examples=path/"synthetic_verified_256.jsonl"
    n=save_generated(examples,options.count)
    manifest=export_log_file(examples,path)
    if n<250 or manifest["verified"]<250: raise SystemExit(f"not enough verified samples: {n} / {manifest['verified']}")
    print(json.dumps(manifest,ensure_ascii=False,sort_keys=True))

if __name__=="__main__":main()
