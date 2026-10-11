from __future__ import annotations
import argparse,json,random
from pathlib import Path
import torch
from model350 import FRONTIER_350,Neral350M
from train_chat350 import list_shards,replay_batch

@torch.inference_mode()
def evaluate(checkpoint, replay_root, *, batches=12, batch_size=2, seq=512, seed=350729):
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if ck.get("config")!=FRONTIER_350.to_dict(): raise ValueError("checkpoint architecture mismatch")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 CUDA required")
    dev=torch.device("cuda");m=Neral350M().to(device=dev,dtype=torch.bfloat16);m.load_state_dict(ck["model"],strict=True);m.eval()
    shards=list_shards(Path(replay_root));rng=random.Random(seed);vals=[]
    for _ in range(batches):
        x,y=replay_batch(shards,batch_size,seq,dev,rng,m.cfg.vocab_size)
        _,loss=m(x,y);vals.append(float(loss))
    return {"batches":batches,"batch_size":batch_size,"seq":seq,"seed":seed,"mean_loss":sum(vals)/len(vals)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--replay-data",required=True);ap.add_argument("--out",required=True);ap.add_argument("--batches",type=int,default=12);ap.add_argument("--batch-size",type=int,default=2);ap.add_argument("--seq",type=int,default=512);a=ap.parse_args()
    r=evaluate(a.checkpoint,a.replay_data,batches=a.batches,batch_size=a.batch_size,seq=a.seq)
    Path(a.out).write_text(json.dumps(r,indent=2)+"\n",encoding="utf8");print(json.dumps(r))
if __name__=="__main__":main()
