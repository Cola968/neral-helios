from __future__ import annotations
import argparse

import torch

from config import NeralConfig
from model import Neral20M
from tokenizer import ByteTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--prompt", default="User: Hallo!\nAssistant:")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = NeralConfig(**ckpt.get("config", {}))
    model = Neral20M(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    tok = ByteTokenizer()

    ids = torch.tensor([tok.encode(args.prompt, bos=True)], dtype=torch.long, device=device)
    out = model.generate(ids, args.tokens, args.temperature, args.top_k, eos_id=tok.eos_id)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
