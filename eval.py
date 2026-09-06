from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from config import NeralConfig
from model import Neral20M
from train import get_batch


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data")
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = NeralConfig(**ckpt.get("config", {}))
    model = Neral20M(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    val = np.memmap(Path(args.data) / "val.bin", dtype=np.uint16, mode="r")

    np.random.seed(42968)
    losses = []
    for _ in range(args.batches):
        x, y = get_batch(val, args.batch_size, args.seq_len, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and torch.cuda.is_bf16_supported())):
            _, loss = model(x, y)
        losses.append(loss.item())
    loss = float(np.mean(losses))
    print(f"validation loss: {loss:.6f}")
    print(f"bits per byte:   {loss / math.log(2):.6f}")
    print(f"perplexity:      {math.exp(min(loss, 20)):.3f}")


if __name__ == "__main__":
    main()
