from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from train_mobile_data_pilot50_v2 import PackedMobilePack


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_pack(root: Path, seq: int = 8) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Three train documents: two are deliberately shorter than seq+1. The validated shard is
    # contiguous and already contains boundary tokens, so packed training can still use them.
    docs = [
        ("train", [1, 11, 12, 2]),
        ("train", [1, 21, 22, 23, 2]),
        ("train", [1, 31, 32, 33, 34, 35, 36, 37, 38, 39, 2]),
        ("val",   [1, 41, 42, 43, 44, 45, 46, 47, 48, 49, 2]),
    ]
    train = np.asarray(sum((ids for split, ids in docs if split == "train"), []), dtype="<u2")
    val = np.asarray(sum((ids for split, ids in docs if split == "val"), []), dtype="<u2")
    train_path = root / "train_000.bin"; val_path = root / "val_000.bin"
    train.tofile(train_path); val.tofile(val_path)

    offsets = {"train": 0, "val": 0}; rows = []
    for i, (split, ids) in enumerate(docs):
        start = offsets[split]; end = start + len(ids); offsets[split] = end
        rows.append({
            "id": f"d{i}", "source": "unit", "repo": "unit", "revision": "x",
            "config": "x", "language": "en", "split": split,
            "shard": f"{split}_000.bin", "start": start, "end": end,
            "tokens": len(ids), "utf8_bytes": max(1, len(ids)-2),
        })
    (root / "documents.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf8")
    manifest = {
        "candidate_only": True,
        "foreign_model_weights_used": False,
        "train_shards": [{"file": train_path.name, "tokens": len(train), "sha256": _sha(train_path)}],
        "val_shards": [{"file": val_path.name, "tokens": len(val), "sha256": _sha(val_path)}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")


def test_packed_training_recovers_short_documents(tmp_path: Path):
    _write_pack(tmp_path)
    pack = PackedMobilePack(tmp_path, seq=8)
    # V1 document-bounded training would lose the first two train documents completely.
    assert len(pack.train_blocks) >= 2
    # The first packed block spans the first and second document boundary.
    shard, start = pack.train_blocks[0]
    window = list(map(int, pack.maps[shard][start:start+9]))
    assert window[:4] == [1, 11, 12, 2]
    assert window[4] == 1
    assert pack.utilization["packed_target_position_ratio"] > 0.5


def test_validation_remains_document_bounded(tmp_path: Path):
    _write_pack(tmp_path)
    pack = PackedMobilePack(tmp_path, seq=8)
    assert pack.val_blocks
    shard, start = pack.val_blocks[0]
    assert shard == "val_000.bin"
    assert start >= 0 and start + 8 < len(pack.maps[shard])
