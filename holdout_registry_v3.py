from __future__ import annotations

"""Central fail-closed registry for Neral internal and public evaluation holdouts.

This module does not expose benchmark answers to training. It gives data builders one place to
ask whether a dataset/file is reserved and to fingerprint local frozen-suite definitions so a
training manifest can prove which holdout contract it excluded.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

from hf_source_policy50 import SOURCES
from public_eval_registry50 import EVALS

REGISTRY_VERSION = 3


@dataclass(frozen=True)
class InternalHoldout:
    name: str
    path: str
    purpose: str
    training_allowed: bool = False


INTERNAL = {
    "rawmath": InternalHoldout("rawmath", "raw_math50_data.py", "frozen raw math ID/transfer/OOD definitions"),
    "nis2_chat": InternalHoldout("nis2_chat", "benchmark_nis2_chat50.py", "frozen instruction/dialogue/truthfulness/robustness/compositional suite"),
    "targeted": InternalHoldout("targeted", "benchmark_targeted50.py", "frozen code/German/math/reasoning targeted suite"),
    "long_context": InternalHoldout("long_context", "benchmark_v5_long_context.py", "frozen single/multi-needle and long-context BPB suite"),
    "v4": InternalHoldout("v4", "benchmark_v4.py", "frozen pair/context benchmark"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_snapshot(repo_root: str | Path = ".") -> dict:
    root = Path(repo_root).resolve()
    internal = {}
    for key, spec in INTERNAL.items():
        p = (root / spec.path).resolve()
        if p.parent != root:
            raise ValueError("internal holdout path escapes repository root")
        if not p.is_file():
            raise FileNotFoundError(f"frozen holdout definition missing: {spec.path}")
        internal[key] = {**asdict(spec), "sha256": _sha256(p)}
    public = {}
    for key, spec in EVALS.items():
        policy = SOURCES.get(spec.dataset_id)
        if policy is None or policy.role != "public-eval-holdout" or policy.allow_training:
            raise ValueError(f"public holdout policy drift: {spec.dataset_id}")
        public[key] = asdict(spec)
    canonical = {"version": REGISTRY_VERSION, "internal": internal, "public": public}
    canonical["registry_sha256"] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return canonical


def assert_dataset_ids_not_holdout(dataset_ids) -> bool:
    reserved = {spec.dataset_id for spec in EVALS.values()}
    bad = sorted(set(dataset_ids) & reserved)
    if bad:
        raise ValueError("evaluation holdout present in training inputs: " + ", ".join(bad))
    unknown = sorted(d for d in dataset_ids if d not in SOURCES)
    if unknown:
        raise ValueError("unregistered dataset IDs fail closed: " + ", ".join(unknown))
    blocked = sorted(d for d in dataset_ids if not SOURCES[d].allow_training)
    if blocked:
        raise ValueError("non-admitted dataset IDs in training inputs: " + ", ".join(blocked))
    return True


def assert_training_manifest(manifest: dict, repo_root: str | Path = ".") -> bool:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a dict")
    ids = manifest.get("dataset_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("training manifest must enumerate dataset_ids")
    assert_dataset_ids_not_holdout(ids)
    expected = registry_snapshot(repo_root)["registry_sha256"]
    if manifest.get("holdout_registry_sha256") != expected:
        raise ValueError("training manifest was not built against the current frozen holdout registry")
    if manifest.get("benchmark_rows_used_for_training") not in (False, 0):
        raise ValueError("benchmark rows may never be training inputs")
    return True


if __name__ == "__main__":
    print(json.dumps(registry_snapshot(), indent=2))
