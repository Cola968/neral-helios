"""Fail-closed external Neral evaluation registry. These datasets are never training inputs."""
from dataclasses import dataclass, asdict

from hf_source_policy50 import SOURCES


@dataclass(frozen=True)
class EvalSpec:
    dataset_id: str
    task: str
    license: str
    execution_policy: str


EVALS = {
    "gsm8k": EvalSpec("openai/gsm8k", "math_numeric_exact", "MIT", "model inference only; no training"),
    "mbpp": EvalSpec("google-research-datasets/mbpp", "python_programming", "CC-BY-4.0", "generate on self-hosted GPU; execute only in isolated GitHub-hosted sandbox"),
    "codecontests": EvalSpec("deepmind/code_contests", "competitive_programming", "CC-BY-4.0", "generation capture only until hardened sandbox exists"),
    "ifeval": EvalSpec("google/IFEval", "instruction_following", "Apache-2.0", "external held-out evaluation only; never training or checkpoint selection"),
    "truthfulqa": EvalSpec("truthfulqa/truthful_qa", "truthfulness", "Apache-2.0", "external held-out evaluation only; report separately from NIS2 internal suite"),
    "boolq": EvalSpec("google/boolq", "reading_comprehension", "CC-BY-SA-3.0", "external held-out evaluation only; preserve upstream attribution/license"),
    "arc": EvalSpec("allenai/ai2_arc", "science_reasoning", "CC-BY-SA-4.0", "external held-out evaluation only; Challenge/Easy config must be reported explicitly"),
}


def validate_registry():
    for name, spec in EVALS.items():
        policy = SOURCES[spec.dataset_id]
        if policy.role != "public-eval-holdout" or policy.allow_training:
            raise ValueError(f"{name}: not fail-closed")
        if policy.license != spec.license:
            raise ValueError(f"{name}: license drift")
    return {k: asdict(v) for k, v in EVALS.items()}


def assert_never_train(dataset_id):
    if dataset_id not in {s.dataset_id for s in EVALS.values()}:
        raise ValueError("not registered")
    if SOURCES[dataset_id].allow_training:
        raise ValueError("public eval unexpectedly trainable")
    return True


if __name__ == "__main__":
    import json
    print(json.dumps(validate_registry(), indent=2))
