"""Fail-closed external Neral evaluation registry. These datasets are never training inputs."""
from dataclasses import dataclass,asdict
from hf_source_policy50 import SOURCES
@dataclass(frozen=True)
class EvalSpec:dataset_id:str;task:str;license:str;execution_policy:str
EVALS={'gsm8k':EvalSpec('openai/gsm8k','math_numeric_exact','MIT','model inference only; no training'),'mbpp':EvalSpec('google-research-datasets/mbpp','python_programming','CC-BY-4.0','generate on self-hosted GPU; execute only in isolated GitHub-hosted sandbox'),'codecontests':EvalSpec('deepmind/code_contests','competitive_programming','CC-BY-4.0','generation capture only until hardened sandbox exists')}
def validate_registry():
 for name,s in EVALS.items():
  p=SOURCES[s.dataset_id]
  if p.role!='public-eval-holdout' or p.allow_training:raise ValueError(f'{name}: not fail-closed')
  if p.license!=s.license:raise ValueError(f'{name}: license drift')
 return {k:asdict(v) for k,v in EVALS.items()}
def assert_never_train(dataset_id):
 if dataset_id not in {s.dataset_id for s in EVALS.values()}:raise ValueError('not registered')
 if SOURCES[dataset_id].allow_training:raise ValueError('public eval unexpectedly trainable')
 return True
if __name__=='__main__':
 import json;print(json.dumps(validate_registry(),indent=2))
