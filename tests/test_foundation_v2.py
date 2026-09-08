import pytest
from architectures100_variants import EXPECTED,report
from neral100_checkpoint import init_audit,experimental_50m_migration_plan
from hf_curriculum50 import route
from public_eval_registry50 import validate_registry,assert_never_train
from hf_source_policy50 import assert_training_allowed

def test_100m_contracts_and_init():
 assert report('a1')['parameters']==EXPECTED['a1']==99_783_168
 assert report('a2')['parameters']==EXPECTED['a2']==100_470_912
 assert report('a2')['kv_cache_bf16_4096_bytes']==48*1024*1024
 a=init_audit('a2');assert a['finite'] and a['tied_embeddings']
def test_migration_stays_experimental():
 p=experimental_50m_migration_plan({'d_model':624},'a1');assert p['status']=='experimental_only' and p['production_pretraining']=='from_scratch'
def test_curriculum_routes_german_math():
 r=route(' Die Gleichung und die Ableitung werden Schritt für Schritt hergeleitet. ','HuggingFaceFW/fineweb-2','de');assert r['scores']['german']>0 and r['scores']['math']>0
def test_public_eval_registry_fail_closed():
 validate_registry()
 for ds in ('openai/gsm8k','google-research-datasets/mbpp','deepmind/code_contests'):
  assert assert_never_train(ds)
  with pytest.raises(ValueError):assert_training_allowed(ds)
