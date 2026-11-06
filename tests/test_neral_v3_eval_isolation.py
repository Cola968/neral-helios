"""NERAL saved-adapter evaluation isolation: unit tests, CPU only."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.neral_v3_eval_isolation import (
    MODEL_KEYS, prepare_cuda, clean_cuda, collect_variant_outputs,
)

class FakeCuda:
    def __init__(self,free=10000,ready=True):
        self.free=free; self.ready=ready; self.events=[]
    def is_available(self):
        self.events.append('available')
        return self.ready
    def is_bf16_supported(self):
        self.events.append('bf16')
        return True
    def empty_cache(self):
        self.events.append('empty_cache')
    def mem_get_info(self,index):
        self.events.append('mem_get_info')
        return self.free*1024**2,12_288*1024**2
    def set_per_process_memory_fraction(self,fraction,index):
        self.events.append('limit')
        assert fraction==0.85 and index==0

class EvalOnlySafety(unittest.TestCase):
    def test_cleanup_before_memory_check_and_limit(self):
        fake=FakeCuda()
        self.assertEqual(prepare_cuda(SimpleNamespace(cuda=fake))['free_mib'],10000)
        self.assertLess(fake.events.index('empty_cache'),fake.events.index('mem_get_info'))
        self.assertLess(fake.events.index('mem_get_info'),fake.events.index('limit'))
    def test_low_vram_refuses(self):
        fake=FakeCuda(free=8900)
        with self.assertRaisesRegex(RuntimeError,'below 9000'):
            prepare_cuda(SimpleNamespace(cuda=fake))
        self.assertNotIn('limit',fake.events)
    def test_no_cuda_refuses(self):
        with self.assertRaisesRegex(RuntimeError,'CUDA BF16'):
            prepare_cuda(SimpleNamespace(cuda=FakeCuda(ready=False)))
    def test_cleanup_calls_allocator(self):
        fake=FakeCuda()
        clean_cuda(SimpleNamespace(cuda=fake))
        self.assertIn('empty_cache',fake.events)
    def test_separate_models_and_hashes_required(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in MODEL_KEYS:
                folder=root/model;folder.mkdir()
                row={'id':'v1','correct':True,'format_valid':True,
                     'answer_sha256':'a'*64,'local_raw':'ok'}
                record={'model':model,'results':{'validation64':[row]},
                        'adapter16_sha256':'b'*64,'adapter32_sha256':'c'*64,'validation_sha256':'d'*64}
                (folder/'local-outputs.json').write_text(json.dumps(record),encoding='utf8')
            data=collect_variant_outputs(root,{'validation64':1},'b'*64,'c'*64,'d'*64)
            self.assertEqual(set(data),set(MODEL_KEYS))
            with self.assertRaisesRegex(ValueError,'provenance'):
                collect_variant_outputs(root,{'validation64':1},'bad','c'*64,'d'*64)
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                collect_variant_outputs(root,{'validation64':2},'b'*64,'c'*64,'d'*64)
    def test_eval_workflow_never_runs_training(self):
        w=Path('.github/workflows/neral-v3-eval-only.yml').read_text(encoding='utf8')
        for v in ('EVALUATE_NERAL_V3_SAVED_80','--variant $variant','--combine-results',
                  'math-v3-rank-ab-42297190703-1','math-v3-two-line-42296646197-1',
                  'neral-gpu','nvidia-smi','HF_HUB_OFFLINE'):
            self.assertIn(v,w)
        for prohibited in ('--run-local-gpu','--steps 80','TRAIN_NERAL_V3_R16_R32_80_STEPS'):
            self.assertNotIn(prohibited,w)

if __name__=='__main__':
    unittest.main()
