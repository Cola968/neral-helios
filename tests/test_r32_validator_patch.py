import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.neral_rank32_copy import patched_trainer, patched_validator

VALIDATOR_FIXTURE = '''\ndef validate_adapter(config):
    if (config.get('peft_type') != 'LORA' or config.get('task_type') != 'CAUSAL_LM'
            or config.get('r') != 16 or config.get('lora_alpha') != 16
            or config.get('lora_dropout') != 0 or config.get('bias') != 'none'):
        raise ValueError('Unsupported adapter configuration.')
    return True
'''
TRAINER_FIXTURE = '''    r=16, lora_alpha=16, lora_dropout=0,
    config = SFTConfig(
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    report = {"status": "started", "steps": args.steps, "max_length": args.max_length, "logs": [],
    samples = []
    report["metrics"] = trainer.train().metrics
    from neral_adapter_candidate import seal_adapter
'''

class ValidatorPatchTests(unittest.TestCase):
    def test_rank32_fixture_accepts_r32_and_rejects_r16(self):
        dst = patched_validator(VALIDATOR_FIXTURE)
        ns = {}
        exec(dst, ns)
        sample = dict(peft_type='LORA', task_type='CAUSAL_LM', r=32, lora_alpha=32, lora_dropout=0, bias='none')
        self.assertTrue(ns['validate_adapter'](sample))
        sample['r'] = 16
        with self.assertRaises(ValueError): ns['validate_adapter'](sample)
        sample['r'] = 32
        sample['lora_alpha'] = 16
        with self.assertRaises(ValueError): ns['validate_adapter'](sample)

    def test_patch_trainer_imports_rank32_validator_module(self):
        src = patched_trainer(TRAINER_FIXTURE)
        self.assertIn('from neral_adapter_candidate_r32 import seal_adapter', src)
        self.assertNotIn('from neral_adapter_candidate import seal_adapter', src)

    def test_patch_fails_closed_on_unknown_and_double_validator_anchors(self):
        with self.assertRaises(Exception): patched_validator(VALIDATOR_FIXTURE.replace("config.get('r') != 16", "config.get('r') != 17"))
        with self.assertRaises(Exception): patched_validator(VALIDATOR_FIXTURE + VALIDATOR_FIXTURE)

    def test_trainer_patch_fails_when_sealing_import_absent(self):
        with self.assertRaises(Exception): patched_trainer(TRAINER_FIXTURE.replace('from neral_adapter_candidate import seal_adapter', 'from other import seal_adapter'))

if __name__=='__main__': unittest.main()

class PowerShellPreflightContracts(unittest.TestCase):
    def test_includes_pinned_separate_r32_validator(self):
        content=(Path(__file__).resolve().parents[1]/'scripts/Start-NERAL-QLoRA-Action.ps1').read_text('utf-8')
        self.assertIn('--validator (Join-Path $src \'neral_adapter_candidate.py\')',content)
        self.assertIn('neral_adapter_candidate_r32.py',content)
        self.assertIn('--validator-output $rankValidator',content)
    def test_preflight_reports_path_and_pinned_stack(self):
        content=(Path(__file__).resolve().parents[1]/'scripts/Start-NERAL-QLoRA-Action.ps1').read_text('utf-8')
        self.assertIn('QWEN_MANIFEST_PATH=',content)
        self.assertIn('TRAINING_INTERPRETER=',content)
        self.assertIn('neral_training_preflight.py',content)
        self.assertIn('bitsandbytes/safetensors',content)
    def test_vram_hard_three_samples_transient_spike_allowed(self):
        content=(Path(__file__).resolve().parents[1]/'scripts/Start-NERAL-QLoRA-Action.ps1').read_text('utf-8')
        self.assertIn('$i -lt 3',content)
        self.assertIn('$free -lt 10000',content)
        self.assertIn('Count -ge 2',content)
