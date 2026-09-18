"""One-shot, SYNTHETIC tiny-model positive native resume proof; never train real Neral.

Uses genuine torch AdamW moments and a real serialized checkpoint, then compares
one further CPU optimizer step with/without reload. No own weights, real corpora,
GPU allocation, persisted output, network, or model-quality claim.
"""
from __future__ import annotations

import argparse
from array import array
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if not (args.execute and args.cpu_only):
        raise ValueError('requires explicit --execute --cpu-only')

    import numpy as np
    import torch
    from model350 import Neral350Config, Neral350M
    from native_resume_preflight350 import (corpus_identity, inspect_checkpoint,
                                             read_verified_resume, run_contract)
    from recovery_identity import file_sha
    from window_sampler350 import SAMPLER_VERSION

    if torch.cuda.is_available():
        raise RuntimeError('this probe is CPU-only; hide CUDA devices before launch')
    torch.set_num_threads(2)
    torch.manual_seed(20260918)
    random.seed(20260918)
    np.random.seed(20260918)
    cfg = Neral350Config(vocab_size=128, d_model=64, n_layers=2, n_heads=4,
                         n_kv_heads=2, ffn_hidden=128, max_seq_len=64)
    model = Neral350M(cfg).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1, fused=True)

    def apply_cpu_step(mod, opt):
        opt.zero_grad(set_to_none=True)
        for p in mod.parameters():
            p.grad = torch.full_like(p, 0.0001)
        opt.step()

    apply_cpu_step(model, optimizer)  # Genuine, nonempty native PyTorch moment slots.
    saved_opt = optimizer.state_dict()
    assert saved_opt['state'] and len(saved_opt['state']) == len(list(model.parameters()))
    assert saved_opt['param_groups'][0]['fused'] is True
    contract = run_contract(seq=8, microbatch=1, grad_accum=1, seed=43,
                            lr=3e-4, schedule='cosine', weight_decay=0.1,
                            warmup_frac=0.01, gradient_checkpointing=False)
    with tempfile.TemporaryDirectory(prefix='phoenix-positive-resume-') as temp:
        root = Path(temp)
        train_dir = root / 'train'
        valid_dir = root / 'valid'
        output_dir = root / 'native-output'
        for folder in (train_dir, valid_dir, output_dir):
            folder.mkdir()
        (train_dir / 'train.bin').write_bytes(array('H', range(48)).tobytes())
        (valid_dir / 'valid.bin').write_bytes(array('H', range(48, 96)).tobytes())
        tokenizer = 'a' * 64  # Synthetic identity, never the real private tokenizer.
        train_id = corpus_identity([train_dir / 'train.bin'], seq=8, role='train',
                                   seed=44, tokenizer_sha=tokenizer)
        val_id = corpus_identity([valid_dir / 'valid.bin'], seq=8, role='valid',
                                 seed=1042, batch=1, batches=2,
                                 tokenizer_sha=tokenizer)
        sampler_rng = random.Random(44)
        ck = {
            'checkpoint_schema': 2, 'model': model.state_dict(),
            'optimizer': saved_opt, 'step': 1, 'processed_tokens': 8,
            'best_validation': 4.2, 'config': cfg.to_dict(),
            'target_tokens': 512, 'tokenizer_sha256': tokenizer,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': [],
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'train_sampler_rng': sampler_rng.getstate(),
            'train_identity': train_id, 'validation_identity': val_id,
            'sampler_version': SAMPLER_VERSION, 'run_contract': contract,
        }
        checkpoint = output_dir / 'resume.pt'
        torch.save(ck, checkpoint)
        checksum = file_sha(checkpoint)
        verified = read_verified_resume(
            checkpoint=checkpoint, expected_sha=checksum, trusted=True,
            out=output_dir, torch=torch, config=cfg.to_dict(),
            target_tokens=512, tokenizer_sha=tokenizer,
            train_identity=train_id, validation_identity=val_id,
            expected_run=contract, run_until=16)
        assert verified['processed_tokens'] == 8
        restored = Neral350M(cfg).cpu()
        restored_opt = torch.optim.AdamW(restored.parameters(), lr=3e-4,
                                         betas=(0.9, 0.95), eps=1e-8,
                                         weight_decay=0.1, fused=True)
        restored.load_state_dict(verified['model'], strict=True)
        restored_opt.load_state_dict(verified['optimizer'])
        sampler_replay = random.Random()
        sampler_replay.setstate(verified['train_sampler_rng'])
        assert sampler_replay.randrange(1000) == sampler_rng.randrange(1000)
        apply_cpu_step(model, optimizer)
        apply_cpu_step(restored, restored_opt)
        for before, after in zip(model.parameters(), restored.parameters()):
            assert torch.equal(before, after), 'CPU reload changed next optimizer update'
        reference_slots = optimizer.state_dict()['state']
        resumed_slots = restored_opt.state_dict()['state']
        assert reference_slots.keys() == resumed_slots.keys()
        for idx in reference_slots:
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                assert torch.equal(reference_slots[idx][key], resumed_slots[idx][key])

        tamper_cases = [
            ('weight_decay', 0.2), ('betas', (0.85, 0.95)),
            ('eps', 1e-7), ('fused', False), ('lr', float('nan')),
        ]
        for option, changed in tamper_cases:
            damaged = deepcopy(verified)
            damaged['optimizer']['param_groups'][0][option] = changed
            try:
                inspect_checkpoint(damaged, config=cfg.to_dict(),
                                   target_tokens=512, tokenizer_sha=tokenizer,
                                   train_identity=train_id,
                                   validation_identity=val_id,
                                   expected_run=contract, run_until=16)
            except ValueError as exc:
                assert 'AdamW' in str(exc), (option, str(exc))
            else:
                raise AssertionError(f'optimizer tampering accepted: {option}')
        assert file_sha(checkpoint) == checksum, 'synthetic source checkpoint altered'
        print(json.dumps({
            'event': 'synthetic_actual_torch_cpu_positive_resume',
            'optimizer_slots': len(saved_opt['state']),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'next_model_and_optimizer_state_bitwise_equal': True,
            'sampler_rng_replay_equal': True,
            'serialized_optimizer_tamper_cases_refused': len(tamper_cases),
            'checkpoint_sha_unchanged': True,
            'source': 'tiny_synthetic_random_model_not_real_350m',
            'cuda_used': False, 'real_training_started': False,
            'quality_improvement_verified': False,
        }, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
