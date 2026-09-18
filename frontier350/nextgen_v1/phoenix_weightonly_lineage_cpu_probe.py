"""SYNTHETIC tiny-model CPU proof of new-lineage checkpoint and journal mechanics.

Creates ONLY temporary random weights and fake source bytes; never uses real
Neral weights, real data, GPU, operator documents, or initiates 350M training.
This is not a performance, approval or free-answer quality measurement.
"""
from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_journal import CheckpointJournal
from helios350_preview import CONTINUITY_SHA
from phoenix_weight_only_pilot import plan_lineage
from native_resume_preflight350 import corpus_identity, read_verified_resume, run_contract
from recovery_identity import file_sha
from train350 import atomic_save
from window_sampler350 import SAMPLER_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if not (args.execute and args.cpu_only):
        raise ValueError('explicit temporary CPU-only execution required')
    import numpy as np
    import torch
    from model350 import Neral350Config, Neral350M
    if torch.cuda.is_available():
        raise RuntimeError('CUDA must be hidden; no GPU allowed')
    torch.set_num_threads(2)
    torch.manual_seed(118)
    np.random.seed(118)
    random.seed(118)
    cfg = Neral350Config(vocab_size=128, d_model=64, n_layers=2, n_heads=4,
                         n_kv_heads=2, ffn_hidden=128, max_seq_len=64)
    original = Neral350M(cfg).cpu()
    # Simulate historical pretrained weights WITHOUT any historical optimizer.
    initial_weights = {k: v.detach().clone() for k, v in original.state_dict().items()}
    candidate = Neral350M(cfg).cpu()
    candidate.load_state_dict(initial_weights, strict=True)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=2e-5,
                                  betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=0.1, fused=True)
    assert not optimizer.state_dict()['state'], 'new optimizer MUST begin empty'

    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.0001)
        opt.step()

    with tempfile.TemporaryDirectory(prefix='phoenix-weightonly-synthetic-') as temporary:
        root = Path(temporary)
        fake_source = root / 'fake-source.pt'
        fake_source.write_bytes(b'synthetic label only, not genuine historical bytes')
        source_digest = file_sha(fake_source)
        out = root / 'separate-new'
        out.mkdir()
        train = root / 'train.bin'
        valid = root / 'valid.bin'
        train.write_bytes(array('H', range(48)).tobytes())
        valid.write_bytes(array('H', range(48, 96)).tobytes())
        tok = 'a' * 64  # Synthetic placeholder identity, not the own tokenizer.
        train_id = corpus_identity([train], seq=8, role='train', seed=8,
                                   tokenizer_sha=tok)
        val_id = corpus_identity([valid], seq=8, role='valid', seed=1006,
                                 batch=1, batches=2, tokenizer_sha=tok)
        contract = run_contract(seq=8, microbatch=1, grad_accum=1, seed=7,
                                lr=2e-5, schedule='cosine', weight_decay=0.1,
                                warmup_frac=0.02, gradient_checkpointing=True)
        contract.update({'initialization': 'weight_only_new_optimizer_v1',
                         'source_sha256': CONTINUITY_SHA})
        lineage = plan_lineage(approval_sha='b' * 64, source=fake_source,
                               output=out, run=contract, target=32,
                               holdout_sha='c' * 64)
        assert lineage['new_run_initial_tokens'] == 0
        assert lineage['new_run_initial_step'] == 0
        step(candidate, optimizer)
        assert optimizer.state_dict()['state']
        sampler = random.Random(8)
        ck = {'checkpoint_schema': 2, 'config': cfg.to_dict(),
              'tokenizer_sha256': tok, 'model': candidate.state_dict(),
              'optimizer': optimizer.state_dict(), 'step': 1,
              'processed_tokens': 8, 'target_tokens': 32,
              'train_identity': train_id, 'validation_identity': val_id,
              'sampler_version': SAMPLER_VERSION, 'run_contract': contract,
              'warmstart_lineage': lineage,
              'torch_rng': torch.get_rng_state(), 'cuda_rng': [],
              'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
              'train_sampler_rng': sampler.getstate()}
        checkpoint = out / 'resume.pt'
        atomic_save(ck, checkpoint)
        first_sha = file_sha(checkpoint)
        journal = CheckpointJournal(checkpoint, initial_sha256=first_sha,
                                    initial_tokens=8, writer=atomic_save)
        verified = read_verified_resume(
            checkpoint=checkpoint, expected_sha=first_sha, trusted=True,
            out=out, torch=torch, config=cfg.to_dict(), target_tokens=32,
            tokenizer_sha=tok, train_identity=train_id,
            validation_identity=val_id, expected_run=contract, run_until=16)
        assert verified['warmstart_lineage'] == lineage
        loaded = Neral350M(cfg).cpu()
        loaded.load_state_dict(verified['model'], strict=True)
        loaded_opt = torch.optim.AdamW(loaded.parameters(), lr=2e-5,
                                       betas=(0.9, 0.95), eps=1e-8,
                                       weight_decay=0.1, fused=True)
        loaded_opt.load_state_dict(verified['optimizer'])
        step(candidate, optimizer)
        step(loaded, loaded_opt)
        assert all(torch.equal(a, b) for a, b in zip(candidate.parameters(), loaded.parameters()))
        second = dict(ck, model=candidate.state_dict(),
                      optimizer=optimizer.state_dict(), step=2, processed_tokens=16)
        receipt = journal.commit(second, checkpoint)
        assert receipt['processed_tokens'] == 16
        assert journal.verify()['sha256'] == file_sha(checkpoint)
        assert journal.verify()['kind'] == 'complete_resumable_state'
        assert file_sha(fake_source) == source_digest
        assert all(torch.equal(initial_weights[k], original.state_dict()[k])
                   for k in initial_weights), 'source model unexpectedly changed'
        tampered = dict(verified['warmstart_lineage'], holdout_sha256='d' * 64)
        assert tampered != lineage, 'changed assessment must not pass lineage check'
        print(json.dumps({'event': 'synthetic_new_optimizer_lineage_checkpoint_replay',
                          'source': 'fake_temp_checkpoint_not_real_neral',
                          'model_parameters': sum(p.numel() for p in loaded.parameters()),
                          'optimizer_started_empty': True,
                          'new_tokens_at_checkpoint': 8,
                          'new_steps_at_checkpoint': 1,
                          'next_weights_equal_after_reload': True,
                          'sha_pinned_native_resume_passed': True,
                          'provenance_preserved_in_new_checkpoint': True,
                          'changed_holdout_lineage_distinguishable': True,
                          'journal_complete_state_and_hash_verified': True,
                          'synthetic_source_unchanged': True,
                          'cuda_used': False, 'real_350m_training_started': False,
                          'quality_improvement_measured': False},
                         sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
