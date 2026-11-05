"""Staged NERAL adapter loader. Not activated by Helios/app.py.

No downloads or active-version pointer. Caller must explicitly select a sealed
version; rollback means a fresh base load, never reuse a partially mutated model.
"""
import gc
import hashlib
import json
from pathlib import Path
import re

TARGETS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_adapter(folder, base):
    from model_catalog import _json, HF_BASE
    folder = Path(folder)
    if folder.is_symlink() or folder.resolve() != folder.absolute() or not folder.is_dir():
        raise ValueError('Adapter must be a flat regular directory.')
    allowed = {'adapter_config.json', 'adapter_model.safetensors', 'neral-adapter.json',
               'neral-training-manifest.json', 'training-report.json', 'README.md'}
    entries = list(folder.iterdir())
    if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in entries):
        raise ValueError('Unexpected adapter file, link or directory.')
    manifest = _json(folder / 'neral-adapter.json')
    if not isinstance(manifest, dict) or set(manifest) != {
        'schema', 'product', 'version', 'base_model', 'base_revision', 'base_snapshot_integrity',
        'dataset_sha256', 'files', 'release_validated'}:
        raise ValueError('Invalid adapter manifest.')
    if (manifest['schema'] != 1 or manifest['product'] != 'NERAL'
            or not isinstance(manifest['version'], str)
            or not re.fullmatch(r'\d+\.\d+\.\d+(?:-pilot\.\d+)?', manifest['version'])
            or manifest['base_model'] != HF_BASE
            or manifest['base_revision'] != base['revision']
            or manifest['base_snapshot_integrity'] != base['sha256']
            or not isinstance(manifest['release_validated'], bool)
            or not re.fullmatch('[a-f0-9]{64}', str(manifest['dataset_sha256']))):
        raise ValueError('Wrong adapter version, base or provenance.')
    files = manifest['files']
    if not isinstance(files, dict) or set(files) != {'adapter_config.json', 'adapter_model.safetensors'}:
        raise ValueError('Invalid adapter inventory.')
    for name, digest in files.items():
        path = folder / name
        if not isinstance(digest, str) or not re.fullmatch('[a-f0-9]{64}', digest) or path.stat().st_size > 512*1024**2:
            raise ValueError('Invalid adapter size or digest.')
        if sha256(path) != digest:
            raise ValueError('Adapter integrity failed.')
    config = _json(folder / 'adapter_config.json')
    # Only the initial standard LoRA recipe is supported. No DoRA, prompt tuning,
    # embedding resize, arbitrary extra trainable modules or auto-mapped code.
    if (config.get('peft_type') != 'LORA' or config.get('task_type') != 'CAUSAL_LM'
            or config.get('r') != 16 or config.get('lora_alpha') != 16
            or config.get('lora_dropout') != 0 or config.get('bias') != 'none'
            or config.get('init_lora_weights', True) is not True
            or not isinstance(config.get('target_modules'), list)
            or set(config['target_modules']) != TARGETS
            or any(config.get(k) for k in ('auto_mapping', 'modules_to_save', 'use_dora', 'use_rslora',
                'rank_pattern', 'alpha_pattern', 'layers_to_transform', 'layers_pattern',
                'layer_replication', 'trainable_token_indices', 'target_parameters', 'megatron_config',
                'exclude_modules', 'loftq_config', 'eva_config', 'corda_config', 'fan_in_fan_out',
                'use_qalora', 'lora_bias'))):
        raise ValueError('Unsupported adapter configuration.')
    return manifest


def seal_adapter(folder, base, dataset_digest, version):
    """Seal new training output; never overwrite an existing version manifest."""
    from model_catalog import HF_BASE
    folder = Path(folder)
    manifest = {'schema': 1, 'product': 'NERAL', 'version': version,
        'base_model': HF_BASE, 'base_revision': base['revision'],
        'base_snapshot_integrity': base['sha256'], 'dataset_sha256': dataset_digest,
        'release_validated': False,
        'files': {name: sha256(folder / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}}
    with (folder / 'neral-adapter.json').open('x', encoding='utf-8') as target:
        json.dump(manifest, target, indent=2)
    validate_adapter(folder, base)
    return manifest


def load_candidate(base_folder, adapter_folder, *, allow_unvalidated=False):
    """Explicit experimental entrypoint; failed load returns a fresh base + reason.

    Helios integration is gated on approved GPU/evaluation evidence. This function
    is deliberately not called by the product. A base-only selection can continue
    to call qwen_local_runtime.load without importing PEFT.
    """
    import qwen_local_runtime as runtime
    base = runtime.validate_hf_snapshot(Path(base_folder), verify_hashes=True)
    checked = validate_adapter(Path(adapter_folder), base)
    if not checked['release_validated'] and not allow_unvalidated:
        raise ValueError('Pilot adapter has no release validation; experimental opt-in required.')
    candidate = runtime.load(base_folder)
    try:
        from peft import PeftModel
        candidate.model = PeftModel.from_pretrained(candidate.model, str(adapter_folder),
            is_trainable=False, local_files_only=True, autocast_adapter_dtype=True)
        candidate.model.eval()
        candidate.model.requires_grad_(False)
        digest = hashlib.sha256(json.dumps(checked, sort_keys=True).encode()).hexdigest()
        candidate.identity = f'{candidate.identity}:neral-lora:{checked["version"]}:{digest}'
        candidate.adapter_status = {'active': True, 'version': checked['version'], 'sha256': digest}
        return candidate
    except Exception as exc:
        reason = type(exc).__name__  # Do not expose local paths/secrets in UI.
        candidate.close()
        candidate = None
        gc.collect()
    fallback = runtime.load(base_folder)
    fallback.adapter_status = {'active': False, 'fallback': 'fresh-base', 'error_type': reason}
    return fallback
