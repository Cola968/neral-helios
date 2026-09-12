"""Bounded V5 source-mix ablation from admitted local recovery TRAIN tokens.

No network, tokenization, new source, held-out validation or synthetic EOS. The
same source prefixes are shared across variants so only source quotas differ.
These tokens are new to the immutable 1B weights, but were used by rejected LR
pilots: this remains a development experiment, never a fresh release holdout.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from build_frontier_bpe_data import StreamingShardWriter, sha256
from build_weakness_recovery_pilot_bpe_data import PROFILES, START_TOKENS
from security_policy350_bpe import validate_manifest


def source_ranges(manifest):
    cursor = 0
    result = {}
    for report in manifest['sources']:
        if report['partition'] != 'train':
            continue
        name = report['source']['name']
        count = int(report['emitted_tokens'])
        if name in result or count <= 0:
            raise ValueError('invalid source partition')
        result[name] = (cursor, count, report)
        cursor += count
    if cursor != manifest['train_tokens']:
        raise ValueError('source accounting differs from train shards')
    return result


def slice_stream(arrays, start, count):
    for array in arrays:
        if start >= len(array):
            start -= len(array)
            continue
        take = min(count, len(array) - start)
        yield array[start:start + take]
        count -= take
        start = 0
        if count == 0:
            return
    raise ValueError('source range exceeds shard stream')


def remix(seed, root, tokenizer, profile):
    seed, root, tokenizer = Path(seed), Path(root), Path(tokenizer)
    validate_manifest(seed/'manifest.json', seed, tokenizer)
    manifest = json.loads((seed/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('continuation_profile') != 'arctus-2b-balanced-pilot-v1' or manifest['train_tokens'] != 20_000_000:
        raise ValueError('requires admitted 20M recovery seed')
    if root.exists() and any(root.iterdir()):
        raise ValueError('output exists; never silently mix corpora')
    ranges = source_ranges(manifest)
    arrays = [np.memmap(seed/'train'/p['path'], dtype='<u2', mode='r') for p in manifest['train_files']]
    writer = StreamingShardWriter(root/'train', 'train', 5_000_000)
    reports = []
    for name, share in PROFILES[profile].items():
        start, available, original = ranges[name]
        count = round(10_000_000 * share)
        if count > available:
            raise ValueError('quota exceeds admitted source train partition')
        for chunk in slice_stream(arrays, start, count):
            writer.add(chunk)
        reports.append(dict(original, source=dict(original['source'], share=share), emitted_tokens=count,
                            parent_train_offset=start, parent_train_tokens=available,
                            accounting='token-prefix remix; row counts belong to parent source'))
    files = writer.finish()
    valid_files = []
    (root/'valid').mkdir(parents=True)
    # The exact existing validation is copied, never read into the training writer.
    import shutil
    for item in manifest['valid_files']:
        shutil.copyfile(seed/'valid'/item['path'], root/'valid'/item['path'])
        valid_files.append(item)
    result = dict(manifest, train_files=files, valid_files=valid_files, sources=[
        r for r in manifest['sources'] if r['partition']=='valid'] + reports,
        train_tokens=10_000_000, target_tokens=10_200_000,
        continuation_profile='arctus-v5-local-remix-'+profile,
        target_cumulative_tokens=START_TOKENS+10_000_000, focus_shares=PROFILES[profile],
        parent_manifest_sha256=sha256(seed/'manifest.json'),
        synthetic_eos_added=0, prior_rejected_pilot_data=True,
        development_only=True)
    for key in ('created_unix', 'new_deduplicated_documents', 'deduplicated_documents'):
        result.pop(key, None)
    (root/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    return validate_manifest(root/'manifest.json', root, tokenizer)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    for name in ('seed','root','tokenizer'):ap.add_argument('--'+name, required=True)
    ap.add_argument('--profile', choices=PROFILES, required=True)
    args=ap.parse_args()
    print(json.dumps(remix(args.seed,args.root,args.tokenizer,args.profile)))
