"""Build a reviewed 86-example NERAL QLoRA dataset without training or models.

Keeps all three existing JSONL sources unchanged. Explicitly validates the
reviewed phase-2.2 corpus and each original per-row review before deriving a
new training-file + review manifest understood by neral_qlora_pilot.py.
Does not authorize training, GPU use, downloads, merges, or model activation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

PACK_PARTS = (
    ('neral_qlora_pilot', 10),
    ('neral_project_sft_v1', 16),
    ('neral_capabilities_sft_v1', 60),
)
SHA = re.compile(r'[0-9a-f]{64}\Z')
MAX_DATA = 1024 * 1024
MAX_REVIEW = 1024 * 1024


class PackError(ValueError):
    """Unreviewed source, changed corpus or unsafe output destination."""


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _checked_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_REVIEW:
        raise PackError('review missing, linked, or oversized')
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise PackError('duplicate JSON key in review')
            obj[k] = v
        return obj
    try:
        review = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)
    except (UnicodeError, ValueError) as exc:
        raise PackError('invalid review JSON') from exc
    if not isinstance(review, dict):
        raise PackError('review must be an object')
    return review


def combine_reviewed(root: Path, parts: list[tuple[Path, Path]]):
    """Join reviewed lines and remap row numbers. No I/O writes or imports.

    All original dataset SHA-256, row SHA-256 and source SHA-256 values remain
    required. `corpus.validate` and `pilot.verify_review` are enforced by the
    public pack_training_data entrypoint before this composition is published.
    """
    root = Path(root).resolve()
    if len(parts) != len(PACK_PARTS):
        raise PackError('wrong dataset inventory')
    result_lines, result_rows, provenance, sources = [], [], [], {}
    for (expected_name, expected_count), (dataset, review_path) in zip(PACK_PARTS, parts):
        dataset, review_path = Path(dataset), Path(review_path)
        if dataset.resolve() != (root / 'datasets' / f'{expected_name}.jsonl').resolve() or \
           review_path.resolve() != (root / 'datasets' / f'{expected_name}.review.json').resolve():
            raise PackError('unexpected dataset/review location')
        if (not dataset.is_file() or dataset.is_symlink() or
                dataset.stat().st_size > MAX_DATA):
            raise PackError('missing, linked or oversized dataset')
        try:
            raw = dataset.read_bytes()
            lines = raw.decode('utf-8').splitlines()
        except UnicodeError as exc:
            raise PackError('invalid dataset encoding') from exc
        if len(lines) != expected_count or not raw.endswith(b'\n'):
            raise PackError('wrong row count or noncanonical line ending')
        review = _checked_json(review_path)
        if review.get('schema') != 1 or review.get('dataset_sha256') != digest(raw):
            raise PackError('source dataset digest differs from reviewed original')
        original_rows = review.get('rows')
        reviewed_sources = review.get('sources')
        if (not isinstance(original_rows, list) or len(original_rows) != expected_count
                or not isinstance(reviewed_sources, dict) or not reviewed_sources):
            raise PackError('incomplete review inventory')
        for relative, source_digest in reviewed_sources.items():
            if (not isinstance(relative, str) or not isinstance(source_digest, str) or
                    not SHA.fullmatch(source_digest) or Path(relative).is_absolute() or
                    any(p in ('', '.', '..') for p in Path(relative).parts)):
                raise PackError('unsafe reviewed source')
            if relative in sources and sources[relative] != source_digest:
                raise PackError('source conflict across review manifests')
            sources[relative] = source_digest
        if {row.get('line') for row in original_rows if isinstance(row, dict)} != set(range(1, expected_count + 1)):
            raise PackError('review line coverage incomplete or duplicated')
        by_line = {row['line']: row for row in original_rows}
        for source_line, line in enumerate(lines, 1):
            original = by_line[source_line]
            if original.get('row_sha256') != digest(line.encode('utf-8')):
                raise PackError('review row digest mismatch')
            if not original.get('evidence') or not original.get('finding'):
                raise PackError('missing review evidence or finding')
            row = copy.deepcopy(original)
            row['line'] = len(result_rows) + 1
            row['source_dataset'] = expected_name + '.jsonl'
            row['source_line'] = source_line
            result_rows.append(row)
            result_lines.append(line)
        provenance.append({'dataset': dataset.name, 'count': expected_count,
                           'sha256': digest(raw), 'review_sha256': digest(review_path.read_bytes())})
    for path, expected in sources.items():
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file() or digest(target.read_bytes()) != expected:
            raise PackError('reviewed source digest changed or source missing')
    combined = ('\n'.join(result_lines) + '\n').encode('utf-8')
    manifest = {'schema': 1, 'dataset_sha256': digest(combined), 'sources': sources, 'rows': result_rows,
                'derivation': 'joined_reviewed_existing_rows_no_new_semantic_review'}
    if len(combined) > MAX_DATA or len(json.dumps(manifest, ensure_ascii=False).encode('utf-8')) > MAX_REVIEW:
        raise PackError('combined corpus exceeds original pilot safety limits')
    return combined, manifest, provenance


def _verify_existing_corpus(root: Path):
    from neral_control import corpus
    return corpus.validate(root)


def pack_training_data(root: Path, out_dir: Path) -> dict:
    root = Path(root).resolve()
    out_dir = Path(out_dir).absolute()
    destination = out_dir.resolve()
    if destination == root or root in destination.parents:
        raise PackError('output must be outside the repository')
    if destination.exists() or out_dir.is_symlink():
        raise PackError('output directory must not exist')
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise PackError('output parent must be an existing regular directory')
    verified = _verify_existing_corpus(root)
    if (verified.get('total_sft_examples') != 86 or verified.get('new_sft_examples') != 60
            or verified.get('leakage', {}).get('flagged_pairs')
            or verified.get('independent_review_completed') is True):
        raise PackError('corpus quality gate is not the frozen 86-example configuration')
    parts = [(root / 'datasets' / (name + '.jsonl'),
              root / 'datasets' / (name + '.review.json')) for name, _ in PACK_PARTS]
    from scripts import neral_qlora_pilot as pilot
    for dataset, review in parts:
        pilot.verify_review(dataset, review)
    combined, manifest, provenance = combine_reviewed(root, parts)
    summary = {'status': 'REVIEWED_BUNDLE_ONLY', 'example_count': 86,
               'dataset_sha256': manifest['dataset_sha256'], 'sources': provenance,
               'independent_semantic_review': False, 'training_executed': False,
               'adapter_created': False, 'holdout_untouched': True}
    with TemporaryDirectory(prefix='.neral-86-pack-', dir=destination.parent) as tmp:
        tmpdir = Path(tmp)
        data = tmpdir / 'neral_train_86.jsonl'
        review_file = tmpdir / 'neral_train_86.review.json'
        data.write_bytes(combined)
        review_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # This is the exact downstream trainer's manifest verification contract.
        pilot.verify_review(data, review_file)
        (tmpdir / 'neral_train_86.provenance.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if destination.exists():
            raise PackError('output appeared concurrently; refusing overwrite')
        tmpdir.rename(destination)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--out-dir', type=Path, required=True, help='NEW folder outside repository')
    args = p.parse_args(argv)
    try:
        report = pack_training_data(args.root, args.out_dir)
    except (PackError, ValueError, OSError, ImportError) as exc:
        p.exit(2, f'Reviewed training bundle refused: {type(exc).__name__}: {exc}\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
