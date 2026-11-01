import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('blind', ROOT / 'scripts/neral_blind_comparison.py')
blind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blind)


@pytest.fixture
def inputs(tmp_path):
    cases = [json.loads(line) for line in blind.HOLDOUT.read_text().splitlines()]
    files = []
    for label in ('base', 'adapter'):
        path = tmp_path / (label+'.jsonl')
        path.write_text(''.join(json.dumps({'id': c['id'], 'text': label+' answer'})+'\n' for c in cases))
        files.append(path)
    meta = {'base_model': 'Qwen/Qwen2.5-Coder-7B-Instruct', 'base_revision': 'a'*40,
        'base_snapshot_integrity': 'b'*64, 'holdout_sha256': hashlib.sha256(blind.HOLDOUT.read_bytes()).hexdigest(),
        'system_prompt': 'NERAL based on Qwen', 'chat_template_sha256': 'c'*64,
        'max_new_tokens': 256, 'do_sample': False, 'quantization': 'nf4/double/bf16',
        'packages': {'torch': '2.8.0+cu128'}, 'adapter_sha256': None}
    for label in ('base', 'adapter'):
        path = tmp_path / (label+'-meta.json')
        path.write_text(json.dumps({**meta, 'adapter_sha256': None if label == 'base' else 'd'*64}))
        files.append(path)
    return files


def test_export_is_complete_and_key_separate(inputs, tmp_path):
    out = tmp_path / 'review'
    blind.prepare(*inputs, out)
    rows = [json.loads(line) for line in (out/'reviewer.jsonl').read_text().splitlines()]
    key = json.loads((out/'private-key.json').read_text())
    assert len(rows) == 8
    for row in rows:
        assert row['A'] == key['mapping'][row['id']]['A']+' answer'
        assert row['B'] == key['mapping'][row['id']]['B']+' answer'
    with pytest.raises(FileExistsError):
        blind.prepare(*inputs, out)


@pytest.mark.parametrize('mutation', ['setting', 'missing', 'duplicate', 'holdout'])
def test_invalid_comparisons_fail_before_output(inputs, tmp_path, mutation):
    if mutation in ('setting', 'holdout'):
        meta = json.loads(inputs[3].read_text())
        meta['max_new_tokens' if mutation == 'setting' else 'holdout_sha256'] = 99
        inputs[3].write_text(json.dumps(meta))
    else:
        rows = inputs[1].read_text().splitlines()
        inputs[1].write_text('\n'.join(rows[:-1] if mutation == 'missing' else rows+[rows[0]]))
    out = tmp_path / 'review'
    with pytest.raises(ValueError):
        blind.prepare(*inputs, out)
    assert not out.exists()


def test_holdout_has_no_exact_training_prompt_overlap():
    training = {json.loads(line)['messages'][1]['content'].strip().casefold()
                for line in (ROOT/'datasets/neral_qlora_pilot.jsonl').read_text().splitlines()}
    cases = [json.loads(line) for line in blind.HOLDOUT.read_text().splitlines()]
    assert len({case['id'] for case in cases}) == len(cases)
    assert not training.intersection(case['prompt'].strip().casefold() for case in cases)
