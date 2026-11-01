"""Source witnesses for new NERAL data. Only trusted repository code is executed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from project_store import ProjectStore, ripple, render, version_live
from chat_math import tool_answer
from stream_runtime import safe_prefix
import storage_codec

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pilot_data', ROOT/'scripts/neral_qlora_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
DATA = ROOT/'datasets/neral_project_sft_v1.jsonl'
REVIEW = ROOT/'datasets/neral_project_sft_v1.review.json'


def test_new_corpus_is_complete_reviewed_and_separate():
    rows = pilot.read_examples(DATA)
    assert len(rows) == 16
    review = pilot.verify_review(DATA, REVIEW)
    for row in review['rows']:
        for evidence in row['evidence']:
            lines = (ROOT/evidence['path']).read_text().splitlines()
            excerpt = '\n'.join(lines[evidence['start']-1:evidence['end']])
            assert hashlib.sha256(excerpt.encode()).hexdigest() == evidence['excerpt_sha256']
    holdout = {json.loads(line)['prompt'].casefold() for line in
               (ROOT/'datasets/neral_qlora_holdout.jsonl').read_text().splitlines()}
    assert not holdout.intersection(row['prompt'][1]['content'].casefold() for row in rows)
    assert not any('Qwen' in row['completion'][0]['content'] for row in rows)


def make_project(tmp_path):
    store = ProjectStore(tmp_path/'projects.sqlite3')
    state = store.create('trusted corpus witness')
    state = store.add(state['id'], {'id': 'preis', 'kind': 'fact', 'title': 'Preis', 'value': '20'})
    return store, state


def test_preview_does_not_commit_and_apply_is_single_use(tmp_path):
    store, state = make_project(tmp_path)
    preview = store.preview(state['id'], 'preis', '25')
    assert preview['requires_confirmation'] and not preview['external_actions']
    assert store.get(state['id'])['nodes']['preis']['value'] == '20'
    assert store.apply(state['id'], preview['id'])['nodes']['preis']['value'] == '25'
    with pytest.raises(ValueError): store.apply(state['id'], preview['id'])


def test_stale_preview_rejects_and_undo_increments_revision(tmp_path):
    store, state = make_project(tmp_path)
    older = store.preview(state['id'], 'preis', '25')
    newer = store.preview(state['id'], 'preis', '30')
    changed = store.apply(state['id'], newer['id'])
    with pytest.raises(ValueError): store.apply(state['id'], older['id'])
    undone = store.undo(state['id'], changed['last_transaction'])
    assert undone['revision'] == changed['revision']+1
    assert undone['nodes']['preis']['value'] == '20'
    assert undone['last_transaction'] is None


def test_old_undo_cannot_overwrite_later_changes(tmp_path):
    store, state = make_project(tmp_path)
    preview = store.preview(state['id'], 'preis', '25')
    changed = store.apply(state['id'], preview['id'])
    store.refresh(state['id'])
    with pytest.raises(ValueError): store.undo(state['id'], changed['last_transaction'])


def test_transitive_paths_and_cycle_failure():
    nodes = {k: {'title': k, 'depends_on': deps} for k, deps in [('a', []), ('b', ['a']), ('c', ['b'])]}
    assert [(r['id'], r['path']) for r in ripple(nodes, 'a')] == [('b', ['a', 'b']), ('c', ['a', 'b', 'c'])]
    nodes['a']['depends_on'] = ['c']
    with pytest.raises(ValueError): render(nodes)


def test_formula_dependencies_and_live_restrictions(tmp_path):
    store, state = make_project(tmp_path)
    state = store.add(state['id'], {'id':'menge','kind':'fact','title':'Menge','value':'3'})
    state = store.add(state['id'], {'id':'summe','kind':'formula','title':'Summe','value':'menge * preis'})
    assert state['nodes']['summe']['depends_on'] == ['menge','preis']
    assert state['nodes']['summe']['result'] == '60'
    with pytest.raises(ValueError):
        store.add(state['id'], {'id':'falsch','kind':'fact','title':'Falsch','value':'1','live':True})


def test_unchanged_live_text_does_not_duplicate_version():
    node = {'live': True, 'result': 'same', 'versions': [{'version': 1, 'text': 'same'}]}
    state = {'nodes': {'answer': node}}
    version_live(state, {}, 'source witness')
    assert len(node['versions']) == 1 and node['last_checked']


def test_calculator_and_streaming_claims():
    assert tool_answer('0.1 + 0.2').text == '0.3'
    assert tool_answer('Berechne 7 + 2 und dann addiere 4') is None
    assert safe_prefix('kurzer Zwischenstand') == ''


def test_nonwindows_storage_is_not_claimed_encrypted():
    if storage_codec.os.name == 'nt': pytest.skip('Linux storage witness; Windows needs actual DPAPI acceptance')
    assert storage_codec.dumps({'local': True}) == '{"local": true}'
    with pytest.raises(ValueError): storage_codec.loads('dpapi1:AAAA')
