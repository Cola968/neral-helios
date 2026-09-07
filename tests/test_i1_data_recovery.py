import json
import sys
import types
from pathlib import Path

import pytest

from bootstrap_b2_sources import FINEWEB2_REV, SOURCE_SPECS, stream_public_source
from bootstrap_b3_sources import SPECS
from run_50m_i1 import junction


def test_fineweb2_pin_shared_by_b2_and_i1():
    assert FINEWEB2_REV == '66d1a3b989bad0c42b88179a7fb60c47ce87af74'
    for specs in (SOURCE_SPECS, SPECS):
        spec = next(s for s in specs if s['id'] == 'fineweb2-de-general')
        assert spec['revision'] == FINEWEB2_REV
        assert spec['provenance_url'].endswith(FINEWEB2_REV)


def test_verified_download_reused_and_tampering_rebuilt(tmp_path, monkeypatch):
    calls=[]
    def loader(*args, **kwargs):
        calls.append(kwargs)
        return [{'id':'one','text':'a'*500}]
    monkeypatch.setitem(sys.modules,'datasets',types.SimpleNamespace(load_dataset=loader))
    spec=SOURCE_SPECS[0];path=tmp_path/'source.jsonl'
    first=stream_public_source(spec,path,400)
    assert stream_public_source(spec,path,400)==first
    assert len(calls)==1
    path.write_text('corrupted',encoding='utf8')
    stream_public_source(spec,path,400)
    assert len(calls)==2


def test_failed_stream_preserves_completed_file(tmp_path, monkeypatch):
    path=tmp_path/'source.jsonl';path.write_text('keep',encoding='utf8')
    def broken():
        yield {'id':'one','text':'a'*500}
        raise OSError('interrupted transfer')
    monkeypatch.setitem(sys.modules,'datasets',types.SimpleNamespace(load_dataset=lambda *a,**k:broken()))
    with pytest.raises(OSError):stream_public_source(SOURCE_SPECS[0],path,1000)
    assert path.read_text(encoding='utf8')=='keep'
    assert not path.with_suffix('.receipt.json').exists()


def test_same_directory_junction_is_noop(tmp_path):
    marker=tmp_path/'weights.pt';marker.write_bytes(b'own')
    junction(tmp_path,tmp_path)
    assert marker.read_bytes()==b'own'


def test_junction_never_replaces_existing_directory(tmp_path):
    link=tmp_path/'link';target=tmp_path/'target';link.mkdir();target.mkdir()
    with pytest.raises(FileExistsError):junction(link,target)
    assert link.is_dir() and target.is_dir()
