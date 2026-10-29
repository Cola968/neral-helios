import json
import launcher


class FakeResponse:
    def __init__(self,data):self.data=data
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,*args):return json.dumps(self.data).encode()


class FakeOpener:
    def __init__(self,data):self.data=data
    def open(self,url,timeout):return FakeResponse(self.data if url.endswith('/api/info') else {})


def test_launcher_accepts_arctus_and_configured_qwen(monkeypatch):
    data={'name':'Neral / Helios','parameters':349825152,'arctus350_enabled':True,
          'assistant':{'enabled':True,'base_model':'qwen3.5:9b'}}
    monkeypatch.setattr(launcher.urllib.request,'build_opener',lambda *a:FakeOpener(data))
    monkeypatch.setenv('NERAL_ASSISTANT_ENABLED','1')
    monkeypatch.setenv('NERAL_ASSISTANT_MODEL','qwen3.5:9b')
    monkeypatch.setenv('HELIOS_ENABLE_ARCTUS350','1')
    monkeypatch.delenv('HELIOS_CHECKPOINT',raising=False)
    assert launcher.status()
    data['assistant']['enabled']=False
    assert not launcher.status()


def test_launcher_will_not_spawn_again_on_occupied_incompatible_port(monkeypatch):
    import pytest
    monkeypatch.setattr(launcher,'status',lambda:False)
    monkeypatch.setattr(launcher,'port_occupied',lambda:True)
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda *a,**k:pytest.fail('must not start second server'))
    with pytest.raises(RuntimeError,match='Port 9680 ist belegt'):launcher.main()
