from types import SimpleNamespace
import os
import pytest
import frontier_manager as manager
from frontier_sidecar import FrontierUnavailable

@pytest.fixture(autouse=True)
def clean_manager_state():
    manager.stop_all()
    yield
    manager.stop_all()

class Proc:
    def __init__(self,alive=True):self.alive=alive;self.terminated=False;self.killed=False
    def poll(self):return None if self.alive else 0
    def terminate(self):self.terminated=True;self.alive=False
    def wait(self,timeout=None):return 0
    def kill(self):self.killed=True;self.alive=False

def test_stop_process_is_bounded_and_idempotent():
    p=Proc();manager._stop_process(p);assert p.terminated and not p.alive
    manager._stop_process(p);assert p.terminated

def test_foreign_occupied_port_is_refused(monkeypatch,tmp_path):
    runtime=tmp_path/"frontier_runtime.py";runtime.write_text("# fixture")
    row={"model_sha256":"a"*64,"runtime_port":9683,"runtime_family":"frontier350-bpe-v1",
         "paths":(tmp_path/"bundle/model.pt",tmp_path/"bundle/model.sha256"),"context_tokens":512,"label":"Arctus"}
    monkeypatch.setattr(manager,"probe",lambda *a,**k:(_ for _ in ()).throw(FrontierUnavailable("mismatch")))
    monkeypatch.setattr(manager,"_port_open",lambda port:True)
    with pytest.raises(FrontierUnavailable,match="anderen Prozess"):
        manager.ensure_running(row,tmp_path,runtime,timeout=0.01)

def test_own_old_sidecar_on_same_port_is_rotated(monkeypatch,tmp_path):
    bundle=tmp_path/"bundle";bundle.mkdir();(bundle/"model.pt").write_bytes(b"x")
    runtime=tmp_path/"frontier_runtime.py";runtime.write_text("# fixture")
    row={"model_sha256":"b"*64,"runtime_port":9683,"runtime_family":"frontier350-bpe-v1",
         "paths":(bundle/"model.pt",bundle/"model.sha256"),"context_tokens":512,"label":"Arctus"}
    old=Proc();manager._processes.clear();manager._processes[("a"*64,9683)]=old
    calls={"probe":0}
    health={"runtime_family":"frontier350-bpe-v1","checkpoint_sha256":"b"*64}
    def probe(*args,**kwargs):
        calls["probe"]+=1
        if calls["probe"]==1:raise FrontierUnavailable("not yet")
        return health
    new=Proc()
    monkeypatch.setattr(manager,"probe",probe)
    monkeypatch.setattr(manager,"_port_open",lambda port:False)
    monkeypatch.setattr(manager.subprocess,"Popen",lambda *a,**k:new)
    monkeypatch.setattr(manager,"_python",lambda:os.sys.executable)
    result=manager.ensure_running(row,tmp_path,runtime,timeout=0.2)
    assert result is health and old.terminated
    assert ("a"*64,9683) not in manager._processes
    assert manager._processes[("b"*64,9683)] is new
