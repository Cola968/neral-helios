"""Computer Use is opt-in and every desktop mutation requires a fresh approval.

Mocks prove behavior without moving a real pointer or taking real screenshots.
"""
import base64
import time
import pytest

from computer_use import ComputerUse, ComputerUseError


class FakeDesktop:
    def __init__(self):
        self.actions = []
    def screenshot(self):
        return {"image_base64":base64.b64encode(b"fake-png").decode(),
                "media_type":"image/png","width":800,"height":600}
    def execute(self, action):
        self.actions.append(action)


@pytest.fixture
def desktop():
    backend=FakeDesktop()
    return ComputerUse(backend, clock=lambda:1000), backend


def test_disabled_by_default_never_reads_screen(desktop):
    control, backend=desktop
    with pytest.raises(ComputerUseError):
        control.screenshot()
    assert backend.actions == []


def test_explicit_start_and_stop_required_for_every_action(desktop):
    control, backend=desktop
    assert control.status()["enabled"] is False
    control.start()
    image=control.screenshot()
    assert image["width"] == 800 and image["height"] == 600
    req=control.preview({"action":"click","x":200,"y":250})
    assert backend.actions == []
    with pytest.raises(ComputerUseError):
        control.execute("fake-approval")
    executed=control.execute(req["approval_token"])
    assert executed["executed"] is True
    assert backend.actions == [{"action":"click","x":200,"y":250}]
    with pytest.raises(ComputerUseError):
        control.execute(req["approval_token"])
    control.stop()
    with pytest.raises(ComputerUseError):
        control.screenshot()


@pytest.mark.parametrize("bad", [
 {"action":"click","x":-1,"y":5},
 {"action":"click","x":800,"y":5},
 {"action":"click","x":5,"y":600},
 {"action":"move","x":1e30,"y":20},
 {"action":"scroll","amount":4000},
 {"action":"hotkey","keys":["win","r"]},
 {"action":"type","text":"\ncmd.exe"},
 {"action":"type","text":"x"*260},
 {"action":"press","key":"win"},
 {"action":"shell","command":"calc.exe"},
 {"action":"click","x":True,"y":12},
])
def test_unsafe_actions_rejected(desktop,bad):
    ctrl,backend=desktop
    ctrl.start()
    with pytest.raises(ComputerUseError):
        ctrl.preview(bad)
    assert not backend.actions


def test_approval_expiry_and_one_pending_action():
    now=[1000]
    backend=FakeDesktop()
    ctrl=ComputerUse(backend,clock=lambda:now[0])
    ctrl.start()
    token=ctrl.preview({"action":"type","text":"safe text"})["approval_token"]
    ctrl.preview({"action":"press","key":"Escape"})
    with pytest.raises(ComputerUseError):
        ctrl.execute(token)
    pending=ctrl.preview({"action":"click","x":1,"y":2})["approval_token"]
    now[0]+=25
    with pytest.raises(ComputerUseError):
        ctrl.execute(pending)
    assert backend.actions==[]


def test_type_actions_need_explicit_approval_and_are_not_logged(desktop):
    ctrl,backend=desktop
    ctrl.start()
    payload={"action":"type","text":"private typing"}
    preview=ctrl.preview(payload)
    assert "private typing" not in str(preview)
    result=ctrl.execute(preview["approval_token"])
    assert result=={"executed":True,"action":"type"}
    assert backend.actions == [payload]


def test_emergency_stop_invalidates_pending_approval(desktop):
    ctrl,backend=desktop
    ctrl.start()
    token=ctrl.preview({"action":"press","key":"Escape"})["approval_token"]
    ctrl.stop()
    with pytest.raises(ComputerUseError):
        ctrl.execute(token)
    assert backend.actions==[]


def test_screenshot_size_limit_fails_closed():
    class Oversize(FakeDesktop):
        def screenshot(self):
            return {"image_base64":"A"*(2_000_000),"media_type":"image/png","width":800,"height":600}
    ctrl=ComputerUse(Oversize(),clock=lambda:1000)
    ctrl.start()
    with pytest.raises(ComputerUseError):
        ctrl.screenshot()


def test_rate_limit_does_not_permit_unbounded_calls(desktop):
    ctrl,backend=desktop
    ctrl.start()
    for i in range(ctrl.MAX_ACTIONS):
        token=ctrl.preview({"action":"move","x":i,"y":0})["approval_token"]
        ctrl.execute(token)
    with pytest.raises(ComputerUseError):
        ctrl.preview({"action":"click","x":1,"y":1})
