"""NERAL Mini: real HTTP cancellation is session-scoped, not just a UI abort."""
from __future__ import annotations

import hashlib
import http.client
import http.cookiejar
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlencode
import urllib.error
import urllib.request

import pytest

from mini.neral_mini import MiniCancelled, MiniServer
from mini.mobile_gateway import create_mobile_gateway


class SlowLocalBackend:
    def __init__(self):
        self.started=threading.Event()
        self.release=threading.Event()
        self.cancel_event=None
        self.calls=0

    def set_cancel_event(self,event):
        self.cancel_event=event

    def status(self):
        return {"name":"NERAL Mini","foundation":"Qwen2.5-Coder-7B-Instruct",
                "adapter_loaded":True,"adapter_configured":True,
                "local_only":True,"experimental":True}

    def respond(self,messages):
        assert self.cancel_event is not None
        self.calls+=1
        if self.calls==1:
            self.started.set()
            self.release.wait(timeout=2)
            if self.cancel_event.is_set():
                raise MiniCancelled("Anfrage abgebrochen.")
        return "Lokale Modellantwort"


@pytest.fixture
def desktop():
    model=SlowLocalBackend()
    server=MiniServer(("127.0.0.1",0),model,'<html>token=%%NERAL_SESSION%%</html>')
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    url=f"http://127.0.0.1:{server.server_address[1]}"
    jar=http.cookiejar.CookieJar()
    client=urllib.request.build_opener(
        urllib.request.ProxyHandler({}),urllib.request.HTTPCookieProcessor(jar))
    response=client.open(url+"/",timeout=3)
    match=re.search(r"token=([a-zA-Z0-9_-]+)",response.read().decode())
    assert match
    try:
        yield server,model,url,client,match.group(1)
    finally:
        model.release.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=4)


def desktop_call(desktop,path,body=b"",origin=True,token=None):
    server,model,url,client,session=desktop
    headers={"X-Neral-Session":session if token is None else token}
    if origin:
        headers["Origin"]=url
    if path=="/api/chat":
        body=json.dumps({"prompt":"Hallo"}).encode()
        headers["Content-Type"]="application/json"
    req=urllib.request.Request(url+path,data=body,headers=headers,method="POST")
    try:
        res=client.open(req,timeout=4)
        return res.status,json.loads(res.read())
    except urllib.error.HTTPError as exc:
        return exc.code,json.loads(exc.read())


def test_desktop_cancel_is_authenticated_and_idle_returns_conflict(desktop):
    assert desktop_call(desktop,"/api/cancel")[0]==409
    assert desktop_call(desktop,"/api/cancel",origin=False)[0]==403
    assert desktop_call(desktop,"/api/cancel",token="wrong")[0]==403
    assert not desktop[1].started.is_set()


def test_desktop_cancel_stops_running_model_and_allows_followup(desktop):
    response=[]
    worker=threading.Thread(target=lambda:response.append(desktop_call(desktop,"/api/chat")),daemon=True)
    worker.start()
    assert desktop[1].started.wait(2)
    code,body=desktop_call(desktop,"/api/cancel")
    assert code==202
    assert body["cancel_requested"] is True
    desktop[1].release.set()
    worker.join(timeout=3)
    assert response and response[0][0]==409
    assert response[0][1]["cancelled"] is True
    assert desktop_call(desktop,"/api/cancel")[0]==409
    assert desktop_call(desktop,"/api/chat")[0]==200
    assert desktop[1].calls==2


@pytest.fixture
def mobile(desktop):
    server,model,url,client,master=desktop
    config={"host":"192.168.18.7","port":9444,
      "pair_hash":hashlib.sha256(b"a1b2c3d4e5f60718").hexdigest()}
    gateway=create_mobile_gateway(config,server,bind_host="127.0.0.1",bind_port=0)
    worker=threading.Thread(target=gateway.serve_forever,daemon=True)
    worker.start()
    try:
        yield gateway,model
    finally:
        model.release.set()
        gateway.shutdown()
        gateway.server_close()
        worker.join(timeout=3)


def mobile_call(mobile,path,method="GET",data=None,session=None,origin=True):
    gateway,model=mobile
    conn=http.client.HTTPConnection("127.0.0.1",gateway.server_address[1],timeout=4)
    headers={"Host":"192.168.18.7:9444"}
    if origin:headers["Origin"]="https://192.168.18.7:9444"
    if session is not None:
        headers["Cookie"]="neral_mobile="+session
        headers["X-Neral-Session"]=session
    if isinstance(data,dict):
        data=json.dumps(data).encode()
        headers["Content-Type"]="application/json"
    elif path=="/pair":
        headers["Content-Type"]="application/x-www-form-urlencoded"
        data=urlencode({"code":data}).encode()
    conn.request(method,path,body=data,headers=headers)
    response=conn.getresponse()
    body=response.read().decode()
    result=response.status,dict(response.getheaders()),body
    conn.close()
    return result


def mobile_pair(mobile):
    status,headers,body=mobile_call(mobile,"/pair","POST","a1b2c3d4e5f60718")
    assert status==200
    return headers["Set-Cookie"].split(";",1)[0].split("=",1)[1]


def test_mobile_cancel_rejects_unauthorized_and_no_active_job(mobile):
    token=mobile_pair(mobile)
    assert mobile_call(mobile,"/api/cancel","POST",b"",session=None)[0]==401
    assert mobile_call(mobile,"/api/cancel","POST",b"",session=token,origin=False)[0]==403
    assert mobile_call(mobile,"/api/cancel","POST",b"",session=token)[0]==409
    assert not mobile[1].started.is_set()


def test_mobile_cancel_stops_same_shared_model_as_desktop(mobile):
    token=mobile_pair(mobile)
    response=[]
    worker=threading.Thread(target=lambda:response.append(
       mobile_call(mobile,"/api/chat","POST",{"prompt":"lange Aufgabe"},session=token)),daemon=True)
    worker.start()
    assert mobile[1].started.wait(2)
    code,headers,body=mobile_call(mobile,"/api/cancel","POST",b"",session=token)
    assert code==202
    assert json.loads(body)["cancel_requested"] is True
    mobile[1].release.set()
    worker.join(timeout=3)
    assert response and response[0][0]==409
    assert json.loads(response[0][2])["cancelled"] is True
    assert mobile_call(mobile,"/api/chat","POST",{"prompt":"Weiter"},session=token)[0]==200


def test_browser_shows_real_cancel_button_and_calls_protected_endpoint():
    html=(Path(__file__).resolve().parents[1]/"mini"/"index.html").read_text(encoding="utf-8")
    assert 'id="cancel"' in html
    assert '"/api/cancel"' in html
    assert 'credentials:"same-origin"' in html
    assert 'X-Neral-Session' in html
    assert "cancel.hidden = false" in html


def test_hf_stopping_criteria_is_driven_by_shared_cancel_event():
    from mini.neral_mini import NeralModel
    model=NeralModel(Path("fake"),Path("fake"))
    event=threading.Event()
    model.set_cancel_event(event)
    check=model._cancel_requested
    assert check() is False
    event.set()
    assert check() is True
