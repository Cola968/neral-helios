"""Local NERAL Mini mobile access: TLS gateway logic and model privacy."""
from __future__ import annotations

import hashlib
import http.client
import json
import threading
from urllib.parse import urlencode
from pathlib import Path

import pytest

from mini.neral_mini import MiniServer, NeralModel
from mini.mobile_gateway import MobileGateway, create_mobile_gateway


class Fake:
    def __init__(self):
        self.calls=[]
        self.fail=False
    def status(self):
        return {"name":"NERAL Mini","adapter_loaded":True,
                "rank":32,"local_only":True,"experimental":True}
    def respond(self,messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("C:/secret/personal-data")
        return "Antwort aus dem simulierten lokalen NERAL-Adapter"


@pytest.fixture
def running():
    model=Fake()
    desktop=MiniServer(("127.0.0.1",0),model,
       '<html><script>const TOKEN="%%NERAL_SESSION%%";</script><p>NERAL</p></html>')
    config={"host":"192.168.17.3","port":9444,
            "pair_hash":hashlib.sha256(b"a1b2c3d4e5f60718").hexdigest()}
    gateway=create_mobile_gateway(config,desktop,bind_host="127.0.0.1",bind_port=0)
    gateway_thread=threading.Thread(target=gateway.serve_forever,daemon=True)
    gateway_thread.start()
    try:
        yield gateway, desktop, model
    finally:
        gateway.shutdown()
        gateway.server_close()
        desktop.server_close()
        gateway_thread.join(timeout=3)


def request(running,path="/",method="GET",data=None,headers=None):
    gateway,desktop,model=running
    port=gateway.server_address[1]
    conn=http.client.HTTPConnection("127.0.0.1",port,timeout=3)
    h={"Host":"192.168.17.3:9444"}
    h.update(headers or {})
    body=data
    if isinstance(data,dict):
        body=json.dumps(data).encode("utf-8")
        h.setdefault("Content-Type","application/json")
    conn.request(method,path,body=body,headers=h)
    response=conn.getresponse()
    payload=response.read().decode("utf-8")
    result=(response.status,dict(response.getheaders()),payload)
    conn.close()
    return result


def pair(running,code="a1b2c3d4e5f60718"):
    status,headers,text=request(running,"/pair","POST",
        data=urlencode({"code":code}).encode(),
        headers={"Content-Type":"application/x-www-form-urlencoded",
                 "Origin":"https://192.168.17.3:9444"})
    token=headers.get("Set-Cookie","").split(";",1)[0].split("=",1)[-1]
    return status,headers,text,token


def secure_headers(token):
    return {"Cookie":"neral_mobile="+token,
            "X-Neral-Session":token,
            "Origin":"https://192.168.17.3:9444"}


def test_mobile_requires_pairing_and_secure_cookie(running):
    status,headers,html=request(running)
    assert status==200
    assert "Kopplungscode" in html
    assert "NERAL Mini" in html
    assert request(running,"/api/status")[0]==401
    assert request(running,"/api/chat","POST",{"prompt":"Hallo"})[0]==401
    status,headers,_,token=pair(running)
    assert status==200
    assert token
    cookie=headers["Set-Cookie"]
    assert "Secure" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "Max-Age=28800" in cookie
    status,headers,html=request(running,headers={"Cookie":"neral_mobile="+token})
    assert status==200
    assert token in html
    assert running[1].token not in html
    assert "NERAL" in html
    assert "no-store" in headers["Cache-Control"]
    status,headers,raw=request(running,"/api/status",headers={"Cookie":"neral_mobile="+token})
    assert status==200
    assert json.loads(raw)["local_only"] is True


def test_mobile_chat_is_actual_same_local_model_and_never_external(running):
    _,_,_,token=pair(running)
    status,headers,raw=request(running,"/api/chat","POST",{"prompt":"Bitte Python erklären."},
                               headers=secure_headers(token))
    assert status==200
    assert json.loads(raw)["text"].startswith("Antwort aus dem simulierten")
    assert running[2].calls[0][-1]=={"role":"user","content":"Bitte Python erklären."}
    assert running[2].calls[0][0]["role"]=="system"
    assert "Strict-Transport-Security" in headers
    assert "Access-Control-Allow-Origin" not in headers


def test_cross_site_requests_cookie_theft_and_host_spoofing_are_rejected(running):
    _,_,_,token=pair(running)
    normal=secure_headers(token)
    tests=[
        dict(normal,Origin="https://attacker.invalid"),
        dict(normal,**{"X-Neral-Session":"bad"}),
        dict(normal,**{"Sec-Fetch-Site":"cross-site"}),
        dict(normal,Host="192.168.17.99:9444"),
    ]
    for headers in tests:
        assert request(running,"/api/chat","POST",{"prompt":"Hi"},headers=headers)[0]==403
    assert request(running,"/api/chat","POST",{"prompt":"Hi"},
            headers={"Origin":"https://192.168.17.3:9444",
                     "X-Neral-Session":token})[0]==401
    assert not running[2].calls


def test_bad_pairing_limited_to_five_attempts_per_minute(running):
    for _ in range(5):
        assert pair(running,code="wrong")[0]==403
    assert pair(running,code="wrong")[0]==429
    assert pair(running,code="a1b2c3d4e5f60718")[0]==429


def test_generated_model_errors_do_not_leak_local_paths(running):
    _,_,_,token=pair(running)
    running[2].fail=True
    status,headers,raw=request(running,"/api/chat","POST",{"prompt":"Hi"},
                               headers=secure_headers(token))
    assert status==503
    assert "secret" not in raw.lower()
    assert "personal-data" not in raw


def test_only_allowed_routes_and_no_unprotected_upload(running):
    _,_,_,token=pair(running)
    assert request(running,"/api/upload",headers={"Cookie":"neral_mobile="+token})[0]==404
    assert request(running,"/private.txt",headers={"Cookie":"neral_mobile="+token})[0]==404
    assert request(running,"/api/chat","POST",{"prompt":"Hi","role":"system"},
                   headers=secure_headers(token))[0]==400


def test_mobile_gateway_is_not_public_and_requires_private_lan_address():
    fake=Fake()
    mini=MiniServer(("127.0.0.1",0),fake,"mini")
    try:
        for bad in ("0.0.0.0","8.8.8.8","127.0.0.1","169.254.1.1","::1"):
            config={"host":bad,"port":9444,
                    "pair_hash":hashlib.sha256(b"abc").hexdigest()}
            with pytest.raises(ValueError):
                create_mobile_gateway(config,mini)
    finally:
        mini.server_close()


def test_setup_phone_accepts_separate_mini_port_without_changing_default(tmp_path):
    from setup_phone import setup
    mini=setup("192.168.1.88",tmp_path/"mini",port=9444)
    assert mini["port"]==9444
    assert "9444" in (tmp_path/"mini"/"HANDY-ZUGANG.txt").read_text()
    assert "Neral-Android-CA.crt" in (tmp_path/"mini"/"HANDY-ZUGANG.txt").read_text()
    desktop=setup("192.168.1.88",tmp_path/"desktop")
    assert desktop["port"]==9443
    with pytest.raises(ValueError):
        setup("192.168.1.88",tmp_path/"invalid",port=80)


def test_mobile_start_scripts_and_tls_package_are_present():
    project=Path(__file__).resolve().parents[1]
    for file in ("Start NERAL Mini Mobile.cmd","Start-NERAL-Mini-Mobile.ps1",
                 "mini/mobile_gateway.py","setup_phone.py"):
        assert (project/file).is_file()
    script=(project/"Start-NERAL-Mini-Mobile.ps1").read_text()
    assert "mini.mobile_gateway" in script
    assert "http://127.0.0.1:9691" not in script or "https://" in script
    gateway=(project/"mini/mobile_gateway.py").read_text()
    assert "ssl.SSLContext" in gateway
    assert "PROTOCOL_TLS_SERVER" in gateway
    assert "create_mobile_gateway" in gateway
    assert "upload-artifact" not in gateway
    assert "requests.post(" not in gateway
