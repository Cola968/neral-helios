import hashlib
import socket
import qualify_frontier350 as q

def test_sha_text_is_stable():
    assert q.sha_text("abc")==hashlib.sha256(b"abc").hexdigest()

def test_summary_uses_median_and_peak():
    runs=[
        {"tokens_per_second":20.0,"first_token_ms":120.0,"peak_vram_mib":900.0,"generated_tokens":8},
        {"tokens_per_second":40.0,"first_token_ms":80.0,"peak_vram_mib":1100.0,"generated_tokens":0},
        {"tokens_per_second":30.0,"first_token_ms":100.0,"peak_vram_mib":1000.0,"generated_tokens":4},
    ]
    s=q.summary(runs)
    assert s["median_decode_tokens_per_second"]==30.0
    assert s["median_first_token_ms"]==100.0
    assert s["max_peak_vram_mib"]==1100.0
    assert s["responses_with_tokens"]==2

def test_free_port_returns_bindable_high_port():
    port=q.free_port()
    assert 1024<=port<=65535
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1",port))
