from types import SimpleNamespace
import pytest
import frontier_runtime as runtime

class Enc:
    def __init__(self,ids):self.ids=ids
class FakeTokenizer:
    def encode(self,text):return Enc(list(text.encode("utf-8")))

def test_validate_messages_requires_alternating_user_and_last_user():
    good=[{"role":"user","content":"Hallo"},{"role":"assistant","content":"Hi"},
          {"role":"user","content":"Noch eine Frage"}]
    assert runtime.validate_messages(good,64)[-1]["role"]=="user"
    with pytest.raises(ValueError):runtime.validate_messages([{"role":"assistant","content":"bad"}],64)
    with pytest.raises(ValueError):runtime.validate_messages(good[:-1],64)

def test_encode_context_drops_complete_old_pairs_only():
    rt={"bos":1,"tokenizer":FakeTokenizer()}
    messages=[{"role":"user","content":"A"*25},{"role":"assistant","content":"B"*25},
              {"role":"user","content":"kurz"}]
    ids,dropped=runtime.encode_context(rt,messages,16,80)
    assert dropped==1 and ids[0]==1
    with pytest.raises(ValueError):
        runtime.encode_context(rt,[{"role":"user","content":"X"*200}],16,40)

def test_health_reports_kv_cache_streaming_and_cpu_without_torch_cuda_calls():
    rt={"device":SimpleNamespace(type="cpu"),"torch":None,"checkpoint_sha":"a"*64,
        "tokenizer_sha":"b"*64,"processed_tokens":123}
    health=runtime.runtime_health(rt,512)
    assert health["runtime_family"]=="frontier350-bpe-v1"
    assert health["kv_cache"] is True and health["streaming"] is True
    assert health["cancel_supported"] is True and health["device"]=="cpu"
    assert health["max_new_tokens"]==runtime.MAX_NEW_TOKENS==256

def test_repetition_guard_detects_real_loop_but_not_short_text():
    assert runtime._looping("kurz text") is False
    text=("alpha beta gamma " * 8).strip()
    assert runtime._looping(text) is True
