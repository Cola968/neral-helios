"""Real loopback HTTP fixtures; no Ollama, model generation or production workers."""
import contextlib
import errno
import importlib
import json
import os
import socket
import threading
import time

import pytest


def api():
    try:
        return importlib.import_module('neral_control.ollama_transport')
    except ModuleNotFoundError:
        pytest.fail('Missing controllably cancellable Ollama HTTP transport')


PAYLOAD = {'model':'qwen3.5:9b', 'messages':[{'role':'user','content':'hello'}], 'stream':True}


def frame(**changes):
    value = {'model':'qwen3.5:9b', 'message':{'role':'assistant','content':''}, 'done':False}
    value.update(changes)
    return json.dumps(value).encode() + b'\n'


@contextlib.contextmanager
def server(parts, *, cancel=None, hold=False, chunked=False):
    """Bounded server fixture owns and joins its sole worker on every exit."""
    listener = socket.socket()
    listener.bind(('127.0.0.1',0)); listener.listen(1); listener.settimeout(2)
    port = listener.getsockname()[1]
    stop = threading.Event(); accepted = threading.Event(); peer_closed = threading.Event()
    received = []
    def serve():
        try:
            with listener.accept()[0] as conn:
                accepted.set(); conn.settimeout(2)
                request = b''
                while b'\r\n\r\n' not in request:
                    request += conn.recv(4096)
                headers, body = request.split(b'\r\n\r\n',1)
                count = int(next(h.split(b':',1)[1] for h in headers.split(b'\r\n')[1:] if h.lower().startswith(b'content-length:')))
                while len(body) < count: body += conn.recv(4096)
                received.append(json.loads(body[:count]))
                encoding = b'Transfer-Encoding: chunked\r\n' if chunked else b''
                conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n'+encoding+b'Connection: close\r\n\r\n')
                for part in parts:
                    conn.sendall((hex(len(part))[2:].encode()+b'\r\n'+part+b'\r\n') if chunked else part)
                if cancel is not None:
                    cancel['at'] = time.monotonic(); cancel['event'].set()
                if hold:
                    conn.settimeout(.02)
                    while not stop.is_set():
                        try:
                            if not conn.recv(1):peer_closed.set();break
                        except socket.timeout:pass
                        except ConnectionResetError:peer_closed.set();break
                elif chunked:
                    conn.sendall(b'0\r\n\r\n')
        except (OSError, ValueError):
            pass
    worker=threading.Thread(target=serve,name='controlled-http-fixture')
    worker.start()
    try:
        yield port, received, peer_closed
    finally:
        stop.set();listener.close();worker.join(3)
        assert not worker.is_alive(), 'test server worker leaked'


def invoke(port, event=None, **kwargs):
    return api().stream_ollama(PAYLOAD, event=event or threading.Event(),
        deadline=time.monotonic()+2, port=port, **kwargs)


@pytest.mark.parametrize('sample', range(20))
def test_timestamped_cancel_reports_latency_after_socket_close(sample):
    module = api()
    event = module.CancellationEvent()
    signal = {'event': event}
    with server([b'{"message":'], cancel=signal, hold=True) as (port, _, closed):
        result = invoke(port, event)
        elapsed = (time.monotonic() - signal['at']) * 1000
        assert closed.wait(.5)
    assert result.completion_status == 'cancelled'
    assert 0 <= result.cancel_ack_latency_ms <= elapsed < 250
    assert result.transport_closed and result.provider_execution_stopped is None
    print(f'cancel_ack_latency_ms={result.cancel_ack_latency_ms:.3f}; external_ms={elapsed:.3f}')


def test_request_timestamp_is_first_set_and_can_be_reset():
    event = api().CancellationEvent()
    event.set(); first = event.requested_at
    event.set()
    assert event.requested_at == first
    event.clear()
    assert not event.is_set() and event.requested_at is None


def test_plain_event_has_no_invented_request_timestamp():
    event = threading.Event(); event.set()
    result = invoke(1, event)
    assert result.completion_status == 'cancelled'
    assert result.cancel_ack_latency_ms is None


@pytest.mark.parametrize('ending,status', [
    ([], 'incomplete'), ([frame(done=True, done_reason='length')], 'incomplete'),
    ([frame(done=True, done_reason='stop')], 'complete')])
def test_completion_status_taxonomy(ending, status):
    with server([frame(), *ending]) as (port, _, _):
        assert invoke(port).completion_status == status


def test_cancel_before_connect_does_not_open_socket(monkeypatch):
    module=api(); event=threading.Event(); event.set()
    monkeypatch.setattr(socket,'socket',lambda *a,**k:pytest.fail('precancel opened socket'))
    result=module.stream_ollama(PAYLOAD,event=event,deadline=time.monotonic()+1)
    assert result.cancel_requested and result.cancel_acknowledged and result.incomplete
    assert result.transport_closed is None and result.provider_execution_stopped is None
    assert result.data['message']['tool_calls']==[]


@pytest.mark.parametrize('chunked',[False,True])
def test_stream_aggregates_complete_frames_without_private_thinking(chunked):
    parts=[frame(message={'role':'assistant','content':'Hel','thinking':'PRIVATE'}),
           frame(message={'role':'assistant','content':'lo'}),
           frame(done=True,done_reason='stop',eval_count=2,prompt_eval_count=7)]
    with server(parts,chunked=chunked) as (port,requests,_):
        result=invoke(port)
    assert result.data['message']['content']=='Hello'
    assert result.data['done'] is True and not result.incomplete
    assert result.data['eval_count']==2 and result.data['prompt_eval_count']==7
    assert 'PRIVATE' not in repr(result.data)
    assert requests[0]['stream'] is True
    assert result.transport_closed is True and result.provider_execution_stopped is None


@pytest.mark.parametrize('partial',[
    frame(message={'role':'assistant','content':'partial'}),
    frame(message={'role':'assistant','content':'','tool_calls':[{'function':{'name':'calculator','arguments':{'expression':'2+2'}}}]}),
    b'{"model":"qwen3.5:9b","message":{"tool_calls":[',
])
def test_mid_stream_cancel_closes_connection_and_discards_calls(partial):
    event=threading.Event(); signal={'event':event}
    with server([partial],cancel=signal,hold=True) as (port,_,closed):
        result=invoke(port,event)
        latency=(time.monotonic()-signal['at'])*1000
        assert closed.wait(.5), 'client did not close its socket'
    assert latency < 250, f'controlled local cancellation took {latency:.3f}ms'
    assert result.cancel_requested and result.cancel_acknowledged and result.transport_closed
    assert result.incomplete and result.data['message']['tool_calls']==[]
    assert result.provider_execution_stopped is None
    print(f'controlled_cancel_latency_ms={latency:.3f}')


def test_deadline_is_distinct_from_user_cancellation_and_closes_socket():
    module=api(); event=threading.Event()
    with server([],hold=True) as (port,_,closed):
        result=module.stream_ollama(PAYLOAD,event=event,deadline=time.monotonic()+.05,port=port)
        assert closed.wait(.5)
    assert result.timed_out and not result.cancel_requested and not result.cancel_acknowledged
    assert not event.is_set() and result.incomplete and result.transport_closed


@pytest.mark.parametrize('parts',[
    [b'not json\n'], [frame(model='foreign')], [b'{"done":NaN}\n'],
    [b'{"done":false,"done":true}\n'], [b'[]\n'],
])
def test_malformed_or_wrong_model_frames_fail_closed(parts):
    module=api()
    with server(parts) as (port,_,_):
        with pytest.raises(module.TransportError):invoke(port)


def test_eof_and_length_discard_all_tool_calls():
    tool={'function':{'name':'calculator','arguments':{'expression':'2+2'}}}
    for ending in ([], [frame(done=True,done_reason='length')]):
        with server([frame(message={'role':'assistant','content':'partial','tool_calls':[tool]}),*ending]) as (port,_,_):
            result=invoke(port)
        assert result.incomplete and result.data['message']['tool_calls']==[]


def test_complete_tools_are_released_only_after_terminal_frame():
    tool={'function':{'name':'calculator','arguments':{'expression':'2+2'}}}
    with server([frame(message={'role':'assistant','content':'','tool_calls':[tool]}),frame(done=True,done_reason='stop')]) as (port,_,_):
        result=invoke(port)
    assert not result.incomplete and result.data['message']['tool_calls']==[tool]


def test_response_and_frame_limits_close_connection():
    module=api()
    with server([b'x'*200],hold=True) as (port,_,closed):
        with pytest.raises(module.TransportError):invoke(port,max_response_bytes=100)
        assert closed.wait(.5)


def test_connect_phase_observes_cancel_without_worker(monkeypatch):
    module=api(); event=threading.Event()
    real=socket.socket
    class PendingSocket:
        def __init__(self,*a,**kw):self.sock=real(*a,**kw)
        def setblocking(self,*a):return self.sock.setblocking(*a)
        def connect_ex(self,address):event.set();return errno.EINPROGRESS
        def close(self):self.sock.close()
        def fileno(self):return self.sock.fileno()
    monkeypatch.setattr(socket,'socket',PendingSocket)
    result=module.stream_ollama(PAYLOAD,event=event,deadline=time.monotonic()+1)
    assert result.cancel_acknowledged and result.transport_closed


def test_repeated_requests_leave_no_transport_workers_or_fds():
    module=api(); names={t.ident for t in threading.enumerate()}
    fd_count=len(os.listdir('/proc/self/fd')) if os.path.isdir('/proc/self/fd') else None
    for _ in range(3):
        with server([frame(done=True,done_reason='stop')]) as (port,_,_):
            assert invoke(port).transport_closed
    assert {t.ident for t in threading.enumerate()}==names
    if fd_count is not None:
        assert len(os.listdir('/proc/self/fd'))==fd_count


def test_cancel_after_received_text_retains_unverified_partial_and_closes():
    module=api(); event=threading.Event(); text=[]
    def received(value):
        text.append(value);event.set()
    with server([frame(message={'role':'assistant','content':'received-partial'})],hold=True) as (port,_,closed):
        result=module.stream_ollama(PAYLOAD,event=event,deadline=time.monotonic()+2,
                                    port=port,on_text=received)
        assert closed.wait(.5)
    assert text==['received-partial']
    assert result.data['message']['content']=='received-partial'
    assert result.incomplete and result.cancel_acknowledged


def test_real_stream_provider_controller_never_executes_cancelled_tool(monkeypatch):
    from neral_control import providers
    from neral_control.controller import ReasoningController
    from neral_control.registry import default_registry
    module=api(); event=threading.Event(); signal={'event':event}
    tool={'function':{'name':'calculator','arguments':{'expression':'2+2'}}}
    with server([frame(message={'role':'assistant','content':'','tool_calls':[tool]})],cancel=signal,hold=True) as (port,_,closed):
        def transport(payload,**kwargs):
            return module.stream_ollama(payload,port=port,**kwargs)
        monkeypatch.setattr(providers,'stream_ollama',transport)
        result=ReasoningController(default_registry(),providers.OllamaProvider()).run(
            'Use a suitable tool to answer the question.',profile='think',event=event,
            permissions=frozenset({'math'}))
        assert closed.wait(.5)
    assert result['cancelled'] and result['tool_calls']==0 and result['tools_used']==[]
    assert result['incomplete'] and not result['text']


def test_controlled_cancellation_latency_distribution():
    samples=[]
    for _ in range(20):
        event=threading.Event(); signal={'event':event}
        with server([frame(message={'role':'assistant','content':'part'})],cancel=signal,hold=True) as (port,_,closed):
            result=invoke(port,event)
            samples.append(round((time.monotonic()-signal['at'])*1000,3))
            assert closed.wait(.5)
        assert result.cancel_acknowledged
    assert max(samples)<250
    print('controlled_cancel_samples_ms='+json.dumps(samples))


def test_missing_model_cannot_be_synthesized_as_verified_identity():
    module=api()
    with server([b'{"message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}\n']) as (port,_,_):
        with pytest.raises(module.TransportError):invoke(port)


@pytest.mark.parametrize('reason',['invented',7,{'action':'stop'}])
def test_invalid_terminal_reason_is_rejected_before_incomplete_return(reason):
    module=api()
    with server([frame(done=True,done_reason=reason)]) as (port,_,_):
        with pytest.raises(module.TransportError):invoke(port)


def test_precancel_reports_no_connection_existed_rather_than_confirmed_close():
    module=api(); event=threading.Event(); event.set()
    result=module.stream_ollama(PAYLOAD,event=event,deadline=time.monotonic()+1)
    assert result.transport_closed is None
