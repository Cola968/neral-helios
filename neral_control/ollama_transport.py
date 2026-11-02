"""Bounded, cancellable HTTP/1.1 NDJSON transport to numeric IPv4 loopback.

No DNS, proxies, redirects, subprocesses, worker threads or model downloads.
Closing the client connection does not prove server-side generation stopped.
"""
from dataclasses import dataclass
import errno
import json
import math
import selectors
import socket
import time


class TransportError(ValueError):
    """The loopback HTTP/NDJSON response violated the transport contract."""


@dataclass
class TransportResult:
    data: dict
    incomplete: bool
    timed_out: bool = False
    cancel_requested: bool = False
    cancel_acknowledged: bool = False
    transport_closed: bool | None = None
    provider_execution_stopped: bool | None = None


class _Stopped(Exception):
    def __init__(self, timed_out):
        self.timed_out = timed_out


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TransportError('Duplicate JSON frame key.')
        result[key] = value
    return result


def _invalid_constant(_):
    raise TransportError('Nonfinite JSON frame value.')


class _Frames:
    def __init__(self, model, limit, on_text=None):
        self.model, self.limit, self.on_text = model, limit, on_text
        self.pending = bytearray()
        self.content, self.calls = [], []
        self.done, self.terminal = False, {}
        self.count = 0

    def feed(self, raw, eof=False):
        self.pending.extend(raw)
        if len(self.pending) > self.limit:
            raise TransportError('NDJSON frame exceeds limit.')
        while b'\n' in self.pending:
            line, _, rest = self.pending.partition(b'\n')
            self.pending = bytearray(rest)
            if line.strip():
                self.frame(line)
        if eof and self.pending:
            self.frame(bytes(self.pending)); self.pending.clear()

    def frame(self, raw):
        if self.done:
            raise TransportError('Data after terminal Ollama frame.')
        self.count += 1
        if self.count > 4096:
            raise TransportError('Too many response frames.')
        try:
            row = json.loads(raw, object_pairs_hook=_object, parse_constant=_invalid_constant)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise TransportError('Malformed Ollama JSON frame.') from exc
        if (not isinstance(row, dict) or type(row.get('done')) is not bool
                or row.get('model') != self.model or 'error' in row):
            raise TransportError('Invalid Ollama frame identity or status.')
        message = row.get('message', {})
        if (not isinstance(message, dict) or message.get('role', 'assistant') != 'assistant'
                or not isinstance(message.get('content', ''), str)):
            raise TransportError('Invalid Ollama message fragment.')
        text = message.get('content', '')
        try:
            text.encode('utf8')
        except UnicodeError as exc:
            raise TransportError('Invalid response Unicode.') from exc
        self.content.append(text)
        if text and self.on_text is not None:
            self.on_text(text)
        calls = message.get('tool_calls', [])
        if not isinstance(calls, list) or len(self.calls) + len(calls) > 4:
            raise TransportError('Invalid or excessive streamed tool calls.')
        self.calls.extend(calls)
        # Never retain native private thinking, arbitrary metadata, or secrets in errors.
        if row['done']:
            if ('done_reason' in row and (not isinstance(row['done_reason'], str)
                    or row['done_reason'] not in {'stop', 'length'})):
                raise TransportError('Invalid Ollama completion reason.')
            self.done = True
            for key in ('done_reason', 'eval_count', 'prompt_eval_count', 'eval_duration'):
                if key in row:
                    self.terminal[key] = row[key]

    def result(self, stopped=None, transport_closed=None):
        incomplete = stopped is not None or not self.done or self.terminal.get('done_reason') == 'length'
        data = {'model': self.model, 'done': self.done,
                'message': {'role': 'assistant', 'content': ''.join(self.content),
                            'tool_calls': [] if incomplete else self.calls}, **self.terminal}
        if not self.done and stopped is None:
            data['done_reason'] = 'connection_closed'
        return TransportResult(data=data, incomplete=incomplete,
            transport_closed=transport_closed,
            timed_out=bool(stopped and stopped.timed_out),
            cancel_requested=bool(stopped is not None and not stopped.timed_out),
            cancel_acknowledged=bool(stopped is not None and not stopped.timed_out))


class _Body:
    """Bounded HTTP body decoder; streams decoded bytes into NDJSON parser."""
    def __init__(self, frames, headers):
        self.frames = frames
        self.chunked = headers.get('transfer-encoding') == 'chunked'
        if 'transfer-encoding' in headers and not self.chunked:
            raise TransportError('Unsupported transfer encoding.')
        if self.chunked and 'content-length' in headers:
            raise TransportError('Ambiguous HTTP body framing.')
        self.remaining = None
        if 'content-length' in headers:
            try:
                self.remaining = int(headers['content-length'])
            except ValueError as exc:
                raise TransportError('Invalid content length.') from exc
            if self.remaining < 0 or self.remaining > frames.limit:
                raise TransportError('Response content length exceeds limit.')
        self.pending, self.chunk_size, self.finished = bytearray(), None, False

    def feed(self, raw):
        if not self.chunked:
            if self.remaining is not None:
                if len(raw) > self.remaining:
                    raise TransportError('Excess HTTP response bytes.')
                self.remaining -= len(raw)
            self.frames.feed(raw)
            return
        self.pending.extend(raw)
        while not self.finished:
            if self.chunk_size is None:
                mark = self.pending.find(b'\r\n')
                if mark < 0:
                    if len(self.pending) > 64:
                        raise TransportError('Oversized HTTP chunk header.')
                    return
                line = bytes(self.pending[:mark]); del self.pending[:mark+2]
                if not line or len(line) > 16 or any(c not in b'0123456789abcdefABCDEF' for c in line):
                    raise TransportError('Invalid HTTP chunk header.')
                self.chunk_size = int(line, 16)
                if self.chunk_size > self.frames.limit:
                    raise TransportError('HTTP chunk exceeds response limit.')
            if len(self.pending) < self.chunk_size + 2:
                return
            size = self.chunk_size
            if self.pending[size:size+2] != b'\r\n':
                raise TransportError('Invalid HTTP chunk delimiter.')
            data = bytes(self.pending[:size]); del self.pending[:size+2]
            self.chunk_size = None
            if size == 0:
                self.finished = True
            else:
                self.frames.feed(data)


def stream_ollama(payload, *, event, deadline, port=11434, max_response_bytes=256*1024, on_text=None):
    """Process a fixed /api/chat stream synchronously with <=20ms polling waits.

    ``port`` exists for explicitly injected loopback test endpoints. The host and
    path cannot be supplied by prompts. Deadline is absolute monotonic time.
    Response limits include private-thinking frames even though never exposed.
    """
    if (not isinstance(payload, dict) or not isinstance(payload.get('model'), str)
            or type(port) is not int or not 1 <= port <= 65535
            or type(max_response_bytes) is not int or not 64 <= max_response_bytes <= 256*1024
            or type(deadline) not in (float, int) or not math.isfinite(deadline)
            or on_text is not None and not callable(on_text)):
        raise TransportError('Invalid loopback request configuration.')
    frames = _Frames(payload['model'], max_response_bytes, on_text)
    def check():
        if event.is_set():
            raise _Stopped(False)
        if time.monotonic() >= deadline:
            raise _Stopped(True)
    conn = None
    selector = None
    try:
        check()
        body = json.dumps({**payload, 'stream': True}, ensure_ascii=False, allow_nan=False).encode('utf8')
        if len(body) > 128*1024:
            raise TransportError('Request body exceeds limit.')
        request = (f'POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n'
                   'Content-Type: application/json\r\nAccept: application/x-ndjson\r\n'
                   f'Content-Length: {len(body)}\r\nConnection: close\r\n\r\n').encode('ascii') + body
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.setblocking(False)
        pending = conn.connect_ex(('127.0.0.1', port))
        if pending not in {0, errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, 10035, 10036, 10037}:
            raise TransportError('Loopback Ollama connection failed.')
        check()
        selector = selectors.DefaultSelector()
        selector.register(conn, selectors.EVENT_WRITE)
        def wait(mask):
            selector.modify(conn, mask)
            while True:
                check()
                if selector.select(min(.02, max(0, deadline-time.monotonic()))):
                    check()
                    return
        if pending:
            wait(selectors.EVENT_WRITE)
            if conn.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR):
                raise TransportError('Loopback Ollama connection failed.')
        offset = 0
        while offset < len(request):
            wait(selectors.EVENT_WRITE)
            try:
                sent = conn.send(request[offset:])
            except BlockingIOError:
                continue
            if sent == 0:
                raise TransportError('Ollama closed request connection.')
            offset += sent
        raw_headers = bytearray(); decoder = None; body_bytes = 0; wire_bytes = 0
        while not frames.done:
            wait(selectors.EVENT_READ)
            try:
                raw = conn.recv(8192)
            except BlockingIOError:
                continue
            check()
            if not raw:
                if decoder is None:
                    raise TransportError('Ollama closed before HTTP headers.')
                frames.feed(b'', eof=True)
                break
            wire_bytes += len(raw)
            if wire_bytes > max_response_bytes*2 + 16384:
                raise TransportError('HTTP response exceeds wire limit.')
            if decoder is None:
                raw_headers.extend(raw)
                split = raw_headers.find(b'\r\n\r\n')
                if split < 0:
                    if len(raw_headers) > 16384:
                        raise TransportError('HTTP headers exceed limit.')
                    continue
                if split > 16384:
                    raise TransportError('HTTP headers exceed limit.')
                head, raw = bytes(raw_headers[:split]), bytes(raw_headers[split+4:])
                lines = head.split(b'\r\n')
                if lines[0] not in (b'HTTP/1.1 200 OK', b'HTTP/1.0 200 OK'):
                    raise TransportError('Non-success Ollama HTTP response.')
                headers = {}
                for line in lines[1:]:
                    if b':' not in line:
                        raise TransportError('Malformed HTTP header.')
                    key, value = line.split(b':', 1)
                    try:
                        key, value = key.decode('ascii').lower(), value.strip().decode('ascii')
                    except UnicodeError as exc:
                        raise TransportError('Invalid HTTP header bytes.') from exc
                    if key in headers:
                        raise TransportError('Duplicate HTTP response header.')
                    headers[key] = value.lower()
                decoder = _Body(frames, headers)
            body_bytes += len(raw)
            if body_bytes > max_response_bytes:
                raise TransportError('Ollama response exceeds byte limit.')
            decoder.feed(raw)
            check()
            if decoder.finished or decoder.remaining == 0:
                frames.feed(b'', eof=True)
                break
        check()
        return frames.result(transport_closed=True)
    except _Stopped as stopped:
        return frames.result(stopped, transport_closed=True if conn is not None else None)
    except (OSError, UnicodeError, TypeError, RecursionError) as exc:
        raise TransportError('Loopback Ollama transport failed.') from exc
    finally:
        if selector is not None:
            selector.close()
        if conn is not None:
            conn.close()
