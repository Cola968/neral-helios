"""Opt-in routes on the original Helios Flask app; never load model weights."""
from __future__ import annotations

import os
import re
import secrets
import threading

from flask import Response, jsonify, render_template, request

from neral_control.context import ProjectReader
from neral_control.providers import LocalQwenProvider, OllamaProvider
from neral_control.registry import default_registry
from stream_runtime import response_stream

_REQUEST_ID = re.compile(r'[a-zA-Z0-9-]{16,64}')


def _identity(provider):
    identity = dict(provider.identity) if provider else dict(source='tools-only', runtime_id='tools-only', base_model=None)
    identity.setdefault('provider_wrapper_version', identity.get('adapter_version'))
    identity.setdefault('lora_weights_version', None)
    identity.setdefault('active_adapter_id', None)
    return identity


def install(helios_module, *, project_root=None, search_provider=None,
            allow_model_inference=False, controller_factory=None):
    """Install once, before serving. Existing Host/Origin/session guards apply.

    Project roots and search transports are deployment choices, never request
    parameters. Inference needs both this opt-in and an explicit request provider.
    """
    if type(allow_model_inference) is not bool:
        raise ValueError('allow_model_inference must be a boolean')
    h = helios_module
    app = h.app
    if 'neral_control' in app.extensions:
        raise ValueError('Neral Control is already installed')
    if controller_factory is None:
        from neral_control.controller import ReasoningController
        controller_factory = ReasoningController
    project_reader = None
    project_unavailable = None
    if project_root is not None:
        if not ProjectReader.supported():
            project_unavailable = 'Safe rooted project reading is unavailable on this platform.'
        else:
            project_reader = ProjectReader(project_root)
    registry = default_registry(project_reader, search_provider)
    app.extensions['neral_control'] = dict(registry=registry, allow_model_inference=allow_model_inference)

    def providers():
        result = []
        if allow_model_inference and getattr(h, 'qwen25', None) is not None:
            result.append(dict(id='local-qwen', identity=_identity(LocalQwenProvider(h.qwen25)), native_think=False,
                               cancellation='cooperative-between-tokens'))
        if (allow_model_inference and getattr(h, 'ASSISTANT_ENABLED', False)
                and not getattr(h, 'PREPARATION_MODE', False)):
            result.append(dict(id='ollama', identity=_identity(OllamaProvider(h.ASSISTANT_MODEL)), native_think=True,
                               cancellation='checked-before-and-after-blocking-http'))
        return result

    @app.get('/neral/control', endpoint='neral_control_panel')
    def panel():
        response = app.make_response(render_template('neral_control.html'))
        response.set_cookie('neral_local', h.SESSION_TOKEN, httponly=True, samesite='Strict', max_age=28800, secure=False)
        return response

    @app.get('/api/neral/capabilities', endpoint='neral_control_capabilities')
    def capabilities():
        return jsonify(profiles=['fast','think','deep'], tools=registry.describe(), providers=providers(),
                       model_inference_enabled=allow_model_inference, identity=_identity(None),
                       streaming='stages-and-final', busy=h._generation_lock.locked(),
                       project_read_unavailable_reason=project_unavailable)

    def body():
        if not request.is_json:
            raise ValueError('A JSON object is required.')
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError('A JSON object is required.')
        return data

    @app.post('/api/neral/cancel', endpoint='neral_control_cancel')
    def cancel():
        try:
            job = body().get('request_id')
            if not isinstance(job, str) or not _REQUEST_ID.fullmatch(job):
                raise ValueError('Invalid request ID.')
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        with h._job_lock:
            event = h._jobs.get(job)
            if event is not None:
                event.set()
        return jsonify(cancelled=event is not None)

    @app.post('/api/neral/task', endpoint='neral_control_task')
    def task():
        try:
            data = body()
            prompt = data.get('prompt')
            if (not isinstance(prompt, str) or not prompt.strip()
                    or len(prompt.encode('utf8')) > getattr(h, 'MAX_PROMPT_BYTES', 4096)):
                raise ValueError('A nonempty prompt of at most 4096 bytes is required.')
            job = data.get('request_id', secrets.token_hex(16))
            if not isinstance(job, str) or not _REQUEST_ID.fullmatch(job):
                raise ValueError('Invalid request ID.')
            profile = data.get('profile', 'fast')
            if type(profile) is not str or profile not in {'fast','think','deep'}:
                raise ValueError('Unknown profile.')
            native_think = data.get('native_think', False)
            stream = data.get('stream', False)
            if type(native_think) is not bool or type(stream) is not bool:
                raise ValueError('Thinking and streaming options must be booleans.')
            selected = data.get('provider')
            if selected is not None and selected not in ('local-qwen','ollama'):
                raise ValueError('Unknown provider.')
            if native_think and selected != 'ollama':
                raise ValueError('Native thinking requires Ollama.')
            tools = data.get('allowed_tools', ['calculator','code_check'])
            if (not isinstance(tools, list) or not all(type(t) is str for t in tools)
                    or len(tools) != len(set(tools)) or set(tools)-set(registry.names())):
                raise ValueError('Only available reviewed tools may be enabled.')
            available_permissions = {t['permission'] for t in registry.describe() if t['available']}
            permissions = data.get('permissions', [])
            if (not isinstance(permissions, list) or not all(type(p) is str for p in permissions)
                    or set(permissions)-available_permissions):
                raise ValueError('Unknown or unavailable permission.')
            permissions = frozenset(permissions) | frozenset({'math','code_check'})
            history = data.get('history', [])
            if not isinstance(history, list) or len(history)>40:
                raise ValueError('Invalid history.')
            budget = data.get('max_new_tokens', 256)
            if type(budget) is not int or not 1 <= budget <= 512:
                raise ValueError('Invalid response token budget.')
        except (ValueError, UnicodeError) as exc:
            return jsonify(error=str(exc)), 400
        if not h._generation_lock.acquire(blocking=False):
            return jsonify(error='A generation or model change is already in progress.'), 409
        event = threading.Event()
        registered = False
        finished = False
        finish_lock = threading.Lock()
        def finish():
            nonlocal finished
            with finish_lock:
                if finished:
                    return
                finished = True
                if registered:
                    with h._job_lock:
                        if h._jobs.get(job) is event:
                            h._jobs.pop(job, None)
                h._generation_lock.release()
        try:
            provider = None
            if selected:
                if not allow_model_inference:
                    return jsonify(error='Model inference was not enabled at installation.'), 503
                if selected == 'local-qwen' and getattr(h, 'qwen25', None) is not None:
                    provider = LocalQwenProvider(h.qwen25)
                elif (selected == 'ollama' and getattr(h, 'ASSISTANT_ENABLED', False)
                      and not getattr(h, 'PREPARATION_MODE', False)):
                    provider = OllamaProvider(h.ASSISTANT_MODEL)
                if provider is None:
                    return jsonify(error='The requested provider is not already available.'), 503
            identity = _identity(provider)
            if ((provider is not None and data.get('runtime_id') != identity['runtime_id'])
                    or ('runtime_id' in data and data['runtime_id'] != identity['runtime_id'])):
                return jsonify(error='Runtime changed. Refresh capabilities and submit again.'), 409
            controller = controller_factory(registry, provider)
            with h._job_lock:
                if job in h._jobs:
                    return jsonify(error='Request ID is already active.'), 409
                h._jobs[job] = event
                registered = True
            def run(callback=None):
                def stage(update):
                    # Controller updates never contain unreviewed model/tool text.
                    if callback:
                        step = update.get('step', 0)
                        label = {'model':'Das konfigurierte Modell bereitet eine Antwort vor.',
                                 'tool':'Ein freigegebenes Werkzeug prüft die Aufgabe.',
                                 'verify':'Antwort und Werkzeugherkunft werden geprüft.'}.get(
                                     update.get('stage'), 'Die angeforderte Aufgabe wird bearbeitet.')
                        # The existing stream holds a redaction look-behind; use
                        # reviewed stage text, never model or tool protocol.
                        callback(label + ' Aktualisierungen zeigen nur Arbeitsschritte. Die finale Antwort erscheint nach Abschluss der laufenden Operation.',
                                 step if type(step) is int and step >= 0 else 0)
                result = controller.run(prompt, history=history, profile=profile, allowed_tools=tuple(tools),
                    permissions=permissions, event=event, native_think=native_think, on_progress=stage,
                    max_new_tokens=budget, runtime_id=identity['runtime_id'] if provider else None)
                result['request_id'] = job
                result['provider_identity'] = identity
                return result
            if stream:
                response = Response(response_stream(run, event, finish), mimetype='application/x-ndjson')
                response.headers['X-Accel-Buffering'] = 'no'
                response.call_on_close(event.set)
                # The existing bounded streaming worker owns cleanup after return.
                registered_stream = True
                return response
            try:
                return jsonify(run())
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
            except Exception:
                app.logger.exception('Neral Control task failed')
                return jsonify(error='The task failed. Check the server log.'), 500
        finally:
            if not locals().get('registered_stream', False):
                finish()
    return app.extensions['neral_control']
