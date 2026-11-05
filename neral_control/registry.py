"""Explicit local tool allowlist. Deadlines are cooperative, never worker threads.

Handlers run synchronously and receive a cancellation event and monotonic deadline.
Only reviewed builtins may be registered; no imports or plugin discovery from data.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import threading
import time
from typing import Callable

from neral_assistant import CALCULATOR_SCHEMA, CODE_CHECK_SCHEMA
from security_guard import sanitize_output

_NAMES = frozenset({'calculator', 'code_check', 'project_files', 'web_search'})


@dataclass(frozen=True)
class ToolContext:
    permissions: frozenset[str] = field(default_factory=frozenset)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    deadline: float = math.inf


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    input_schema: dict
    output_schema: dict
    permission: str
    timeout_seconds: float
    handler: Callable | None
    description: str


def _validate(value, schema, depth=0):
    if depth > 20:
        raise ValueError('JSON nesting exceeds limit')
    typ = schema.get('type')
    types = {'object':dict, 'array':list, 'string':str, 'integer':int,
             'number':(int,float), 'boolean':bool}
    if typ not in types or not isinstance(value, types[typ]):
        raise ValueError('Schema type mismatch')
    if typ in {'integer','number'}:
        if isinstance(value,bool) or not math.isfinite(value):
            raise ValueError('Invalid numeric value')
        if value < schema.get('minimum', -math.inf) or value > schema.get('maximum', math.inf):
            raise ValueError('Number outside bounds')
    elif typ == 'string':
        if len(value) > schema.get('maxLength', 65536) or len(value) < schema.get('minLength',0):
            raise ValueError('String outside bounds')
    elif typ == 'array':
        if len(value) > schema.get('maxItems',1000):
            raise ValueError('Array outside bounds')
        for item in value:
            _validate(item, schema.get('items', _json_schema(item)), depth+1)
    elif typ == 'object':
        if not all(type(k) is str for k in value) or len(value)>1000:
            raise ValueError('Invalid object keys')
        props = schema.get('properties',{})
        if set(schema.get('required',[]))-set(value):
            raise ValueError('Missing required field')
        additional = schema.get('additionalProperties',False)
        if additional is False and set(value)-set(props):
            raise ValueError('Unexpected fields')
        for key,item in value.items():
            if item is None and key not in props and additional is True:
                continue
            _validate(item, props.get(key, additional if isinstance(additional,dict) else _json_schema(item)), depth+1)


def _json_schema(value):
    for kind,typ in [('boolean',bool),('integer',int),('number',float),('string',str),('array',list),('object',dict)]:
        if isinstance(value,typ):
            return {'type':kind, 'additionalProperties':True}
    raise ValueError('Non-JSON value')


def _redact(value):
    if isinstance(value,str):
        return sanitize_output(value)[0]
    if isinstance(value,list):
        return [_redact(x) for x in value]
    if isinstance(value,dict):
        return {k:_redact(v) for k,v in value.items()}
    return value


class _NoCalculation(ValueError):
    pass


class ToolRegistry:
    def __init__(self):
        self._specs = {}

    def register(self, spec: ToolSpec):
        if (not isinstance(spec,ToolSpec) or spec.name not in _NAMES or spec.name in self._specs
            or not spec.version or not spec.permission or not math.isfinite(spec.timeout_seconds)
            or not 0 < spec.timeout_seconds <= 60 or (spec.handler is not None and not callable(spec.handler))):
            raise ValueError('Invalid or duplicate reviewed tool specification')
        self._specs[spec.name] = ToolSpec(spec.name,spec.version,deepcopy(spec.input_schema),
            deepcopy(spec.output_schema),spec.permission,spec.timeout_seconds,spec.handler,spec.description)

    def names(self):
        return tuple(name for name,spec in self._specs.items() if spec.handler is not None)

    @property
    def available_names(self):
        return self.names()

    def describe(self):
        return [dict(name=s.name,version=s.version,input_schema=deepcopy(s.input_schema),
            output_schema=deepcopy(s.output_schema),permission=s.permission,
            timeout_seconds=s.timeout_seconds,description=s.description,
            available=s.handler is not None,timeout_mode='cooperative') for s in self._specs.values()]

    def schemas(self, allowed_names=None):
        allowed = set(self.names() if allowed_names is None else allowed_names)
        return [{'type':'function','function':{'name':s.name,'description':s.description,
                'parameters':deepcopy(s.input_schema)}} for s in self._specs.values()
                if s.name in allowed and s.handler is not None]

    def route_calculation(self, prompt, context: ToolContext):
        from chat_math import tool_answer
        def handler(args,ctx):
            answer = tool_answer(args['prompt'])
            if answer is None:
                raise _NoCalculation()
            return {'text':answer.text,'kind':answer.kind,'model_generated':False}
        spec = ToolSpec('calculator','1',{'type':'object','properties':{'prompt':{'type':'string','maxLength':4096}},'required':['prompt'],'additionalProperties':False},
            {'type':'object','additionalProperties':True},'math',2,handler,
            'Existing whole-request arithmetic router.')
        return self._run('calculator',{'prompt':prompt},context,spec)

    def execute(self, name, arguments, context: ToolContext):
        spec = self._specs.get(name) if isinstance(name,str) else None
        return self._run(name,arguments,context,spec)

    def _run(self, name, arguments, context, spec):
        started = time.monotonic()
        source = {'calculator':'tool:calculator','code_check':'NERAL static AST checker',
                  'project_files':'local project','web_search':'configured public search'}.get(name,'unavailable') if isinstance(name,str) else 'unavailable'
        def result(status, output=None, code=None):
            return {'tool_id':name if isinstance(name,str) else 'invalid','version':spec.version if spec else None,
                    'status':status,'output':output,'provenance':{'source':source,'executed':False},
                    'error':None if code is None else {'code':code,'message':'Tool request did not complete.'},
                    'elapsed_seconds':round(time.monotonic()-started,6)}
        if spec is None or spec.handler is None:
            return result('unavailable',code='not_configured')
        if not isinstance(context,ToolContext) or spec.permission not in context.permissions:
            return result('denied',code='permission_denied')
        if context.cancel_event.is_set():
            return result('cancelled',code='cancelled')
        deadline = min(context.deadline, started+spec.timeout_seconds)
        if time.monotonic() >= deadline:
            return result('timeout',code='deadline_exceeded')
        bounded = ToolContext(context.permissions,context.cancel_event,deadline)
        try:
            _validate(arguments,spec.input_schema)
            if len(json.dumps(arguments,allow_nan=False).encode('utf8')) > 65536:
                raise ValueError('Input too large')
            output = spec.handler(deepcopy(arguments),bounded)
            if context.cancel_event.is_set():
                return result('cancelled',code='cancelled')
            if time.monotonic() >= deadline:
                return result('timeout',code='deadline_exceeded')
            if name == 'web_search' and isinstance(output,dict) and output.get('available') is False:
                return result('unavailable',code='provider_unavailable')
            _validate(output,spec.output_schema)
            if len(json.dumps(output,allow_nan=False).encode('utf8')) > 65536:
                raise ValueError('Output too large')
            return result('ok',_redact(output))
        except _NoCalculation:
            return None
        except TimeoutError:
            return result('timeout',code='deadline_exceeded')
        except PermissionError:
            return result('denied',code='access_denied')
        except Exception:
            return result('error',code='invalid_request_or_output')


def default_registry(project_reader=None, search_provider=None):
    # Import explicit reviewed helpers; call canonical envelopes irrespective of
    # provider metadata. The existing helpers remain the only execution engines.
    from neral_assistant import _calculate, _code_check
    registry = ToolRegistry()
    output = {'type':'object','additionalProperties':True}
    for schema,permission,helper in [(CALCULATOR_SCHEMA,'math',_calculate),(CODE_CHECK_SCHEMA,'code_check',_code_check)]:
        fn = schema['function']
        inputs = deepcopy(fn['parameters'])
        key = 'expression' if fn['name']=='calculator' else 'code'
        inputs['properties'][key]['maxLength'] = 120 if key=='expression' else 4096
        def handler(args,ctx, helper=helper,name=fn['name']):
            return json.loads(helper({'function':{'name':name,'arguments':args}}))
        registry.register(ToolSpec(fn['name'],'1',inputs,output,permission,2,handler,fn['description']))
    registry.register(ToolSpec('project_files','1',{'type':'object','properties':{'relative_path':{'type':'string','minLength':1,'maxLength':1024}},'required':['relative_path'],'additionalProperties':False}, output,'project_read',2,
        (lambda args,ctx: project_reader.read(args['relative_path'])) if project_reader is not None else None,
        'Read a bounded text file from the explicitly configured project root.'))
    registry.register(ToolSpec('web_search','1',{'type':'object','properties':{'query':{'type':'string','minLength':1,'maxLength':400},'limit':{'type':'integer','minimum':1,'maximum':5}},'required':['query'],'additionalProperties':False},output,'network',15,
        (lambda args,ctx: search_provider.search(args['query'],limit=args.get('limit',5),cancel_event=ctx.cancel_event,deadline=ctx.deadline)) if search_provider is not None else None,
        'Search public sources with the explicitly configured free provider.'))
    return registry
