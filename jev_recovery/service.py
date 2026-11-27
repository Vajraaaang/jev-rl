"""Localhost recovery lab. HTTP sessions never execute production operations."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit
import uuid

from .environment import RecoveryEnv, Scenario, Action
from .jev import DiagnosticProvider

PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PACKAGE.parent / 'artifacts' / 'research'
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}
POLICIES = ('checklist', 'rl_without_jev', 'rl_with_jev')
ACTIONS = ('logs', 'config', 'connections', 'dependency', 'jev', 'restart',
           'rollback', 'wait', 'stop', 'escalate')
MAX_BODY = 8192
MAX_SESSIONS = 64


def _checkpoint(output, name):
    """Only fixed checkpoint names under the configured experiment are loadable."""
    candidates = (output / 'checkpoints' / f'{name}.json',
                  output / 'checkpoints' / f'{name}_11.json',
                  output / 'checkpoints' / f'{name}-11.json',
                  output / 'checkpoints' / f'{name}_seed11.json',
                  output / 'checkpoints' / f'{name}-seed-11.json')
    return next((path for path in candidates if path.is_file()), None)


def create_server(host='127.0.0.1', port=8080, output=None, *, advisor_mode='offline', cache_path=None):
    """Create an unstarted testable server. All mutation uses one serialization lock."""
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('Recovery Lab binds localhost only')
    output = Path(output) if output is not None else DEFAULT_OUTPUT
    if not output.is_absolute():
        raise ValueError('Benchmark output must be an absolute path')
    output = output.resolve()
    if advisor_mode == 'live' and not os.environ.get('TYPESAFE_API_KEY', '').strip():
        raise ValueError('Live Jev requires TYPESAFE_API_KEY or serve --advisor live --prompt-key')
    # Share the provider so sessions use the same cache and acquisition budgets.
    advisor = DiagnosticProvider(mode=advisor_mode, cache_path=cache_path)
    advisor_source = 'offline-heuristic' if advisor_mode == 'offline' else 'hosted-jev'
    advisor_info = {'advisor_mode': advisor_mode, 'advisor_source': advisor_source,
                    'hosted_api_enabled': advisor_mode == 'live'}
    sessions = {}
    lock = threading.RLock()
    legacy = RecoveryEnv(Scenario(()))

    def snapshot(identifier, session):
        env = session['env']
        return {'id': identifier, 'observation': env.observation(), 'metrics': env.metrics(),
                'trace': env.trace, 'decisions': session['decisions'], 'done': env.done,
                'simulation': True, **advisor_info}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, payload, content_type='application/json; charset=utf-8'):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def route(self):
            return urlsplit(self.path).path

        def local_request(self):
            allowed_hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed_hosts:
                raise PermissionError('Use the localhost Recovery Lab address')

        def data(self):
            self.local_request()
            # A remote web page cannot mutate the local lab through browser fetch.
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                raise PermissionError('Cross-origin mutation is disabled')
            if self.headers.get('Transfer-Encoding'):
                raise ValueError('Transfer-Encoding is unsupported')
            size = int(self.headers.get('Content-Length', '0'))
            if size < 1 or size > MAX_BODY:
                raise ValueError(f'JSON body must be between 1 and {MAX_BODY} bytes')
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError('Payload must be a JSON object')
            return data

        def do_GET(self):
            try:
                self.local_request()
            except PermissionError as exc:
                self.respond(403, {'error': str(exc)})
                return
            path = self.route()
            with lock:
                if path in ASSETS:
                    filename, mime = ASSETS[path]
                    self.respond(200, (PACKAGE / 'web' / filename).read_bytes(), mime)
                elif path == '/health':
                    self.respond(200 if legacy.healthy() else 503, {'healthy': legacy.healthy(), 'tick': legacy.tick})
                elif path == '/request':
                    if legacy.done:
                        self.respond(409, {'error': 'Reset the sandbox episode first'})
                    else:
                        legacy.step(Action.WAIT)
                        self.respond(200 if legacy.healthy() else 503, {'healthy': legacy.healthy(), 'tick': legacy.tick})
                elif path == '/logs':
                    self.respond(200, {'logs': legacy.logs()})
                elif path == '/api/meta':
                    self.respond(200, {'simulation': True, **advisor_info,
                        'faults': ['credentials', 'connections', 'dependency'],
                        'wordings': ['train', 'validation', 'test'], 'actions': list(ACTIONS),
                        'policies': [{'id': name, 'available': name == 'checklist' or _checkpoint(output, name) is not None}
                                     for name in POLICIES], 'max_sessions': MAX_SESSIONS})
                elif path == '/api/report':
                    report = output / 'report.json'
                    try:
                        payload = json.loads(report.read_text())
                        if not isinstance(payload, dict):
                            raise ValueError('Report root must be an object')
                        self.respond(200, payload)
                    except FileNotFoundError:
                        self.respond(404, {'error': 'No benchmark report yet. Run the v2 benchmark to populate this view.'})
                    except (ValueError, OSError):
                        self.respond(409, {'error': 'The configured benchmark report is unavailable or invalid'})
                elif re.fullmatch(r'/api/incidents/[a-f0-9]{32}', path):
                    identifier = path.rsplit('/', 1)[-1]
                    session = sessions.get(identifier)
                    self.respond(200, snapshot(identifier, session)) if session else self.respond(404, {'error': 'Unknown incident'})
                else:
                    self.respond(404, {'error': 'Unknown route'})

        def do_POST(self):
            nonlocal legacy
            try:
                data = self.data()
                path = self.route()
                with lock:
                    if path == '/reset':
                        legacy = RecoveryEnv(Scenario(tuple(data.get('faults', ())), data.get('wording', 'train')))
                        self.respond(200, {'healthy': legacy.healthy()})
                    elif path == '/action':
                        action = Action[data['action'].upper()]
                        observation, reward, done = legacy.step(action)
                        self.respond(200, {'observation': observation, 'reward': reward, 'done': done, 'success': legacy.success})
                    elif path == '/api/incidents':
                        from .scenarios import Incident
                        from .sandbox import IncidentEnv
                        if len(sessions) >= MAX_SESSIONS:
                            self.respond(409, {'error': 'Session limit reached. Restart the lab to clear sessions.'})
                            return
                        if set(data) - {'faults', 'wording', 'seed'}:
                            raise ValueError('Only faults, wording and seed can configure an incident')
                        faults = data.get('faults', ['credentials', 'connections'])
                        if not isinstance(faults, list) or any(not isinstance(fault, str) for fault in faults):
                            raise ValueError('faults must be an array of named causes')
                        seed = data.get('seed', 11)
                        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
                            raise ValueError('seed must be an integer from 0 to 2147483647')
                        wording = data.get('wording', 'test')
                        if not isinstance(wording, str):
                            raise ValueError('wording must be a named split')
                        identifier = uuid.uuid4().hex
                        incident = Incident(id=identifier, faults=tuple(faults), wording=wording, seed=seed)
                        sessions[identifier] = {'env': IncidentEnv(incident, advisor=advisor),
                                                'decisions': []}
                        self.respond(201, snapshot(identifier, sessions[identifier]))
                    elif re.fullmatch(r'/api/incidents/[a-f0-9]{32}/(?:step|run)', path):
                        identifier, operation = path.split('/')[-2:]
                        session = sessions.get(identifier)
                        if session is None:
                            self.respond(404, {'error': 'Unknown incident'})
                            return
                        env = session['env']
                        if env.done:
                            self.respond(409, {'error': 'Incident has terminated; inject a new incident'})
                            return
                        if operation == 'step':
                            if set(data) != {'action'} or not isinstance(data['action'], str) or data['action'] not in ACTIONS:
                                raise ValueError('action must be one of the supported tools')
                            if data['action'] == 'jev' and env.observation()['evidence']['logs'] is None:
                                raise ValueError('Collect logs before consulting the advisor')
                            env.step(data['action'])
                        else:
                            from .agent import RecoveryAgent, ChecklistPolicy
                            from .learning import LinearQPolicy
                            name = data.get('policy', 'checklist')
                            if set(data) - {'policy'} or not isinstance(name, str) or name not in POLICIES:
                                raise ValueError('Unknown policy; choose checklist, rl_without_jev or rl_with_jev')
                            if name == 'checklist':
                                policy = ChecklistPolicy()
                            else:
                                checkpoint = _checkpoint(output, name)
                                if checkpoint is None:
                                    self.respond(409, {'error': 'No trained checkpoint for this policy. Run the v2 benchmark first.'})
                                    return
                                policy = LinearQPolicy.load(checkpoint)
                            agent = RecoveryAgent(policy)
                            while not env.done:
                                decision = agent.decide(env.observation())
                                record = asdict(decision) if is_dataclass(decision) else dict(decision)
                                record['trace_index'] = len(env.trace)
                                session['decisions'].append(record)
                                env.step(decision.action)
                        self.respond(200, snapshot(identifier, session))
                    else:
                        self.respond(404, {'error': 'Unknown route'})
            except PermissionError as exc:
                self.respond(403, {'error': str(exc)})
            except (ValueError, KeyError, RuntimeError, TypeError) as exc:
                self.respond(400, {'error': str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(port=8080, output=None, *, advisor_mode='offline', cache_path=None):
    server = create_server(port=port, output=output, advisor_mode=advisor_mode, cache_path=cache_path)
    print(f'Recovery Lab: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
