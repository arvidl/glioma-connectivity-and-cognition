"""Test a running Binder image through Jupyter's REST and kernel WebSocket APIs.

Needs websocket-client and BINDER_TEST_TOKEN. Runtime requests are real public
downloads; only their URLs are observed to check that the launcher is on demand.
"""
import argparse
import ast
import datetime
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


PROBE = '''import json as _json, os as _os, sys as _sys
import urllib.request as _request
from pathlib import Path as _Path
import nbformat as _nbformat
from pat16 import ROOT as _root
from pat16.notebook_assets import catalog as _catalog
from pat16.release_assets import verify as _verify
assert _sys.version.split()[0] == (_root/'.python-version').read_text().strip()
assert _Path(_sys.prefix) == _root/'.venv', _sys.prefix
_widget_path = _root/'.venv/share/jupyter'
assert str(_widget_path) in _os.environ['JUPYTER_PATH'].split(_os.pathsep)
assert (_widget_path/'labextensions/@jupyter-widgets/jupyterlab-manager/package.json').is_file()
_profiles = _catalog()['notebooks']
_members = {m['path'] for p in _profiles.values() for b in p['bundles'] for m in b['members']}
def _snapshot():
    return {p for p in _members if (_root/p).exists()}
_before = _snapshot()
assert not (_before - set(_tracked_files)), 'Image already contains untracked runtime assets'
_requests = []
_real_download = _request.urlretrieve
def _observed_download(url, *args, **kwargs):
    _requests.append(url)
    return _real_download(url, *args, **kwargs)
_request.urlretrieve = _observed_download
try:
    _welcome = _nbformat.read(_root/'binder/launch.ipynb', as_version=4)
    for _cell in _welcome.cells:
        if _cell.cell_type == 'code':
            exec(compile(_cell.source, 'binder/launch.ipynb', 'exec'))
    assert selected.value == '04' and not button.disabled
    assert not _requests and _snapshot() == _before, 'Launcher initialization fetched runtime data'
    _expected_requests = []
    for _bundle in _profiles['04']['bundles']:
        try:
            _verify(_root, _bundle)
        except ValueError:
            _expected_requests.extend(p['url'] for p in _bundle['parts'])
    assert _expected_requests, 'No missing notebook-04 pack: cannot test an on-demand download'
    download(None)
    assert not button.disabled
    assert _requests == _expected_requests, (_requests, _expected_requests)
    for _bundle in _profiles['04']['bundles']:
        _verify(_root, _bundle)
    _selected_members = {m['path'] for b in _profiles['04']['bundles'] for m in b['members']}
    assert not ((_snapshot() - _before) - _selected_members), 'Unselected runtime was restored'
    download(None)
    assert not button.disabled and _requests == _expected_requests, 'Repeat request downloaded again'
    _path = _root/'notebooks'/_profiles['04']['notebook']
    _cells = 0
    for _cell in _nbformat.read(_path, as_version=4).cells:
        if _cell.cell_type == 'code':
            exec(compile(_cell.source, str(_path), 'exec'))
            _cells += 1
    _binder_result = dict(python=_sys.version.split()[0], interpreter=_sys.executable,
        widget_extension='present in server JUPYTER_PATH',
        launcher_initialization='passed: no downloads or restored files',
        selected_download='passed: only notebook 04; archive/member SHA256 verified',
        downloaded_urls=_requests, repeat_request='passed: no additional downloads',
        notebook='04', code_cells=_cells, errors=0)
finally:
    _request.urlretrieve = _real_download
'''


def run_code(ws, code, timeout=600):
    """Wait for both the matching execution reply and idle; propagate cell errors."""
    import websocket
    mid = uuid.uuid4().hex
    message = dict(header=dict(msg_id=mid, username='binder-validation',
        session=uuid.uuid4().hex, msg_type='execute_request', version='5.3',
        date=datetime.datetime.now(datetime.timezone.utc).isoformat()),
        parent_header={}, metadata={}, content=dict(code=code, silent=False,
        store_history=False, allow_stdin=False, stop_on_error=True,
        user_expressions={'result': '_json.dumps(_binder_result)'}))
    ws.send(json.dumps(message))
    deadline = time.monotonic() + timeout
    reply = None
    idle = False
    while not (reply is not None and idle):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Kernel execution did not finish before the deadline')
        ws.settimeout(min(30, remaining))
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if not raw:
            raise RuntimeError('Kernel WebSocket closed before execution completed')
        if isinstance(raw, bytes):
            continue  # widget binary buffers are not execution replies
        event = json.loads(raw)
        if event.get('parent_header', {}).get('msg_id') != mid:
            continue
        kind = event.get('msg_type', event.get('header', {}).get('msg_type'))
        content = event['content']
        if kind == 'stream':
            print(content['text'], end='', flush=True)
        elif kind == 'error' or (kind == 'execute_reply' and content['status'] != 'ok'):
            raise RuntimeError(f"Kernel failed: {content.get('ename')}: {content.get('evalue')}")
        elif kind == 'execute_reply':
            reply = content
        elif kind == 'status' and content['execution_state'] == 'idle':
            idle = True
    result = reply['user_expressions']['result']
    if result['status'] != 'ok':
        raise RuntimeError(f'Kernel result expression failed: {result}')
    return json.loads(ast.literal_eval(result['data']['text/plain']))


def wait_for_server(api, timeout=120, pause=2):
    """Docker's published port can reset connections until Jupyter is ready."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return api('api/kernelspecs')
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (502, 503, 504):
                raise  # authentication/path errors are not startup readiness
            if time.monotonic() >= deadline:
                raise
            time.sleep(pause)


def main():
    import websocket
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8888/')
    parser.add_argument('--tracked-files', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    token = os.environ['BINDER_TEST_TOKEN']
    if not token:
        raise ValueError('BINDER_TEST_TOKEN must not be empty')
    base = args.url.rstrip('/') + '/'
    headers = {'Authorization': 'token ' + token, 'Content-Type': 'application/json'}

    def api(path, method='GET', data=None):
        body = json.dumps(data).encode() if data is not None else None
        request = urllib.request.Request(base + path, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    report = dict(status='failed', interface='Jupyter REST and kernel WebSocket APIs')
    kernel = None
    ws = None
    try:
        specs = wait_for_server(api)
        spec = specs['kernelspecs']['python3']['spec']
        if not Path(spec['argv'][0]).is_absolute() or Path(spec['argv'][0]).parent.parent.name != '.venv':
            raise AssertionError(f'Generic or unexpected python3 kernel: {spec}')
        report['kernel_spec'] = spec
        print('Server selected:', spec['display_name'], spec['argv'][0], flush=True)
        kernel = api('api/kernels', 'POST', {'name': 'python3'})
        parts = urllib.parse.urlsplit(base)
        wsurl = urllib.parse.urlunsplit(('wss' if parts.scheme == 'https' else 'ws',
            parts.netloc, parts.path + 'api/kernels/' + kernel['id'] + '/channels',
            'session_id=' + uuid.uuid4().hex, ''))
        ws = websocket.create_connection(wsurl, header=['Authorization: token ' + token],
            origin=parts.scheme + '://' + parts.netloc, timeout=30)
        tracked = json.loads(args.tracked_files.read_text())
        code = '_tracked_files = ' + repr(tracked) + '\n' + PROBE
        report.update(run_code(ws, code), status='passed')
        print('PASS: image kernel, launcher, on-demand download, reuse and notebook 04', flush=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            if ws is not None:
                ws.close()
            if kernel is not None:
                api('api/kernels/' + kernel['id'], 'DELETE')
        except Exception as error:
            report.update(status='failed', cleanup_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
