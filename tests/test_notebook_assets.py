"""On-demand packs preserve provenance, local edits and a download-free launcher."""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from pat16 import ROOT
from pat16.notebook_assets import ensure_notebook
from pat16.release_assets import bundle


@pytest.fixture
def pack(tmp_path):
    source = tmp_path/'source'; (source/'data').mkdir(parents=True)
    (source/'data/example.txt').write_text('Original test fixture\n')
    assets = tmp_path/'assets'
    spec = bundle(source, 'data', assets, 'example.tar.gz')
    for part in spec['parts']:
        part['url'] = (assets/part['name']).as_uri()
    root = tmp_path/'clone'; (root/'reproducibility').mkdir(parents=True)
    manifest = dict(notebooks={'04': dict(notebook='04_example.ipynb', bundles=[spec])})
    (root/'reproducibility/notebook-assets.json').write_text(json.dumps(manifest))
    return root, assets, spec


def test_verified_download_reuse_cleanup_and_edit_protection(pack, monkeypatch):
    root, assets, spec = pack
    requests = []
    def download(url, target):
        requests.append(url); shutil.copy(assets/spec['parts'][0]['name'], target)
    monkeypatch.setattr('urllib.request.urlretrieve', download)
    ensure_notebook('04', root=root)
    ensure_notebook('04', root=root)
    assert len(requests) == 1
    assert not list((root/'.cache/notebook-assets').iterdir())
    (root/'data/example.txt').write_text('Local edit')
    with pytest.raises(ValueError, match='Modified destination'):
        ensure_notebook('04', root=root)
    assert (root/'data/example.txt').read_text() == 'Local edit'
    assert len(requests) == 1


def test_bad_download_never_extracts_and_cleans_temporary_files(pack, monkeypatch):
    root, _, _ = pack
    monkeypatch.setattr('urllib.request.urlretrieve', lambda url, target: Path(target).write_bytes(b'bad archive'))
    with pytest.raises(ValueError, match='checksum/size'):
        ensure_notebook('04', root=root)
    assert not (root/'data').exists()
    assert not list((root/'.cache/notebook-assets').iterdir())


def test_notebook_packs_are_exact_subsets_of_original_release():
    original = json.loads((ROOT/'reproducibility/assets.json').read_text())
    known = {m['path']: m for b in original['bundles'].values() for m in b['members']}
    manifest = json.loads((ROOT/'reproducibility/notebook-assets.json').read_text())
    assert manifest['source_release'] == original['release']
    assert {p['notebook'] for p in manifest['notebooks'].values()} == {p.name for p in (ROOT/'notebooks').glob('*.ipynb')}
    for profile in manifest['notebooks'].values():
        for spec in profile['bundles']:
            assert all(member == known[member['path']] for member in spec['members'])


def test_launcher_initialization_never_downloads(monkeypatch):
    import nbformat
    def unexpected(*args, **kwargs):
        raise AssertionError('Launcher attempted to download before a button click')
    monkeypatch.setattr('pat16.notebook_assets.ensure_notebook', unexpected)
    monkeypatch.setattr('urllib.request.urlretrieve', unexpected)
    scope = {}
    for cell in nbformat.read(ROOT/'binder/launch.ipynb', as_version=4).cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source, 'binder/launch.ipynb', 'exec'), scope)
    assert scope['selected'].value == '04'
    assert not scope['button'].disabled


def test_binder_hooks_override_generic_kernel_with_locked_interpreter(tmp_path):
    """Reproduce the generic-kernel shadowing seen in the public Binder test.

    Dependencies are already installed by pytest's environment; only uv sync is
    skipped. Run the actual hooks, real ipykernel installer and spec discovery.
    """
    bin_dir = tmp_path/'bin'; bin_dir.mkdir()
    uv = bin_dir/'uv'; uv.write_text('#!/bin/sh\nexit 0\n'); uv.chmod(0o755)
    venv_bin = tmp_path/'.venv/bin'; venv_bin.mkdir(parents=True)
    python = venv_bin/'python'
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    generic = tmp_path/'.venv/share/jupyter/kernels/python3'
    generic.mkdir(parents=True)
    (generic/'kernel.json').write_text(json.dumps(dict(argv=['python', '-m', 'ipykernel_launcher', '-f', '{connection_file}'], display_name='Generic Python', language='python')))
    env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],
               JUPYTER_DATA_DIR=str(tmp_path/'user-jupyter'),
               JUPYTER_PATH='', IPYTHONDIR=str(tmp_path/'ipython'))
    subprocess.run(['bash', str(ROOT/'binder/postBuild')], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    probe = 'import json; from jupyter_client.kernelspec import KernelSpecManager; s=KernelSpecManager().get_kernel_spec("python3"); print(json.dumps(dict(argv=s.argv, name=s.display_name, directory=s.resource_dir)))'
    result = subprocess.run(['bash', str(ROOT/'binder/start'), sys.executable, '-c', probe], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    spec = json.loads(result.stdout)
    assert spec['argv'][0] == sys.executable
    assert spec['name'] == 'PAT16 (uv, Python 3.12.14)'
    assert Path(spec['directory']) == generic


@pytest.fixture
def image_probe():
    import importlib.util
    spec = importlib.util.spec_from_file_location('binder_image_probe', ROOT/'scripts/validate_binder_image.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelSocket:
    """Minimal protocol peer: keep errors and ordering independent of real data."""
    def __init__(self, failure=None):
        self.failure = failure

    def send(self, raw):
        mid = json.loads(raw)['header']['msg_id']
        def event(kind, content, parent=mid):
            return json.dumps(dict(msg_type=kind, parent_header={'msg_id': parent}, content=content))
        if self.failure:
            self.events = iter([event(self.failure, dict(status='error', ename='AssertionError', evalue='download check failed'))])
        else:
            self.events = iter([
                event('error', dict(ename='OldError', evalue='unrelated'), parent='other-execution'),
                b'widget buffer',
                event('status', dict(execution_state='idle')),
                event('execute_reply', dict(status='ok', user_expressions=dict(result=dict(
                    status='ok', data={'text/plain': repr(json.dumps({'verified': True}))})))),
            ])

    def settimeout(self, value):
        pass

    def recv(self):
        return next(self.events, '')


def test_image_probe_waits_for_reply_and_idle_and_ignores_other_messages(image_probe):
    assert image_probe.run_code(KernelSocket(), 'probe') == {'verified': True}


@pytest.mark.parametrize('kind', ['error', 'execute_reply'])
def test_image_probe_propagates_execution_errors(image_probe, kind):
    with pytest.raises(RuntimeError, match='AssertionError.*download check failed'):
        image_probe.run_code(KernelSocket(failure=kind), 'probe')


def test_image_probe_retries_container_startup_connection_reset(image_probe, monkeypatch):
    calls = []
    monkeypatch.setattr(image_probe.time, 'sleep', lambda _: None)
    def api(path):
        calls.append(path)
        if len(calls) == 1:
            raise ConnectionResetError('Jupyter has not started yet')
        return {'kernelspecs': {}}
    assert image_probe.wait_for_server(api) == {'kernelspecs': {}}
    assert calls == ['api/kernelspecs', 'api/kernelspecs']


def test_image_probe_does_not_retry_authentication_errors(image_probe):
    from urllib.error import HTTPError
    def api(_):
        raise HTTPError('http://localhost/api/kernelspecs', 401, 'Unauthorized', {}, None)
    with pytest.raises(HTTPError, match='401'):
        image_probe.wait_for_server(api)
