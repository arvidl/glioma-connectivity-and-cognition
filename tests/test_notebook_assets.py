"""On-demand packs preserve provenance, local edits and a download-free launcher."""
import json
import shutil
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
