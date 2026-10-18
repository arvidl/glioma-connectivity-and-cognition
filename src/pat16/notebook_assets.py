"""On-demand, authentic per-notebook runtime subsets for Binder and local use."""
import argparse
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

from . import ROOT
from .release_assets import digest, restore, safe_path, verify


def catalog(root=ROOT):
    return json.loads((Path(root)/'reproducibility/notebook-assets.json').read_text())


def ensure_notebook(notebook='04', *, root=ROOT, assets_dir=None):
    """Download only missing selected packs, preserving modified local files.

    Compressed and extracted hashes are checked. Packs are removed after use;
    repeated calls verify existing files and make no network requests.
    """
    root = Path(root)
    manifest = catalog(root)
    key = str(notebook).zfill(2)
    if key not in manifest['notebooks']:
        raise ValueError(f'Unknown notebook {notebook!r}; choose 00–14')
    profile = manifest['notebooks'][key]
    for spec in profile['bundles']:
        try:
            verify(root, spec)
            print(f"Already verified notebook {key}: {spec['prefix']}", flush=True)
            continue
        except ValueError:
            pass
        # Fail before network work when a destination contains a local edit.
        for member in spec['members']:
            target = safe_path(root, member['path'], spec['prefix'])
            if target.exists() and (target.stat().st_size != member['bytes'] or
                                    digest(target) != member['sha256']):
                raise ValueError(f"Modified destination: {member['path']}; preserve your edit before restoring")
        if assets_dir is not None:
            restore(root, spec, Path(assets_dir))
        else:
            cache = root/'.cache/notebook-assets'
            cache.mkdir(parents=True, exist_ok=True)
            needed = sum(m['bytes'] for m in spec['members']) + sum(p['bytes'] for p in spec['parts'])
            if shutil.disk_usage(cache).free < needed:
                raise RuntimeError(f'Not enough free disk space: need up to {needed/2**20:.1f} MiB')
            with tempfile.TemporaryDirectory(prefix='download-', dir=cache) as directory:
                for part in spec['parts']:
                    if Path(part['name']).name != part['name']:
                        raise ValueError('Unsafe asset part name')
                    print(f"Downloading {part['name']} ({part['bytes']/2**20:.2f} MiB)…", flush=True)
                    path = Path(directory)/part['name']
                    urllib.request.urlretrieve(part['url'], path)
                    if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                        raise ValueError(f"Download checksum/size mismatch: {part['name']}")
                restore(root, spec, directory)
        verify(root, spec)
        print(f"Restored and verified notebook {key}: {spec['prefix']}", flush=True)
    return root/'notebooks'/profile['notebook']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', nargs='?', default='04')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--assets-dir', type=Path)
    args = parser.parse_args()
    if args.list:
        for key, profile in catalog()['notebooks'].items():
            size = sum(p['bytes'] for b in profile['bundles'] for p in b['parts'])
            print(f"{key}: {size/2**20:.2f} MiB — {profile['notebook']}")
    else:
        print('Ready:', ensure_notebook(args.notebook, assets_dir=args.assets_dir))


if __name__ == '__main__':
    main()
