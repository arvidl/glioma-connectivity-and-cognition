"""Package per-notebook subsets of the unchanged v0.1.0 runtime snapshot.

Reads explicit lists plus per-path-component fnmatch patterns from binder/notebook-profiles.json.
Never uploads; every selected member must match the original release checksum.
"""
import argparse
import fnmatch
import gzip
import json
import tarfile
from pathlib import Path
from pat16 import ROOT
from pat16.release_assets import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/notebook-assets')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT/'reproducibility/assets.json').read_text())
    known = {m['path']: m for b in original['bundles'].values() for m in b['members']}
    profiles = json.loads((ROOT/'binder/notebook-profiles.json').read_text())
    manifest = dict(schema=1, repository=original['repository'], release='v0.1.1-notebooks',
                    source_release=original['release'], notebooks={})
    manifest_path = ROOT/'reproducibility/notebook-assets.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'notebooks': {}}
    for key, profile in profiles.items():
        selected = set(profile['files'])
        for pattern in profile['extra_globs']:
            segments = pattern.split('/')
            matches = {p for p in known if len(p.split('/')) == len(segments)
                       and all(fnmatch.fnmatchcase(a, b) for a, b in zip(p.split('/'), segments))}
            if not matches:
                raise ValueError(f'Empty pattern: {key}: {pattern}')
            selected |= matches
        bundles = []
        for prefix in ['data', 'results']:
            members = [known[p] for p in sorted(selected) if p.startswith(prefix+'/')]
            if not members:
                continue
            name = f'pat16-notebook-{key}-{prefix}.tar.gz'
            path = args.output/name
            cached = next((b for b in previous['notebooks'].get(key, {}).get('bundles', [])
                           if b['prefix'] == prefix and b['members'] == members), None)
            if cached and path.is_file() and digest(path) == cached['parts'][0]['sha256']:
                bundles.append(cached)
                continue
            with path.open('wb') as f, gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0, compresslevel=6) as gz:
                with tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                    for member in members:
                        source = ROOT/member['path']
                        if source.is_symlink() or source.stat().st_size != member['bytes'] or digest(source) != member['sha256']:
                            raise ValueError(f"Changed source snapshot: {member['path']}")
                        info = tarfile.TarInfo(member['path'])
                        info.size = member['bytes']; info.mode = 0o644; info.mtime = 0
                        with source.open('rb') as data:
                            tar.addfile(info, data)
            part = dict(name=name, bytes=path.stat().st_size, sha256=digest(path),
                        url=f"{manifest['repository']}/releases/download/{manifest['release']}/{name}")
            bundles.append(dict(prefix=prefix, members=members, parts=[part]))
        manifest['notebooks'][key] = dict(notebook=profile['notebook'], bundles=bundles)
        print(key, sum(p['bytes'] for b in bundles for p in b['parts'])/2**20, 'MiB', flush=True)
    (ROOT/'reproducibility/notebook-assets.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
