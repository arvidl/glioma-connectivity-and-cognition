"""Restore verified runtime assets. Runs with plain Python before uv installation."""
import argparse,json,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from pat16.release_assets import restore,verify,digest
p=argparse.ArgumentParser();p.add_argument('--assets-dir',type=Path);p.add_argument('--profile',choices=['inputs','all'],default='all');p.add_argument('--verify-only',action='store_true');p.add_argument('--overwrite',action='store_true');a=p.parse_args()
m=json.loads((ROOT/'reproducibility/assets.json').read_text());names=['inputs'] if a.profile=='inputs' else ['inputs','analysis']
for name in names:
    spec=m['bundles'][name]
    if a.verify_only:verify(ROOT,spec);print('Verified',name);continue
    directory=a.assets_dir
    if directory is None:
        if not m['published']:p.error('Assets are prepared locally, not published. Supply --assets-dir /path/to/dist/assets (see docs/REPRODUCING.md).')
        directory=ROOT/'.cache/assets';directory.mkdir(parents=True,exist_ok=True)
        for part in spec['parts']:
            dest=directory/part['name']
            if dest.exists() and digest(dest)==part['sha256']:continue
            temp=dest.with_suffix('.download');urllib.request.urlretrieve(part['url'],temp)
            if temp.stat().st_size!=part['bytes'] or digest(temp)!=part['sha256']:temp.unlink();raise ValueError('Download checksum mismatch')
            temp.replace(dest)
    restore(ROOT,spec,directory,a.overwrite);print('Restored and verified',name)
