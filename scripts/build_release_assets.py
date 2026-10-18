"""Prepare local assets. This script never publishes or uploads anything."""
import argparse,json
from pathlib import Path
from pat16 import ROOT
from pat16.release_assets import bundle
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'dist/assets');a=p.parse_args()
manifest=dict(schema=1,repository='https://github.com/arvidl/glioma-connectivity-and-cognition',release='v0.1.0-assets',published=False,bundles={})
for name,prefix in [('inputs','data'),('analysis','results')]:
    print('Bundling',prefix,flush=True)
    spec=bundle(ROOT,prefix,a.output,'pat16-'+name+'.tar.gz')
    for part in spec['parts']:part['url']=manifest['repository']+'/releases/download/'+manifest['release']+'/'+part['name']
    manifest['bundles'][name]=spec
(ROOT/'reproducibility/assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Local assets and manifest prepared; published=false')
