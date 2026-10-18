"""Record inputs, code, configurations, versions and SHA256 hashes; verify later."""
import argparse,hashlib,json,platform,sys
from importlib.metadata import version
from pathlib import Path
from pat16 import ROOT


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();target=ROOT/'data/manifest.json'
if a.verify:
    manifest=json.loads(target.read_text());bad=[r['path'] for r in manifest['files'] if not (ROOT/r['path']).exists() or digest(ROOT/r['path'])!=r['sha256']]
    if bad:raise SystemExit('Hash mismatch: '+', '.join(bad))
    print('All',len(manifest['files']),'bundled input files match SHA256.')
else:
    paths=sorted(p for p in (ROOT/'data').rglob('*') if p.is_file() and p!=target and p.name!='.DS_Store')
    obj={'subject':'sub-PAT16','analysis_date':'2026-10-18','sources':{'preop':'https://openneuro.org/datasets/ds001226/versions/5.0.0','postop':'https://openneuro.org/datasets/ds002080/versions/4.0.0','local_release':'glioma-brain-digital-twin-public-2026/data/BIDS_preop_postop; complete copied subset','atlas':'ThomasYeoLab/CBIG tag v0.14.3-Update_Yeo2011_Schaefer2018_labelname; Schaefer100/7-network FSLMNI152 2mm','template':'FSL MNI152 2mm local distribution; see data/atlas/NOTICE.md'},'python':sys.version,'platform':platform.platform(),'packages':{x:version(x) for x in ['numpy','scipy','pandas','nibabel','dipy','SimpleITK','nilearn','networkx','matplotlib','plotly','nbclient']},'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in paths]}
    target.write_text(json.dumps(obj,indent=2));print('Recorded',len(paths),'files')
