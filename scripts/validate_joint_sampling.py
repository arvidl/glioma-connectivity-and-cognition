"""Audit the final joint study, preserved evidence, executed notebooks and PDF."""
import hashlib,json,re
import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader
from pat16 import ROOT,RESULTS
from pat16.bold import sha
from pat16.joint_sampling import OUT,VERSION,B,F,lineage

archive=RESULTS/'before_joint_sampling'
m=json.loads((archive/'manifest.json').read_text())
for r in m['files']:
    p=ROOT/r['path'] if r['immutable'] else archive/r['path']
    assert sha(p)==r['sha256'],p
clinical='\\section{Complete supplied assessments}'+(ROOT/'report/main.tex').read_text().split('\\section{Complete supplied assessments}')[1]
assert hashlib.sha256(clinical.encode()).hexdigest()==m['clinical_appendix_source_sha256']
for name in ['cantab_all.tex','psychosocial_all.tex','cantab_all_dictionary.tex','psychosocial_all_dictionary.tex']:
    assert sha(ROOT/'report/generated'/name)==sha(archive/'report/generated'/name)
d=json.loads((OUT/'design.json').read_text())
assert d['version']==VERSION and d['inputs']==lineage()
for p,h in d['outputs'].items():assert sha(OUT/p)==h,p
execution=json.loads((RESULTS/'notebook_execution.json').read_text())
assert len(execution)==len(list((ROOT/'notebooks').glob('*.ipynb')))>=11 and all(r['errors']==0 for r in execution)
cells=0
for path in (ROOT/'notebooks').glob('*.ipynb'):
    nb=nbformat.read(path,as_version=4)
    assert 'Current SC policy' in nb.cells[1].source
    assert 'Notebook 10' in nb.cells[1].source
    for c in nb.cells:
        if c.cell_type!='code':continue
        cells+=1;assert c.execution_count is not None,path
        assert not any(o.output_type=='error' for o in c.outputs),path
tests=(RESULTS/'joint_sampling_tests.log').read_text()
match=re.search(r'(\d+) passed',tests);assert match and 'FAILED' not in tests
summary=pd.read_csv(OUT/'summary.csv')
joint=summary[summary.source=='joint']
assert len(joint)==24 and len(summary)==72
for r in joint.itertuples():
    z=np.load(OUT/f'{r.protocol}_{r.support}.npz')
    assert z['delta'].shape==(B,F)
    assert np.allclose(z['delta'],z['rho'][1]-z['rho'][0])
    assert np.allclose(np.quantile(z['delta'],[.025,.5,.975]),[r.q025,r.median,r.q975])
    assert np.isclose(z['delta'].var(),r.SC_variance+r.FC_variance+r.interaction_variance)
old=pd.read_csv(RESULTS/'primary_sc/matched_coupling.csv')
diff=[]
for protocol in d['protocols'][:6]:
    z=np.load(OUT/f'{protocol}_fixed.npz')
    v=old[old.protocol==protocol].sort_values('replicate')
    diff.extend(abs(z['fc_only_rho'][1]-z['fc_only_rho'][0]-v.coupling_delta))
assert len(diff)==1200 and max(diff)<1e-12
qc=pd.read_csv(OUT/'fc_qc.csv')
matched=qc[qc.nuisance=='fully_refit']
assert (matched.retained==68).all() and (matched.unique_frames==68).all()
assert (matched.design_rank==39).all() and (matched.residual_df==29).all()
assert qc[qc.nuisance=='frozen_fit'].residual_df.isna().all()
pdf=PdfReader(ROOT/'report/main.pdf')
text='\n'.join(p.extract_text() for p in pdf.pages)
assert 'Joint SC and FC sampling results' in text and 'Joint tractography and FC sampling design' in text
log=(ROOT/'report/main.log').read_text()
assert not any(x in log for x in ['Overfull','Undefined control','undefined references','Citation `'])
pipeline=(RESULTS/'joint_sampling_pipeline.log').read_text()
assert 'Completed in ' in pipeline and 'Traceback' not in pipeline
v=joint[(joint.protocol=='primary_block10')&(joint.support=='fixed')].iloc[0]
obj=dict(version=VERSION,SC_draws=B,FC_draws=F,protocols=len(d['protocols']),support_modes=2,
    crossed_grids=len(joint),dependent_cells_per_grid=B*F,tests_passed=int(match[1]),
    notebooks=len(execution),executed_code_cells=cells,
    archived_files=sum(not r['immutable'] for r in m['files']),
    unchanged_derived_inputs=sum(r['immutable'] for r in m['files']),
    clinical_appendix_source_and_four_tables_unchanged=True,
    original_primary_point_estimates_and_all_image_SC_FC_inputs_unchanged=True,
    fixed_SC_matched_replays=1200,max_replay_absolute_difference=max(diff),
    primary_joint_fixed_median=float(v['median']),primary_joint_fixed_q025=float(v.q025),
    primary_joint_fixed_q975=float(v.q975),primary_joint_fixed_FC_variance_share=float(v.FC_share),
    PDF_pages=len(pdf.pages),PDF_sha256=sha(ROOT/'report/main.pdf'),
    PDF_render_review='All pages in tmp/pdfs/joint_sampling; new methods/results pages additionally reviewed at higher resolution',
    interpretation='Conditional computational ranges; full-run blocks freeze nuisance training; matched subsets fully refit; no clinical confidence intervals')
(RESULTS/'joint_sampling_validation.json').write_text(json.dumps(obj,indent=2))
print(json.dumps(obj,indent=2))
