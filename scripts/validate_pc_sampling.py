"""Audit the PC-training controls, preserved clinical evidence and final artifacts."""
import hashlib,json,re
import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader
from pat16 import ROOT,RESULTS
from pat16.bold import sha
from pat16.pc_sampling import OUT,VERSION,inputs

archive=RESULTS/'before_pc_sampling';manifest=json.loads((archive/'manifest.json').read_text())
for r in manifest['files']:
    p=ROOT/r['path'] if r['immutable'] else archive/r['path']
    assert sha(p)==r['sha256'],p
source='\\section{Complete supplied assessments}'+(ROOT/'report/main.tex').read_text().split('\\section{Complete supplied assessments}')[1]
assert hashlib.sha256(source.encode()).hexdigest()==manifest['clinical_appendix_source_sha256']
for name in ['cantab_all.tex','psychosocial_all.tex','cantab_all_dictionary.tex','psychosocial_all_dictionary.tex']:
    assert sha(ROOT/'report/generated'/name)==sha(archive/'report/generated'/name)
d=json.loads((OUT/'design.json').read_text());assert d['version']==VERSION and d['inputs']==inputs()
for p,h in d['outputs'].items():assert sha(OUT/p)==h,p
s=pd.read_csv(OUT/'summary.csv');f=pd.read_csv(OUT/'feasibility.csv');q=pd.read_csv(OUT/'fc_metrics.csv')
assert len(s)==39 and len(f)==12 and len(q)==14400
assert (f.paired_eligible==200).all() and q.eligible.all()
assert (q[q['mode']!='frozen'].design_rank==39).all()
for r in s.itertuples():
    z=np.load(OUT/f'{r.protocol}_{r.support}_{r.mode}.npz')
    assert z['delta'].shape==(500,200)
    assert np.allclose(z['delta'],z['rho'][1]-z['rho'][0])
    assert np.allclose(np.quantile(z['delta'],[.025,.5,.975]),[r.q025,r.median,r.q975])
assert d['bootstrap_max_absolute_replay_difference']<1e-10
mf=np.load(OUT/'mechanism_fc.npy',mmap_mode='r');temporal_error=float(np.max(np.abs(mf[0]-mf[1])))
assert temporal_error<1e-10
clinical=pd.read_csv(RESULTS/'assessments_all.csv');ledger=pd.read_csv(OUT/'clinical_field_links.csv')
pd.testing.assert_frame_equal(ledger[clinical.columns],clinical)
assert len(ledger)==47 and (ledger.group=='CANTAB').sum()==28
assert ledger[ledger.group!='CANTAB'].candidate_networks.isna().all()
execution=json.loads((RESULTS/'notebook_execution.json').read_text())
assert len(execution)==len(list((ROOT/'notebooks').glob('*.ipynb')))==14
assert all(r['errors']==0 for r in execution)
cells=0
for path in (ROOT/'notebooks').glob('*.ipynb'):
    nb=nbformat.read(path,as_version=4)
    for c in nb.cells:
        if c.cell_type!='code':continue
        cells+=1;assert c.execution_count is not None,path
        assert not any(o.output_type=='error' for o in c.outputs),path
log=(ROOT/'report/main.log').read_text()
assert not any(x in log for x in ['Overfull','Undefined control','undefined references','Citation `','multiply defined'])
assert (ROOT/'tmp/pdfs/pc_sampling/contact-1.png').exists()
assert sha(ROOT/'report/main.pdf')==(RESULTS/'pc_sampling_render_pdf_sha256.txt').read_text().strip()
pdf=PdfReader(ROOT/'report/main.pdf');text='\n'.join(p.extract_text() for p in pdf.pages)
for phrase in ['Diagnosing PC-training shifts','Why PC refitting recenters','Clinical interpretation after sampling-design controls','All CANTAB measures','Psychosocial and contextual fields']:
    assert phrase in text,phrase
pipeline=(RESULTS/'pc_sampling_pipeline.log').read_text();assert 'Completed in ' in pipeline and 'Traceback' not in pipeline
log=(RESULTS/'pc_sampling_tests.log').read_text();tests=re.search(r'(\d+) passed',log);assert tests and 'FAILED' not in log
v=s[(s.protocol=='block_unique_block10')&(s.support=='fixed')&(s['mode']=='full_refit')].iloc[0]
o=dict(version=VERSION,protocols=len(f),paired_draws_each=200,crossed_grids=len(s),dependent_cells_per_grid=100000,nuisance_FC_fits_or_frozen_metrics=len(q),mechanism_conditions=5,mechanism_session_draws_each=400,bootstrap_replay_error=d['bootstrap_max_absolute_replay_difference'],temporal_only_FC_error=temporal_error,matched_count_full_refit=dict(median=v['median'],q025=v.q025,q975=v.q975),tests_passed=int(tests[1]),notebooks=len(execution),executed_code_cells=cells,clinical_fields=len(ledger),cantab_fields=28,context_fields=19,archived_files=sum(not r['immutable'] for r in manifest['files']),unchanged_inputs_and_derived_files=sum(r['immutable'] for r in manifest['files']),clinical_appendix_source_and_four_tables_unchanged=True,PDF_pages=len(pdf.pages),PDF_sha256=sha(ROOT/'report/main.pdf'),PDF_render_review='All-page contacts and detailed new methods/results/clinical pages reviewed in tmp/pdfs/pc_sampling',interpretation='PC-loading retraining under multiplicities dominates empirical recentering; unit-coverage and nonreplacement controls retained; subset ranges depend on fraction; population bias and neural specificity unresolved; clinical evidence descriptive and unchanged.')
(RESULTS/'pc_sampling_validation.json').write_text(json.dumps(o,indent=2));print(json.dumps(o,indent=2))
