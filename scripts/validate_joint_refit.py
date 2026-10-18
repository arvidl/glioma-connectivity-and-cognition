"""Final delivery audit: preserved data, paired estimates, execution and PDF."""
import hashlib,json,re
import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader
from pat16 import ROOT,RESULTS
from pat16.bold import sha
from pat16.joint_refit import OUT,VERSION,inputs
from pat16.evidence_integration import OUT as EVIDENCE

archive=RESULTS/'before_joint_refit';m=json.loads((archive/'manifest.json').read_text())
for r in m['files']:
    p=ROOT/r['path'] if r['immutable'] else archive/r['path']
    assert sha(p)==r['sha256'],p
source='\\section{Complete supplied assessments}'+(ROOT/'report/main.tex').read_text().split('\\section{Complete supplied assessments}')[1]
assert hashlib.sha256(source.encode()).hexdigest()==m['clinical_appendix_source_sha256']
for name in ['cantab_all.tex','psychosocial_all.tex','cantab_all_dictionary.tex','psychosocial_all_dictionary.tex']:
    assert sha(ROOT/'report/generated'/name)==sha(archive/'report/generated'/name)
d=json.loads((OUT/'design.json').read_text());assert d['version']==VERSION and d['inputs']==inputs()
for p,h in d['outputs'].items():assert sha(OUT/p)==h,p
e=json.loads((EVIDENCE/'design.json').read_text())
for p,h in e['inputs'].items():assert sha(ROOT/p)==h,p
for p,h in e['outputs'].items():assert sha(EVIDENCE/p)==h,p
s=pd.read_csv(OUT/'summary.csv');q=pd.read_csv(OUT/'fit_qc.csv');f=pd.read_csv(OUT/'feasibility.csv')
assert len(s)==18 and len(q)==4800 and len(f)==6
assert (f[f.protocol.str.startswith('primary')].paired_eligible==200).all()
assert (f[f.protocol.str.startswith('strict')].paired_eligible==0).all()
for r in s.itertuples():
    z=np.load(OUT/f'{r.protocol}_{r.support}_{r.mode}.npz')
    assert z['delta'].shape==(500,200)
    assert np.allclose(z['delta'],z['rho'][1]-z['rho'][0])
    assert np.allclose(np.quantile(z['delta'],[.025,.5,.975]),[r.q025,r.median,r.q975])
clinical=pd.read_csv(RESULTS/'assessments_all.csv');ledger=pd.read_csv(EVIDENCE/'assessment_evidence_ledger.csv')
pd.testing.assert_frame_equal(ledger[clinical.columns],clinical)
assert len(ledger)==47 and (ledger.group=='CANTAB').sum()==28
assert ledger[ledger.group!='CANTAB'].candidate_networks.isna().all()
execution=json.loads((RESULTS/'notebook_execution.json').read_text())
assert len(execution)==13 and all(r['errors']==0 for r in execution)
cells=0
for path in (ROOT/'notebooks').glob('*.ipynb'):
    nb=nbformat.read(path,as_version=4)
    for c in nb.cells:
        if c.cell_type!='code':continue
        cells+=1;assert c.execution_count is not None,path
        assert not any(o.output_type=='error' for o in c.outputs),path
log=(ROOT/'report/main.log').read_text()
assert not any(x in log for x in ['Overfull','Undefined control','undefined references','Citation `'])
assert (ROOT/'tmp/pdfs/joint_refit/contact-4.png').exists()
assert sha(ROOT/'report/main.pdf')==(RESULTS/'joint_refit_render_pdf_sha256.txt').read_text().strip()
pdf=PdfReader(ROOT/'report/main.pdf');text='\n'.join(p.extract_text() for p in pdf.pages)
for phrase in ['Full-run refits versus frozen nuisance training','Integrating uncertainty with CANTAB','All CANTAB measures','Psychosocial and contextual fields']:assert phrase in text,phrase
pipeline=(RESULTS/'joint_refit_pipeline.log').read_text();assert 'Completed in ' in pipeline and 'Traceback' not in pipeline
log=(RESULTS/'joint_refit_tests.log').read_text();tests=re.search(r'(\d+) passed',log);assert tests and 'FAILED' not in log
v=s[(s.protocol=='primary_block10')&(s.support=='fixed')&(s['mode']=='full_refit')].iloc[0]
o=dict(version=VERSION,attempted_refits=len(q),paired_primary_draws_per_block=200,paired_strict_draws_per_block=0,crossed_grids=len(s),dependent_cells_per_grid=100000,primary_full_refit=dict(median=v['median'],q025=v.q025,q975=v.q975),tests_passed=int(tests[1]),notebooks=len(execution),executed_code_cells=cells,clinical_fields=len(ledger),cantab_fields=28,context_fields=19,archived_files=sum(not r['immutable'] for r in m['files']),unchanged_derived_inputs=sum(r['immutable'] for r in m['files']),clinical_appendix_source_and_four_tables_unchanged=True,PDF_pages=len(pdf.pages),PDF_sha256=sha(ROOT/'report/main.pdf'),PDF_render_review='All-page contacts and detailed new methods/results/clinical pages reviewed in tmp/pdfs/joint_refit',interpretation='Conditional computational sampling with frequency-weighted nuisance refits; strict feasibility failure explicit; descriptive clinical alignment only.')
(RESULTS/'joint_refit_validation.json').write_text(json.dumps(o,indent=2));print(json.dumps(o,indent=2))
