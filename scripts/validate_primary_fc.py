"""Current-delivery audit; historical audit records are never rewritten."""
import hashlib,json,re
import nbformat,pandas as pd
from pypdf import PdfReader
from pat16 import ROOT,RESULTS
from pat16.bold import sha
from pat16.primary_fc import OUT,VERSION,inputs

manifest=json.loads((RESULTS/'before_primary_fc/manifest.json').read_text())
for r in manifest['files']:
    p=(ROOT if r['immutable'] else RESULTS/'before_primary_fc')/r['path']
    assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
source=(ROOT/'report/main.tex').read_text();appendix=source[source.index('\\section{Complete supplied assessments}'):]
assert hashlib.sha256(appendix.encode()).hexdigest()==manifest['clinical_appendix_source_sha256']
for name in ['cantab_all','cantab_all_dictionary','psychosocial_all','psychosocial_all_dictionary']:
    assert sha(ROOT/'report/generated'/f'{name}.tex')==sha(RESULTS/'before_primary_fc/report/generated'/f'{name}.tex')
d=json.loads((OUT/'design.json').read_text());assert d['inputs']==inputs()
for p,h in d['outputs'].items():assert sha(OUT/p)==h,p
s=pd.read_csv(OUT/'summary.csv');f=pd.read_csv(OUT/'feasibility.csv');q=pd.read_csv(OUT/'fit_metrics.csv')
assert len(s)==28 and len(f)==9 and len(q)==10800 and (f.paired_eligible==200).all()
assert (s.role=='primary').sum()==1
old=pd.read_csv(RESULTS/'evidence_integration/assessment_evidence_ledger.csv');ledger=pd.read_csv(OUT/'clinical_evidence_ledger.csv')
pd.testing.assert_frame_equal(ledger[old.columns],old);assert len(ledger)==47
execution=json.loads((RESULTS/'notebook_execution.json').read_text());assert len(execution)==15 and all(r['errors']==0 for r in execution)
cells=0
for path in (ROOT/'notebooks').glob('*.ipynb'):
    for c in nbformat.read(path,as_version=4).cells:
        if c.cell_type=='code':
            cells+=1;assert c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs),path
log=(ROOT/'report/main.log').read_text();assert not any(x in log for x in ['Overfull','Undefined control','undefined references','Citation `','multiply defined'])
pdf=PdfReader(ROOT/'report/main.pdf');text='\n'.join(p.extract_text() for p in pdf.pages)
for phrase in ['Recommended primary FC resampling','Primary fixed-duration FC stability results','All CANTAB measures','Psychosocial and contextual fields']:assert phrase in text,phrase
audit=RESULTS/'primary_fc_audit'
render=audit/'render_review.json';assert render.exists()
assert json.loads(render.read_text())['PDF_sha256']==sha(ROOT/'report/main.pdf')
testlog=(audit/'tests.log').read_text();n=re.search(r'(\d+) passed',testlog);assert n and 'FAILED' not in testlog
pipeline=(audit/'pipeline.log').read_text();assert 'Completed in ' in pipeline and 'Traceback' not in pipeline
obj=dict(version=VERSION,primary=json.loads((OUT/'main_summary.json').read_text()),temporal_designs=9,crossed_grids=28,SC_draws=500,FC_draws=200,tests_passed=int(n[1]),notebooks=15,executed_code_cells=cells,clinical_fields=47,clinical_appendix_and_four_tables_unchanged=True,archived_files=sum(not r['immutable'] for r in manifest['files']),unchanged_files=sum(r['immutable'] for r in manifest['files']),PDF_pages=len(pdf.pages),PDF_sha256=sha(ROOT/'report/main.pdf'),PDF_render_review=json.loads(render.read_text()),classification='Post hoc conditional subset stability recommendation; no full-run coverage or clinical inference')
(RESULTS/'primary_fc_validation.json').write_text(json.dumps(obj,indent=2)+'\n');print(json.dumps({k:v for k,v in obj.items() if k not in ['primary','PDF_render_review']},indent=2))
