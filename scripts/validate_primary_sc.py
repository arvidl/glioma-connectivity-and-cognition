"""Audit the delivered policy, preserved inputs, notebooks and report."""
import hashlib,json,re
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader
from pat16 import ROOT,RESULTS,DATA,SESSIONS
from pat16.bold import sha
from pat16.primary_sc import OUT,VERSION,load_weights

archive=RESULTS/'before_primary_sc'
m=json.loads((archive/'manifest.json').read_text())
for r in m['files']:
    p=ROOT/r['path'] if r['immutable'] else archive/r['path']
    assert sha(p)==r['sha256'],p
clinical=(ROOT/'report/main.tex').read_text().split('\\section{Complete supplied assessments}')[1]
assert hashlib.sha256(('\\section{Complete supplied assessments}'+clinical).encode()).hexdigest()==m['clinical_appendix_source_sha256']
for name in ['cantab_all.tex','psychosocial_all.tex','cantab_all_dictionary.tex','psychosocial_all_dictionary.tex']:
    assert sha(ROOT/'report/generated'/name)==sha(archive/'report/generated'/name)
execution=json.loads((RESULTS/'notebook_execution.json').read_text())
assert len(execution)>=10 and len(execution)==len(list((ROOT/'notebooks').glob('*.ipynb'))) and all(x['errors']==0 for x in execution)
cells=0
for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(p,as_version=4)
    for c in nb.cells:
        if c.cell_type!='code':continue
        cells+=1;assert c.execution_count is not None,p
        assert not any(o.output_type=='error' for o in c.outputs),p
    assert 'Current SC policy' in nb.cells[1].source,p
tests=(RESULTS/'primary_sc_tests.log').read_text()
match=re.search(r'(\d+) passed',tests);assert match and 'FAILED' not in tests
reader=PdfReader(ROOT/'report/main.pdf');pdftext='\n'.join(p.extract_text() for p in reader.pages)
assert 'Primary reliability-based SC results' in pdftext
assert 'Primary coupling: common' in pdftext and '823' in pdftext
log=(ROOT/'report/main.log').read_text()
assert not any(x in log for x in ['Overfull','Undefined control','undefined references','Citation `'])
stages=pd.read_csv(OUT/'censoring_decomposition.csv')
for s,prefix in zip(SESSIONS,['pre','post']):
    rho=pd.read_csv(OUT/'denoising_comparison.csv').query("session==@s and model=='anatomical5' and BOLD_processing=='corrected'").iloc[0].sc_fc_spearman
    assert np.isclose(rho,stages.iloc[0][prefix+'_rho'],atol=1e-12)
    pc=pd.read_csv(OUT/'pc_sensitivity.csv').query("session==@s and components==5 and BOLD_processing=='corrected'").iloc[0].sc_fc_spearman
    assert np.isclose(rho,pc,atol=1e-12)
obj={'policy':VERSION,'recommendation':'post hoc, unblinded; conditional seed stability, not anatomical validity',
     'tests_passed':int(match[1]),'notebooks':len(execution),'executed_code_cells':cells,
     'preserved_archive_files':sum(not x['immutable'] for x in m['files']),
     'unchanged_derived_inputs':sum(x['immutable'] for x in m['files']),
     'clinical_appendix_source_unchanged':True,'clinical_tables_and_dictionaries_unchanged':True,
     'raw_input_manifest':'scripts/run_all.py verifies all 63 bundled inputs before analysis',
     'SC_edges_own':[int(np.count_nonzero(np.triu(load_weights(s),1))) for s in SESSIONS],
     'SC_edges_common':823,'FC_unchanged':True,'historical_SC_and_all_40_repeats_unchanged':True,
     'coupling_primary_delta':float(stages.iloc[0].delta),'coupling_strict_delta':float(stages.iloc[-1].delta),
     'matched_refit_draws':len(pd.read_csv(OUT/'matched_coupling.csv')),
     'coupling_sensitivity_designs':len(pd.read_csv(OUT/'coupling_sensitivity.csv')),
     'pipeline_log':'results/primary_sc_pipeline.log',
     'PDF_pages':len(reader.pages),'PDF_sha256':sha(ROOT/'report/main.pdf'),
     'PDF_render_review':'all pages in tmp/pdfs/primary_sc; key new method/results/graph pages reviewed at higher resolution'}
(RESULTS/'primary_sc_validation.json').write_text(json.dumps(obj,indent=2))
print(json.dumps(obj,indent=2))
