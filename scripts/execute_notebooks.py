import os,sys,time,json
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from pat16 import ROOT
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.cache/jupyter')
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True,exist_ok=True)
specdir=ROOT/'.cache/jupyter-data/kernels/pat16-local'
specdir.mkdir(parents=True,exist_ok=True)
(specdir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'PAT16 uv','language':'python'}))
os.environ['JUPYTER_PATH']=str(ROOT/'.cache/jupyter-data')
class PlainOutputClient(NotebookClient):
    # nbclient 0.11 OutputWidget comm mirroring can loop on large inline Plotly HTML.
    # Execute real callbacks and retain their static outputs; live controls activate
    # normally when the notebook is run in Jupyter. Ignore headless widget comms.
    def handle_comm_msg(self, outs, msg, cell_index):
        return None

records=[]
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
    print('Executing',path.name,flush=True);start=time.time()
    nb=nbformat.read(path,as_version=4)
    client=PlainOutputClient(nb,timeout=180,kernel_name='pat16-local',resources={'metadata':{'path':str(ROOT)}},record_timing=True, store_widget_state=False, iopub_timeout=30, on_cell_start=lambda cell, cell_index: print('  cell',cell_index,cell.cell_type,flush=True))
    # Use exactly the uv project's interpreter without installing a global kernelspec.
    client.execute()
    nbformat.write(nb,path)
    errors=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
    if errors:raise RuntimeError(path.name+' contains cell errors')
    records.append({'notebook':path.name,'seconds':round(time.time()-start,2),'code_cells':sum(c.cell_type=='code' for c in nb.cells),'errors':len(errors)})
(ROOT/'results/notebook_execution.json').write_text(json.dumps(records,indent=2));print(records)
