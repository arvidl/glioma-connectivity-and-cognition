"""Execute each notebook from its own runtime subset, without sibling data.

Also call widget functions directly: frontend Output widgets otherwise hide
callback exceptions from nbclient. Exercise each control's options/endpoints.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from pat16 import ROOT
from pat16.notebook_assets import catalog, ensure_notebook


class Client(NotebookClient):
    def handle_comm_msg(self, outs, msg, cell_index):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path)
    parser.add_argument('--notebook', action='append')
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/binder-validation.json')
    parser.add_argument('--executed-dir', type=Path, help='Save successfully executed study notebooks')
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    records = []
    profiles = catalog()['notebooks']
    for key in args.notebook or profiles:
        with tempfile.TemporaryDirectory(prefix='pat16-notebook-') as directory:
            root = Path(directory).resolve()
            for name in ['src', 'config', 'binder', 'reproducibility']:
                shutil.copytree(ROOT/name, root/name)
            shutil.copy(ROOT/'pyproject.toml', root/'pyproject.toml')
            (root/'notebooks').mkdir()
            path = root/'notebooks'/profiles[key]['notebook']
            shutil.copy(ROOT/'notebooks'/path.name, path)
            ensure_notebook(key, root=root, assets_dir=args.assets_dir)
            runtime = root/'.cache/jupyter'; runtime.mkdir(parents=True)
            specs = root/'.cache/jupyter-data/kernels/pat16-test'; specs.mkdir(parents=True)
            (specs/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], display_name='PAT16 test', language='python')))
            os.environ.update(JUPYTER_RUNTIME_DIR=str(runtime), JUPYTER_PATH=str(root/'.cache/jupyter-data'), IPYTHONDIR=str(root/'.cache/ipython'))
            nb = nbformat.read(path, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == 'code':
                    cell.outputs = []
                    cell.execution_count = None
            code_cells = sum(cell.cell_type == 'code' for cell in nb.cells)
            setup = f'''import sys, os
sys.path.insert(0, {str(root/'src')!r})
os.environ['MPLCONFIGDIR'] = {str(root/'.cache/matplotlib')!r}
from pat16 import ROOT as _root
assert str(_root) == {str(root)!r}, _root
import ipywidgets as _widgets
_original_interact = _widgets.interact
_callbacks = []
def _capture(f=None, **kwargs):
    result = _original_interact(f, **kwargs)
    _callbacks.append((f, kwargs))
    return result
_widgets.interact = _capture
'''
            checks = '''import resource, json
from IPython.display import clear_output
_count = 0
for _f, _controls in _callbacks:
    _base = {k: (v.value if hasattr(v, 'value') else v) for k,v in _controls.items()}
    _f(**_base); _count += 1
    clear_output(wait=True); plt.close('all')
    for _key, _control in _controls.items():
        if hasattr(_control, 'options'):
            _options = _control.options
            _values = list(_options.values()) if isinstance(_options, dict) else [v[1] if isinstance(v, tuple) else v for v in _options]
        elif hasattr(_control, 'min') and hasattr(_control, 'max'):
            _values = [_control.min, _control.max]
        elif isinstance(_base[_key], bool):
            _values = [False, True]
        else:
            continue
        for _value in _values:
            _f(**{**_base, _key:_value}); _count += 1
            clear_output(wait=True); plt.close('all')
_root.joinpath('.cache/validation.json').write_text(json.dumps(dict(widget_calls=_count, max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
'''
            nb.cells.insert(0, nbformat.v4.new_code_cell(setup))
            nb.cells.append(nbformat.v4.new_code_cell(checks))
            Client(nb, timeout=600, kernel_name='pat16-test', resources={'metadata': {'path': str(root/'notebooks')}}, store_widget_state=False, iopub_timeout=60).execute()
            record = json.loads((root/'.cache/validation.json').read_text())
            record.update(notebook=path.name, code_cells=code_cells, errors=0,
                          max_rss_units='bytes' if sys.platform == 'darwin' else 'KiB')
            if args.executed_dir:
                args.executed_dir.mkdir(parents=True, exist_ok=True)
                nb.cells = nb.cells[1:-1]  # exclude setup and callback test cells
                nbformat.write(nb, args.executed_dir/path.name)
            records.append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, indent=2)+'\n')
            print(key, record, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
