"""Matched rigid-only vs topup/eddy reconstruction; no biological inference."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pat16 import RESULTS, SESSIONS
from pat16.imaging import anatomy, rigid_motion_correct
from pat16.connectivity import structural
from pat16.networks import metrics, sc_fc_coupling

anatomy()
for session in SESSIONS:
    out=RESULTS/'uncorrected_matched'/session
    rigid_motion_correct('dwi',session,out)
    structural(session,variant='uncorrected_matched')
    primary=json.loads((RESULTS/session/'sc_qc.json').read_text())
    if primary.get('preprocessing')!='corrected':
        raise RuntimeError('Primary SC is not topup/eddy-derived. Run run_all.py --from-raw first.')

matrices={(s,v):np.load((RESULTS/s if v=='corrected' else RESULTS/'uncorrected_matched'/s)/'sc_fraction.npy') for s in SESSIONS for v in ['corrected','uncorrected_matched']}
# One scale across both sessions AND both processing variants.
scale=max(w.max() for w in matrices.values())
rows=[]
for (s,v),w in matrices.items():
    out=RESULTS/s if v=='corrected' else RESULTS/'uncorrected_matched'/s
    q=json.loads((out/'sc_qc.json').read_text())
    rows.append({'session':s,'preprocessing':v,'assigned_streamlines':q['assigned_interparcel_streamlines'],
        'length_filtered_streamlines':q['length_filtered_streamlines'],
        'sc_fc_spearman':sc_fc_coupling(w,np.load(RESULTS/s/'fc_primary.npy')),
        **metrics(w,.2,scale)})
pd.DataFrame(rows).to_csv(RESULTS/'correction_comparison.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(9.4,3.6))
for ax,column,title in zip(axs,['assigned_streamlines','binary_global_efficiency','weighted_global_efficiency'],['Assigned interparcel streamlines','Binary efficiency (20% density)','Weighted efficiency (shared scale)']):
    df=pd.DataFrame(rows)
    for v,color,label in [('uncorrected_matched','#999999','Rigid-only, matched model'),('corrected','#328c91','Topup + eddy')]:
        d=df[df.preprocessing==v].set_index('session').loc[list(SESSIONS)]
        ax.plot(['Pre-op','Post-op'],d[column],marker='o',color=color,label=label)
    ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
axs[0].legend(fontsize=9);fig.tight_layout();fig.savefig(RESULTS/'figures/correction_comparison.png',dpi=180);plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False))
