"""Adjacent-PC and wider-count sensitivity for the recommended primary model."""
import os
from pat16 import ROOT,RESULTS,SESSIONS
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pat16.bold import variant_folder
from pat16.nuisance import OUT
from pat16.connectivity import clean_timeseries
from pat16.networks import sc_fc_coupling

common=np.load(RESULTS/'common_fc_nodes.npy');i,j=np.triu_indices(common.sum(),1)
rows=[]
for s in SESSIONS:
    sc=np.load(RESULTS/s/'sc_fraction.npy')[np.ix_(common,common)]
    for variant in ['uncorrected','corrected']:
        src=variant_folder(s,variant)
        y=np.load(src/'roi_timeseries_raw.npy');motion=np.loadtxt(src/'func_motion.tsv')
        pcs=np.load(OUT/s/variant/'anatomical_pcs20.npy')
        for k in range(21):
            f,_,_,q,_=clean_timeseries(y,motion,pcs[:,:k]);f=f[np.ix_(common,common)]
            rows.append({'session':s,'BOLD_processing':variant,'components':k,'PVE_threshold':.95,
                'FD_threshold':.5,'mean_fc_r':float(f[i,j].mean()),'sc_fc_spearman':sc_fc_coupling(sc,f),
                'residual_df':q['residual_df'],'role':'primary' if k==5 else 'sensitivity'})
df=pd.DataFrame(rows);df.to_csv(RESULTS/'primary_pc_sensitivity.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(9.7,3.8))
for s,color in [('preop','#357da8'),('postop','#d88b3c')]:
    d=df[(df.session==s)&(df.BOLD_processing=='corrected')]
    for ax,col in zip(axs,['mean_fc_r','sc_fc_spearman']):
        ax.plot(d.components,d[col],'o-',color=color,label=s,ms=3)
        ax.axvline(5,color='#328c91',ls=':',lw=1)
        ax.set_xlabel('Retained-frame anatomical PC count');ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('Mean FC r');axs[1].set_ylabel('SC–FC Spearman correlation');axs[0].legend()
fig.suptitle('Primary-model count sensitivity | fixed tissue masks, signals and censoring',fontsize=12)
fig.tight_layout();fig.savefig(RESULTS/'figures/primary_pc_sensitivity.png',dpi=180);plt.close(fig)
print(df.query("BOLD_processing=='corrected' and components in [2,4,5,6,10]").to_string(index=False))
