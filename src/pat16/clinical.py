"""All reported assessment measures, with raw changes and cautious directions."""
import json
import numpy as np
import pandas as pd
from . import DATA, RESULTS, ROOT


def assessment_tables():
    base=DATA/'sub-PAT16/sub-PAT16_sessions'
    meta=json.loads(base.with_suffix('.json').read_text())
    annotations=json.loads((ROOT/'config/assessment_annotations.json').read_text())
    sessions=pd.read_csv(base.with_suffix('.tsv'),sep='\t',na_values=['n/a']).set_index('session_id')
    rows=[]
    for measure in sessions:
        pre,post=sessions.loc['ses-preop',measure],sessions.loc['ses-postop',measure]
        desc=meta.get(measure,{}).get('Description',measure)
        numeric=isinstance(pre,(int,float,np.number)) and isinstance(post,(int,float,np.number))
        cantab=measure.startswith(('mot_','rti_','rvp_','soc_','ssp_'))
        direction=None
        if cantab and not measure.endswith('_sd'):
            direction=-1 if any(x in measure for x in ['latency','_rt','_mt','error','false_alarms','mean_moves']) else 1
        delta=float(post-pre) if numeric and np.isfinite(pre) and np.isfinite(post) else np.nan
        annotation=annotations[measure]
        rows.append({'measure':measure,'description':desc,'group':'CANTAB' if cantab else 'psychosocial/context','source_units':meta.get(measure,{}).get('Units','not supplied'),'preop':pre,'postop':post,'delta_post_minus_pre':delta,'percent_change':100*delta/abs(pre) if numeric and pre!=0 else np.nan,'favorable_direction':direction,'direction_aligned_percent':direction*100*delta/abs(pre) if direction and numeric and pre!=0 else np.nan,'interpretation':'raw descriptive change; no reliable-change threshold available',**annotation})
    table=pd.DataFrame(rows)
    table.to_csv(RESULTS/'assessments_all.csv',index=False)
    table[table.group=='CANTAB'].to_csv(RESULTS/'cantab_all.csv',index=False)
    table[table.group!='CANTAB'].to_csv(RESULTS/'psychosocial_all.csv',index=False)
    return table


HYPOTHESES = [
    ('MOT','Sensorimotor / visual attention','SomMot; Vis','latency and spatial error; no isolated motor deficit can be diagnosed'),
    ('RTI','Speeded detection, decision and motor execution','DorsAttn; SalVentAttn; SomMot','reaction and movement times differ; ceiling accuracy'),
    ('RVP',"Sustained attention / target detection",'DorsAttn; SalVentAttn; Cont',"A-prime and hit probability vs latency and false alarms"),
    ('SOC','Planning / executive control','Cont; DorsAttn','problem difficulty matters; 4-move and 5-move changes differ'),
    ('SSP','Visuospatial short-term memory','DorsAttn; Cont; Vis','span is one raw ordinal score; practice effects unknown'),
]
