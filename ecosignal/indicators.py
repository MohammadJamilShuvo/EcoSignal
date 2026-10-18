import numpy as np
import pandas as pd

def build_indicators(metadata, detections, traits):
    required={'site','year','taxon','detected'}
    if not required.issubset(detections): raise ValueError('detections: missing required columns')
    if detections.duplicated(['site','year','taxon']).any(): raise ValueError('Duplicate site-year-taxon')
    if not detections['detected'].isin([0,1]).all(): raise ValueError('detected must be 0/1')
    if traits['taxon'].duplicated().any(): raise ValueError('Duplicate traits')
    x=detections.merge(traits,on='taxon',validate='many_to_one')
    x['weighted_affinity']=x.detected*x.thermal_affinity
    totals=x.groupby(['site','year'],as_index=False).agg(richness=('detected','sum'),affinity_sum=('weighted_affinity','sum'))
    totals['cti']=totals.affinity_sum/totals.richness.replace(0,np.nan)
    totals=totals.drop(columns=['affinity_sum'])
    wide=detections.pivot_table(index=['site','year'],columns='taxon',values='detected',aggfunc='first')
    if wide.isna().any().any(): raise ValueError('Incomplete detection matrix; do not silently turn missing taxa into absence')
    turnover=[]
    for site, group in wide.groupby(level='site'):
        group=group.sort_index(level='year')
        previous=None
        for (s,y),r in group.iterrows():
            current=r.to_numpy().astype(bool)
            if previous is None: dissimilarity=np.nan
            else:
                union=(current|previous).sum(); dissimilarity=(current^previous).sum()/union if union else np.nan
            turnover.append(dict(site=s,year=y,jaccard_turnover=dissimilarity))
            previous=current
    out=metadata.merge(totals,on=['site','year'],validate='one_to_one').merge(pd.DataFrame(turnover),on=['site','year'],validate='one_to_one')
    return out

def reference_alerts(df,metric='cti', baseline_end=2009,quantile=.95):
    baseline=df.loc[df.year<=baseline_end,metric].dropna()
    if baseline.empty: raise ValueError('No observations in reference period')
    cutoff=float(baseline.quantile(quantile))
    out=df.copy();out[f'{metric}_reference_threshold']=cutoff;out[f'{metric}_alert']=(out[metric]>cutoff).astype(int)
    return out
