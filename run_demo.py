from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ecosignal.simulation import simulate
from ecosignal.indicators import build_indicators,reference_alerts
from ecosignal.models import fixed_effect_trend,breakpoint_screen,power_screen

def main():
    out=Path('results');(out/'figures').mkdir(parents=True,exist_ok=True)
    meta,detect,traits=simulate()
    for filename,frame in [('metadata.csv',meta),('detections.csv',detect),('taxon_traits.csv',traits)]: frame.to_csv(out/filename,index=False)
    indicators=build_indicators(meta,detect,traits)
    indicators=reference_alerts(indicators)
    indicators.to_csv(out/'indicators.csv',index=False)
    models=[fixed_effect_trend(indicators,x) for x in ['richness','cti','jaccard_turnover']]
    bp=breakpoint_screen(indicators)
    design=power_screen();design.to_csv(out/'power_screen.csv',index=False)
    (out/'summary.json').write_text(json.dumps({'scenario':'synthetic only','trends':models,'breakpoint_screen':bp,'disclaimer':'Statistical screens, not validated ecological inference or calibrated authority thresholds.'},indent=2))
    plt.figure(figsize=(9,4.7))
    for metric in ['richness','cti']:
        mean=indicators.groupby('year')[metric].mean()
        z=(mean-mean.iloc[0])/mean.std()
        plt.plot(z.index,z.values,marker='o',label=f'{metric}: standardized')
    plt.axvline(2015,ls='--',color='gray',label='simulated acceleration')
    plt.xlabel('Year');plt.ylabel('Standardized mean indicator');plt.legend();plt.tight_layout();plt.savefig(out/'figures'/'01_indicator_trajectories.png',dpi=170);plt.close()
    plt.figure(figsize=(9,4.7))
    for site,grp in indicators.groupby('site'):
        plt.plot(grp.year,grp.cti,alpha=.35,lw=1)
    plt.plot(indicators.groupby('year').cti.mean(),color='black',lw=2.5,label='Across-site mean')
    plt.axhline(indicators.cti_reference_threshold.iloc[0],color='gray',ls='--',label='Pooled early-period 95th percentile')
    plt.xlabel('Year');plt.ylabel('CTI (simulated °C affinities)');plt.legend();plt.tight_layout();plt.savefig(out/'figures'/'02_cti_reference_screen.png',dpi=170);plt.close()
    plt.figure(figsize=(9,4.7))
    for span,grp in design.loc[design.true_slope==.14].groupby('interval'):
        plt.plot(grp.years,grp.positive_rate,marker='o',label=f'every {span} year(s)')
    plt.xlabel('Monitoring duration (years)');plt.ylabel('Positive test fraction (simulation)');plt.ylim(0,1);plt.legend();plt.tight_layout();plt.savefig(out/'figures'/'03_design_screen.png',dpi=170);plt.close()
    print('Completed simulated demonstration; see results/summary.json and figures/.')
if __name__=='__main__':main()
