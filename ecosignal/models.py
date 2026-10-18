import numpy as np
import pandas as pd
from scipy.stats import norm

def _ols(x,y):
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    residual=y-x@beta
    return beta,float(residual@residual)

def fixed_effect_trend(frame, response='cti'):
    """Illustrative site-fixed-effects OLS with site-cluster bootstrap CI.
    Not a calibrated longitudinal model; see README limitations.
    """
    data=frame[['site','year',response]].dropna().copy()
    if data.site.nunique()<2 or data.year.nunique()<4: raise ValueError('Need >=2 sites and >=4 years')
    sites=sorted(data.site.unique()); years=data.year.to_numpy(); year0=years.min()
    x=np.column_stack((np.ones(len(data)),years-year0,pd.get_dummies(data.site,drop_first=True,dtype=float).to_numpy()))
    y=data[response].to_numpy(); beta,sse=_ols(x,y)
    # Nonparametric cluster bootstrap resamples whole sites and re-labels them.
    rng=np.random.default_rng(1234); slopes=[]
    for i in range(250):
        pieces=[]
        for b,s in enumerate(rng.choice(sites,size=len(sites),replace=True)):
            z=data.loc[data.site==s].copy();z['site']=f'bootstrap_{b}';pieces.append(z)
        d=pd.concat(pieces,ignore_index=True)
        xb=np.column_stack((np.ones(len(d)),d.year.to_numpy()-year0,pd.get_dummies(d.site,drop_first=True,dtype=float).to_numpy()))
        slopes.append(_ols(xb,d[response].to_numpy())[0][1])
    lo,hi=np.quantile(slopes,[.025,.975])
    return dict(metric=response,estimate_per_year=float(beta[1]),ci_low=float(lo),ci_high=float(hi),bootstrap_replicates=250,n_sites=len(sites))

def breakpoint_screen(frame,response='cti',min_edge=3):
    """Exploratory piecewise pooled site-FE model selection; not a change-point significance test."""
    d=frame[['site','year',response]].dropna().copy(); year0=d.year.min()
    sites=pd.get_dummies(d.site,drop_first=True,dtype=float).to_numpy()
    base=np.column_stack((np.ones(len(d)),d.year.to_numpy()-year0,sites))
    y=d[response].to_numpy(); _,rss0=_ols(base,y)
    n=len(y); best={'break_year':None,'bic':float(n*np.log(max(rss0/n,1e-12))+base.shape[1]*np.log(n)),'delta_bic_vs_linear':0.}
    for year in sorted(d.year.unique())[min_edge:-min_edge]:
        x=np.column_stack((base,np.maximum(0,d.year.to_numpy()-year)))
        _,rss=_ols(x,y);bic=float(n*np.log(max(rss/n,1e-12))+x.shape[1]*np.log(n))
        if bic<best['bic']:
            best.update(break_year=int(year),bic=bic)
    best['delta_bic_vs_linear']=round(best['bic']-(n*np.log(max(rss0/n,1e-12))+base.shape[1]*np.log(n)),4)
    return best

def power_screen(seed=7, repetitions=160):
    """Known linear Gaussian trend; false positive rates are evaluated at slope zero.
    Simplified iid model used ONLY as a baseline sanity check, not field design guidance.
    """
    from scipy.stats import linregress
    rng=np.random.default_rng(seed)
    rows=[]
    for years in (6,10,15,20):
        for interval in (1,2,3):
            t=np.arange(0,years,interval,dtype=float)
            if len(t)<4: continue
            for slope in (0.,0.07,0.14):
                significant=0
                for _ in range(repetitions):
                    y=slope*t+rng.normal(0,.75,len(t))
                    significant+=linregress(t,y).pvalue<.05
                rows.append(dict(years=years,interval=interval,observations=len(t),true_slope=slope,positive_rate=significant/repetitions))
    return pd.DataFrame(rows)
