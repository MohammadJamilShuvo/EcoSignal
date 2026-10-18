import numpy as np
import pandas as pd

def simulate(seed=42, n_sites=12, n_years=18, n_taxa=65):
    """Simulated detection histories; NOT measured DNA or organism abundances."""
    rng=np.random.default_rng(seed)
    years=np.arange(2005,2005+n_years)
    taxa=[f"Taxon_{i:03d}" for i in range(n_taxa)]
    affinity=rng.normal(14,3.2,n_taxa)
    rows=[]; det=[]
    for site_i in range(n_sites):
        site=f"S{site_i+1:02d}"
        site_offset=rng.normal(0,0.9)
        latent_env=0.
        for j,year in enumerate(years):
            latent_env=0.52*latent_env+rng.normal(0,0.32)
            temp=12.5+site_offset+0.082*j+latent_env
            rain=800+28*site_offset+12*np.sin(j/2.1)+rng.normal(0,26)
            shock=year>=2015
            rows.append(dict(site=site,year=int(year),temperature=round(temp,4),precipitation=round(rain,3),extreme_event=int(shock and j in (10,11)),true_break_year=2015))
            # Warm-associated taxa become more likely; cold taxa less likely; change accelerates after 2015.
            warming=((year-2005)*0.045 + max(0,year-2015)*0.085)
            logits=-0.15+0.17*(affinity-14)*(warming+site_offset*.12)+rng.normal(0,.16,n_taxa)
            true_occ=rng.random(n_taxa)<1/(1+np.exp(-logits))
            detection_p=0.83-0.12*(j%4==0)
            detected=true_occ & (rng.random(n_taxa)<detection_p)
            det.extend(dict(site=site,year=int(year),taxon=taxa[k],detected=int(detected[k])) for k in range(n_taxa))
    return pd.DataFrame(rows),pd.DataFrame(det),pd.DataFrame({'taxon':taxa,'thermal_affinity':affinity})
