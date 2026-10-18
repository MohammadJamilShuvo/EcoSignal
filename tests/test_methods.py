import numpy as np
import pytest
from ecosignal.simulation import simulate
from ecosignal.indicators import build_indicators,reference_alerts
from ecosignal.models import fixed_effect_trend,breakpoint_screen,power_screen

def data():
    m,d,t=simulate(seed=23,n_sites=4,n_years=8,n_taxa=20)
    return m,d,t,build_indicators(m,d,t)

def test_reproducibility():
    a=simulate(seed=7);b=simulate(seed=7)
    assert all(x.equals(y) for x,y in zip(a,b))

def test_indicator_bounds():
    *_,ind=data()
    assert ind.richness.between(0,20).all()
    assert ind.jaccard_turnover.dropna().between(0,1).all()

def test_incomplete_matrix_rejected():
    m,d,t,_=data()
    with pytest.raises(ValueError,match='Incomplete'):
        build_indicators(m,d.iloc[:-1],t)

def test_reference():
    *_,ind=data()
    ref=reference_alerts(ind,baseline_end=2007)
    assert ref.cti_alert.isin([0,1]).all()

def test_trend():
    *_,ind=data()
    result=fixed_effect_trend(ind)
    assert np.isfinite(result['estimate_per_year'])

def test_breakpoint():
    *_,ind=data()
    assert 'break_year' in breakpoint_screen(ind,min_edge=2)

def test_power():
    assert power_screen(repetitions=8).positive_rate.between(0,1).all()


def test_missing_traits_are_rejected():
    m, d, t, _ = data()
    with pytest.raises(ValueError, match="Missing thermal affinities"):
        build_indicators(m, d, t.iloc[:-1].copy())

def test_site_year_mismatch_rejected():
    m, d, t, _ = data()
    with pytest.raises(ValueError, match="identical site-year coverage"):
        build_indicators(m.iloc[:-1].copy(), d, t)

def test_missing_traits_do_not_change_richness_silently():
    m, d, t, _ = data()
    t = t.loc[t.taxon != t.taxon.iloc[0]].copy()
    with pytest.raises(ValueError, match="Missing thermal affinities"):
        build_indicators(m, d, t)

def test_cti_exact_known_case():
    import pandas as pd
    m = pd.DataFrame({"site": ["A"], "year": [2005]})
    d = pd.DataFrame({"site": ["A", "A"], "year": [2005, 2005],
                      "taxon": ["cold", "warm"], "detected": [1, 1]})
    t = pd.DataFrame({"taxon": ["cold", "warm"], "thermal_affinity": [10.0, 20.0]})
    r = build_indicators(m, d, t)
    assert r.richness.iloc[0] == 2
    assert r.cti.iloc[0] == 15.0

def test_missing_cti_is_not_a_negative_alert():
    import pandas as pd
    frame = pd.DataFrame({"site": ["A", "A"], "year": [2005, 2006],
                          "cti": [15.0, np.nan]})
    out = reference_alerts(frame, baseline_end=2005)
    assert pd.isna(out.cti_alert.iloc[1])
