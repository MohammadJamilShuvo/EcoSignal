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
