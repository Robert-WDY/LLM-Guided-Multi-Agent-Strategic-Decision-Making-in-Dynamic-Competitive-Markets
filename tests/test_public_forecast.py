import numpy as np
import pytest

from game_theory_agent.game_theory.public_forecast import inputs, realized_skill


def history(n=12):
    return [[100+i, 120-i, 42+i, 500-i, 180+i, i%4, 100+i, 80, 200] for i in range(n)]


def test_simultaneous_quote_has_no_access_to_current_own_offer():
    h=history()
    for native in (False,True):
        q1,s1=inputs(h,85,80,200,native_context=native,quote=True)
        q2,s2=inputs(h,180,80,200,native_context=native,quote=True)
        np.testing.assert_array_equal(q1,q2)
        assert s1==s2==h[-1][1]
        assert not np.array_equal(inputs(h,85,80,200,native_context=native)[0],
                                  inputs(h,180,80,200,native_context=native)[0])


def test_native_fields_are_allowlisted_for_ablation():
    h=history();altered=[r.copy() for r in h]
    altered[-1][4:7]=[300,80,190]
    basic,_=inputs(h,104,80,200)
    np.testing.assert_array_equal(basic,inputs(altered,104,100,150)[0])
    a,_=inputs(h,85,80,200,native_context=True)
    b,_=inputs(altered,85,100,150,native_context=True)
    assert not np.array_equal(a,b)
    np.testing.assert_array_equal(a[:-10],inputs(h,85,80,200)[0])


def test_zero_sales_and_unchanging_quote_are_finite():
    h=history()
    for r in h:r[2]=r[4]=r[5]=0
    x,s=inputs(h,90,80,200,native_context=True)
    assert s==1 and np.isfinite(x).all()
    np.testing.assert_array_equal(realized_skill([[0,8]],[[0,8]],[[0,8]],[[1,8]]),[0,0])
    with pytest.raises(ValueError):
        inputs(h[:5],90,80,200)


def test_relative_error_skill_compares_settled_errors_per_target():
    actual=[[10,100],[10,100]]
    score=realized_skill(actual,[[8,100],[7,100]],[[10,105],[10,106]],[[10,100],[10,100]])
    assert score[0]>0>score[1]

