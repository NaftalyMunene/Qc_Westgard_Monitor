import numpy as np
from labqc.westgard import evaluate_westgard, zscores


def test_zscores():
    assert np.allclose(zscores([5.3], 5.0, 0.15), [2.0])


def test_1_3s():
    assert evaluate_westgard([0, 0, 3.2, 0])["1_3s"].tolist() == [False, False, True, False]


def test_2_2s_same_side_only():
    assert evaluate_westgard([0, 2.1, 2.2, 0])["2_2s"].tolist() == [False, False, True, False]
    assert not evaluate_westgard([0, 2.1, -2.2, 0])["2_2s"].any()


def test_R_4s():
    r = evaluate_westgard([0, 2.3, -2.3, 0])
    assert r["R_4s"].tolist() == [False, False, True, False]


def test_4_1s():
    r = evaluate_westgard([0, 1.1, 1.2, 1.3, 1.4, 0])
    assert r["4_1s"].tolist() == [False, False, False, False, True, False]


def test_10_x():
    z = [0.2] * 10
    assert evaluate_westgard(z)["10_x"].iloc[-1]
    assert not evaluate_westgard(z[:9])["10_x"].any()


def test_in_control_no_reject():
    assert not evaluate_westgard([0.1, -0.3, 0.5, -0.2, 0.0])["reject"].any()
