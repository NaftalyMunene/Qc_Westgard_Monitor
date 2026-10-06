"""Westgard multirule evaluation for a single control level.

Rules implemented (z = (value - target mean) / target SD):
  1_2s  warning   : one point beyond +/-2 SD
  1_3s  reject    : one point beyond +/-3 SD (random error)
  2_2s  reject    : two consecutive points beyond the same 2 SD limit (systematic error)
  R_4s  reject    : consecutive points on opposite sides, spanning more than 4 SD (random error)
  4_1s  reject    : four consecutive points beyond the same 1 SD limit (systematic error)
  10_x  reject    : ten consecutive points on the same side of the mean (systematic error)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

REJECT_RULES = ["1_3s", "2_2s", "R_4s", "4_1s", "10_x"]
ALL_RULES = ["1_2s"] + REJECT_RULES


def zscores(values, mean: float, sd: float) -> np.ndarray:
    if sd <= 0:
        raise ValueError("sd must be positive")
    return (np.asarray(values, dtype=float) - mean) / sd


def _run_all(flag: np.ndarray, n: int) -> np.ndarray:
    """True at index i if flag is True for the n points ending at i."""
    out = np.zeros(len(flag), dtype=bool)
    if len(flag) < n:
        return out
    c = np.convolve(flag.astype(int), np.ones(n, dtype=int), mode="valid")
    out[n - 1:] = c == n
    return out


def evaluate_westgard(z) -> pd.DataFrame:
    """Return a DataFrame with one boolean column per rule, indexed by run order."""
    z = np.asarray(z, dtype=float)
    res = pd.DataFrame(index=range(len(z)))
    res["1_2s"] = np.abs(z) > 2
    res["1_3s"] = np.abs(z) > 3
    res["2_2s"] = _run_all(z > 2, 2) | _run_all(z < -2, 2)
    r4 = np.zeros(len(z), dtype=bool)
    if len(z) > 1:
        r4[1:] = (np.abs(np.diff(z)) > 4) & (np.sign(z[1:]) != np.sign(z[:-1]))
    res["R_4s"] = r4
    res["4_1s"] = _run_all(z > 1, 4) | _run_all(z < -1, 4)
    res["10_x"] = _run_all(z > 0, 10) | _run_all(z < 0, 10)
    res["reject"] = res[REJECT_RULES].any(axis=1)
    return res


def first_rule_violated(row: pd.Series) -> str:
    for r in REJECT_RULES:
        if row.get(r, False):
            return r
    return "1_2s" if row.get("1_2s", False) else ""
