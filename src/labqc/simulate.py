"""Synthetic QC data generator. No real patient or clinic data is used."""
from __future__ import annotations
import numpy as np
import pandas as pd

ANALYTES = {
    # analyte: (unit, {level: (target mean, target SD)})
    "Glucose":    ("mmol/L",  {"L1": (5.0, 0.15),  "L2": (15.0, 0.45)}),
    "Creatinine": ("umol/L",  {"L1": (80.0, 2.4),  "L2": (400.0, 12.0)}),
    "ALT":        ("U/L",     {"L1": (40.0, 1.6),  "L2": (150.0, 6.0)}),
}
BRANCHES = ["Branch_A", "Branch_B", "Branch_C", "Branch_D"]


def simulate(days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Daily QC, two levels, four branches, with three planted problems:
      - Branch_B Glucose: abrupt systematic shift (+1.5 SD) from day 45
      - Branch_C Creatinine: gradual upward trend from day 30
      - Branch_D ALT: increased random error (imprecision) from day 60
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    rows = []
    for b in BRANCHES:
        for a, (unit, levels) in ANALYTES.items():
            for lvl, (mean, sd) in levels.items():
                z = rng.normal(0, 1, days)
                if b == "Branch_B" and a == "Glucose":
                    z[45:] += 1.5
                if b == "Branch_C" and a == "Creatinine":
                    z[30:] += np.linspace(0, 2.0, days - 30)
                if b == "Branch_D" and a == "ALT":
                    z[60:] *= 2.0
                vals = mean + z * sd
                for d, v in zip(dates, vals):
                    rows.append((d, b, a, lvl, unit, mean, sd, v))
    return pd.DataFrame(rows, columns=["date", "branch", "analyte", "level", "unit", "target_mean", "target_sd", "value"])
