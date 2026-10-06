"""Run Westgard rules on every branch/analyte/level and detect cross-branch drift."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from .westgard import evaluate_westgard, zscores, first_rule_violated
from .simulate import simulate


def run_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (b, a, l), g in df.sort_values("date").groupby(["branch", "analyte", "level"]):
        g = g.reset_index(drop=True)
        z = zscores(g["value"], g["target_mean"].iloc[0], g["target_sd"].iloc[0])
        res = evaluate_westgard(z)
        g = pd.concat([g, res], axis=1)
        g["z"] = z
        g["rule"] = res.apply(first_rule_violated, axis=1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def summary(flagged: pd.DataFrame) -> pd.DataFrame:
    s = (flagged.groupby(["branch", "analyte"])
         .agg(runs=("value", "size"), rejections=("reject", "sum"), warnings=("1_2s", "sum"),
              mean_z=("z", "mean"), sd_z=("z", "std"))
         .reset_index())
    s["rejection_rate_pct"] = (100 * s["rejections"] / s["runs"]).round(1)
    return s.round(3).sort_values("rejection_rate_pct", ascending=False)


def drift_anomalies(flagged: pd.DataFrame, window: int = 7, contamination: float = 0.05) -> pd.DataFrame:
    """Rolling mean and SD of z per branch/analyte/level, scored by Isolation Forest.

    Complements Westgard rules: flags windows where bias (rolling mean z) or
    imprecision (rolling SD z) departs from the pattern seen across all branches.
    """
    f = flagged.sort_values("date").copy()
    grp = f.groupby(["branch", "analyte", "level"])["z"]
    f["roll_mean_z"] = grp.transform(lambda s: s.rolling(window).mean())
    f["roll_sd_z"] = grp.transform(lambda s: s.rolling(window).std())
    f = f.dropna(subset=["roll_mean_z", "roll_sd_z"]).copy()
    X = f[["roll_mean_z", "roll_sd_z"]].values
    model = IsolationForest(contamination=contamination, random_state=0).fit(X)
    f["anomaly"] = model.predict(X) == -1
    return f


def plot_lj(flagged: pd.DataFrame, branch: str, analyte: str, level: str, path: Path) -> None:
    g = flagged[(flagged.branch == branch) & (flagged.analyte == analyte) & (flagged.level == level)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g["date"], g["z"], marker="o", ms=3, lw=1, color="black")
    for k, ls in [(0, "-"), (1, ":"), (2, "--"), (3, "-.")]:
        for sign in ([1] if k == 0 else [1, -1]):
            ax.axhline(sign * k, color="grey", ls=ls, lw=0.8)
    rej = g[g["reject"]]
    ax.scatter(rej["date"], rej["z"], color="red", zorder=5, label="Westgard reject")
    ax.set_title(f"Levey-Jennings: {branch} | {analyte} | {level}")
    ax.set_ylabel("z-score (SD from target mean)")
    ax.set_ylim(-4.5, 4.5)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Westgard multirule QC and drift detection")
    ap.add_argument("--input", help="CSV with date,branch,analyte,level,target_mean,target_sd,value. Omit to use simulated data.")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    df = pd.read_csv(args.input, parse_dates=["date"]) if args.input else simulate()
    if not args.input:
        Path("data").mkdir(exist_ok=True)
        df.to_csv("data/simulated_qc.csv", index=False)
    flagged = run_rules(df)
    flagged.to_csv(out / "qc_flagged.csv", index=False)
    s = summary(flagged); s.to_csv(out / "branch_analyte_summary.csv", index=False)
    d = drift_anomalies(flagged); d[d["anomaly"]].to_csv(out / "drift_anomalies.csv", index=False)
    worst = s.head(3)[["branch", "analyte"]].values.tolist()
    for b, a in worst:
        plot_lj(flagged, b, a, "L1", out / f"lj_{b}_{a}_L1.png")
    print(s.to_string(index=False))
    print(f"\nDrift anomaly windows flagged: {int(d['anomaly'].sum())} of {len(d)}")
    print("Charts written for:", worst)


if __name__ == "__main__":
    main()
