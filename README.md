# Lab QC Westgard Monitor

Westgard multirule internal quality control (IQC) evaluation with cross-branch drift detection, built for multi-site clinical laboratory networks working towards ISO 15189:2022.

**Data:** all data in this repository is synthetic. No patient, clinic or employer data is used. The simulator plants three known faults so the methods can be verified.

## What it does
1. **Westgard multirule engine** (`src/labqc/westgard.py`): 1_2s (warning), 1_3s, 2_2s, R_4s, 4_1s and 10_x (reject), each unit tested.
2. **Runs every branch, analyte and control level** and produces a flagged run log, a branch/analyte summary and Levey-Jennings charts.
3. **Drift detection** (`analyse.py`): rolling mean z (bias) and rolling SD z (imprecision) scored with Isolation Forest, to surface problems that single-point rules detect late or not at all.

## Planted faults and what detected them
| Fault | Type | Westgard rejection rate | Drift detector |
|---|---|---|---|
| Branch_C Creatinine, gradual upward trend from day 30 | Systematic (trend) | 23.9% | Flagged |
| Branch_B Glucose, +1.5 SD shift from day 45 | Systematic (shift) | 22.8% | Flagged |
| Branch_D ALT, doubled imprecision from day 60 | Random error | 5.6% | Most flagged windows of any series |

Takeaway: the rules catch bias quickly, but increased imprecision produces few rule violations. A rolling-SD monitor makes it visible. The three planted series were the top three for drift anomalies.

## Run it
```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m labqc.analyse                      # simulated data
PYTHONPATH=src python -m labqc.analyse --input my_qc.csv    # your own data
```
Input CSV columns: `date, branch, analyte, level, target_mean, target_sd, value`.

## Limitations
- Evaluates one control level at a time. Across-level rules (for example R_4s between L1 and L2 in the same run) are a planned extension.
- Targets and SDs are assumed fixed. Real use should take them from manufacturer or established laboratory values.
- Isolation Forest thresholds are illustrative and need tuning against real data. The tool supports QC review and does not replace a laboratory director's judgement.

## Roadmap
Across-level rules, sigma-metric calculation from EQA bias and CV, turnaround-time pattern analysis, Power BI export.
