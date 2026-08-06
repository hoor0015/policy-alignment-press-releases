# -*- coding: utf-8 -*-
"""Similarity regressions and Figure 5.

Reproduces, from data/analysis/press_releases_analysis.parquet:
  - Table 3  OLS without ministry controls
  - Table 4  OLS with ministry controls
  - Table 5  Two-way ANOVA
  - Table 6  Group-specific means (no-intercept fully one-hot model, HC1)
  - Appendix Table A1  Group-specific means at the ministry-type level (HC1)
  - Appendix Table E4 specification  Within-ministry contrasts with Holm correction
  - Figure 5  Mean cosine similarity by ministry and document type

All tables are written to output/ as CSV; Figure 5 is written to figures/ at 300 dpi.

Usage: python code/05_similarity_regressions.py   (run from the repository root)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "analysis"
OUT = ROOT / "output"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load the analysis dataset and keep minister / vice-minister documents
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA / "press_releases_analysis.parquet")
d = df[df["type"] != "general"].copy()
d["ministry"] = d["ministry"].astype(str).str.strip()
d["type"] = d["type"].astype(str).str.strip()
print(f"Analysis sample: N = {len(d)} "
      f"(minister = {(d['type'] == 'minister').sum()}, "
      f"vice minister = {(d['type'] == 'vice minister').sum()})")
print(f"Overall mean similarity: {d['avg_similarity'].mean():.4f}")
print(f"Minister mean: {d.loc[d['type'] == 'minister', 'avg_similarity'].mean():.4f}, "
      f"Vice-minister mean: {d.loc[d['type'] == 'vice minister', 'avg_similarity'].mean():.4f}")

# ---------------------------------------------------------------------------
# Table 3. Similarity analysis without ministry controls
# ---------------------------------------------------------------------------
m3 = ols("avg_similarity ~ C(type)", data=d).fit()
print("\n===== Table 3 =====")
print(m3.summary())
m3.summary2().tables[1].to_csv(OUT / "table3_ols_no_ministry_controls.csv", encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Table 4. Similarity analysis with ministry controls
# (baseline: minister, Ministry of Agriculture, Forestry, and Food)
# ---------------------------------------------------------------------------
m4 = ols("avg_similarity ~ C(type) + C(ministry)", data=d).fit()
print("\n===== Table 4 =====")
print(m4.summary())
m4.summary2().tables[1].to_csv(OUT / "table4_ols_with_ministry_controls.csv", encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Table 5. Two-way ANOVA
# ---------------------------------------------------------------------------
m5 = ols("avg_similarity ~ C(type) * C(ministry)", data=d).fit()
anova = sm.stats.anova_lm(m5, typ=2)
print("\n===== Table 5 =====")
print(anova)
anova.to_csv(OUT / "table5_two_way_anova.csv", encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# Table 6. Group-specific means: no-intercept fully one-hot model with
# HC1 robust standard errors, one coefficient per ministry x type cell
# ---------------------------------------------------------------------------
type_order = ["minister", "vice minister"]
ministry_order = sorted(d["ministry"].unique())
m6 = ols("avg_similarity ~ 0 + C(ministry):C(type)", data=d).fit(cov_type="HC1")
print("\n===== Table 6 =====")
print(m6.summary())


def parse_term(term):
    group = typ = None
    for part in term.split(":"):
        if part.startswith("C(type)"):
            typ = part.split("[")[-1].rstrip("]").replace("T.", "")
        else:
            group = part.split("[")[-1].rstrip("]").replace("T.", "")
    return group, typ


def group_mean_table(model):
    conf = model.conf_int()
    rows = []
    for term in model.params.index:
        g1, g2 = parse_term(term)
        rows.append({
            "group": g1, "type": g2,
            "coef": model.params[term], "se": model.bse[term],
            "ci_low": conf.loc[term, 0], "ci_high": conf.loc[term, 1],
            "p_value": model.pvalues[term],
        })
    return (pd.DataFrame(rows)
            .sort_values(["group", "type"])
            .reset_index(drop=True))


t6 = group_mean_table(m6)
t6.to_csv(OUT / "table6_group_means_by_ministry.csv", index=False, encoding="utf-8-sig")
print(t6.round(4))

# ---------------------------------------------------------------------------
# Within-ministry contrasts (Appendix Table E4 specification):
# minister minus vice minister within each ministry, Holm-adjusted p-values
# ---------------------------------------------------------------------------
def get_param_name(model, group, typ):
    candidates = [n for n in model.params.index if group in n and typ in n]
    if not candidates:
        raise ValueError(f"parameter not found: {group}, {typ}")
    return sorted(candidates, key=len)[0]


def contrast_table(model, groups):
    rows = []
    for g in groups:
        p1 = get_param_name(model, g, "minister")
        p2 = get_param_name(model, g, "vice minister")
        test = model.t_test(f"{p1} - {p2} = 0")
        diff = float(np.squeeze(test.effect))
        se = float(np.squeeze(test.sd))
        crit = stats.t.ppf(0.975, df=model.df_resid)
        rows.append({
            "group": g,
            "contrast": "minister - vice minister",
            "difference": diff, "se": se,
            "ci_low": diff - crit * se, "ci_high": diff + crit * se,
            "t": float(np.squeeze(test.tvalue)),
            "p_value": float(np.squeeze(test.pvalue)),
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_value"], method="holm")[1]
    return out


contrasts = contrast_table(m6, ministry_order)
contrasts.to_csv(OUT / "contrasts_within_ministry.csv", index=False, encoding="utf-8-sig")
print("\n===== Within-ministry contrasts (Table E4 specification) =====")
print(contrasts.round(4))

# Joint Wald test: all within-ministry differences equal to zero
constraints = [
    f"{get_param_name(m6, g, 'minister')} - {get_param_name(m6, g, 'vice minister')} = 0"
    for g in ministry_order
]
print("\nJoint Wald test (all within-ministry differences = 0):")
print(m6.wald_test(constraints, scalar=True))

# ---------------------------------------------------------------------------
# Appendix Table A1. Group-specific means at the ministry-type level
# ---------------------------------------------------------------------------
HIGH_PRESTIGE = {"MEF", "DEF"}
LOW_PROFILE = {"GEF", "AFF"}


def classify(ministry):
    if ministry in HIGH_PRESTIGE:
        return "High-Prestige"
    if ministry in LOW_PROFILE:
        return "Low-Profile"
    return "Political-Leverage"


d["ministry_type"] = d["ministry"].map(classify)
mA1 = ols("avg_similarity ~ 0 + C(ministry_type):C(type)", data=d).fit(cov_type="HC1")
print("\n===== Appendix Table A1 =====")
print(mA1.summary())
tA1 = group_mean_table(mA1)
tA1.to_csv(OUT / "tableA1_group_means_by_ministry_type.csv", index=False, encoding="utf-8-sig")
print(tA1.round(4))

contrasts_mt = contrast_table(mA1, ["High-Prestige", "Political-Leverage", "Low-Profile"])
contrasts_mt.to_csv(OUT / "contrasts_within_ministry_type.csv", index=False, encoding="utf-8-sig")
print(contrasts_mt.round(4))

# ---------------------------------------------------------------------------
# Figure 5. Mean cosine similarity by ministry and document type (300 dpi)
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
palette = {"minister": "#ff7f0e", "vice minister": "#2ca02c"}
ministry_display_order = ["AFF", "EDU", "MEF", "GEF", "HSW", "DEF"]

plt.figure(figsize=(10, 6))
sns.barplot(x="ministry", y="avg_similarity", hue="type", data=d,
            order=ministry_display_order, hue_order=type_order,
            palette=palette, seed=42)
plt.title("Mean of Cosine Similarity by Ministry and Type")
plt.savefig(FIG / "Figure5_cosine_similarity_by_ministry.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved {FIG / 'Figure5_cosine_similarity_by_ministry.png'}")
