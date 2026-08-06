# -*- coding: utf-8 -*-
"""Figure 6: policy commitments and press release documents in UMAP space.

Projects the 100-dimensional Doc2Vec vectors of all minister / vice-minister
press releases and the 170 policy commitments into two dimensions with UMAP
(n_neighbors=30, min_dist=0.5), then plots a ministry-by-role facet grid.

Note on reproducibility: the published figure was generated with
np.random.seed(42) and UMAP's default (parallel, non-deterministic) mode, so
the exact point layout can differ slightly across runs and library versions
while the cluster structure is stable. This script passes random_state=42 to
UMAP so that repeated runs of this repository produce identical output.

Usage: python code/06_umap_figure.py   (run from the repository root)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "analysis"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

press = pd.read_parquet(DATA / "press_releases_analysis.parquet")
commitments = pd.read_parquet(DATA / "policy_commitments_with_vectors.parquet")

merged = pd.concat([press, commitments], axis=0).reset_index(drop=True)

X = np.stack(merged["Inferred_Vectors"].apply(np.asarray).values)
X_scaled = StandardScaler().fit_transform(X)

np.random.seed(42)
umap_model = umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

df_umap = pd.DataFrame({
    "x": X_umap[:, 0],
    "y": X_umap[:, 1],
    "type": merged["분석대상"],
    "regime": merged["정권"],
    "ministry": merged["부처"],
})

replace_dict = {"농림부": "AFF", "농림축산식품부": "AFF", "농림수산식품부": "AFF",
                "교육부": "EDU", "교육과학기술부": "EDU", "국방부": "DEF",
                "보건복지부": "HSW", "여가부": "GEF", "여성가족부": "GEF",
                "여성부": "GEF", "기획재정부": "MEF",
                "이명박": "LEE", "문재인": "MOON", "박근혜": "PARK"}
df_umap["ministry"] = df_umap["ministry"].replace(replace_dict)
df_umap["regime"] = df_umap["regime"].str.strip().replace(replace_dict)

# Press releases carry 분석대상 in {0, 1, 2}; policy commitments carry NaN.
sub1 = df_umap[df_umap["type"] != 2].copy()
sub1["category"] = "minister"
sub2 = df_umap[df_umap["type"] != 1].copy()
sub2["category"] = "vice minister"
sub3 = pd.concat([sub1, sub2], axis=0, ignore_index=True)

sub3["type"] = sub3["type"].map(
    lambda v: {1.0: "minister", 2.0: "vice minister", 0.0: np.nan}.get(v, "policy commitment"))
sub3 = sub3.dropna(subset=["type"])

palette = {"policy commitment": "red", "minister": "#ff7f0e", "vice minister": "#2ca02c"}
marker = {"policy commitment": "*", "vice minister": "o", "minister": "o"}
size_mapping = {"policy commitment": 50, "minister": 20, "vice minister": 20}
sub3["size"] = sub3["type"].map(size_mapping)

row_order = ["AFF", "EDU", "MEF", "GEF", "HSW", "DEF"]
g = sns.FacetGrid(sub3, col="category", row="ministry", row_order=row_order,
                  height=2, aspect=2, sharex=True, sharey=True)
g.map_dataframe(sns.scatterplot, x="x", y="y", hue="type", palette=palette,
                style="type", markers=marker, size=sub3["size"])
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set_axis_labels("UMAP Dimension 1", "UMAP Dimension 2")

legend_data = {k: v for k, v in g._legend_data.items()
               if k in ("policy commitment", "minister", "vice minister")}
g.add_legend(title="Type", legend_data=legend_data)

g.savefig(FIG / "Figure6_umap_space.png", dpi=300, bbox_inches="tight")
plt.close("all")
print("Saved", FIG / "Figure6_umap_space.png")
