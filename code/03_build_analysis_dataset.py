# -*- coding: utf-8 -*-
"""Step 3. Build the analysis dataset: policy commitments, document vectors,
and the average cosine similarity measure.

Input : data/raw/policy_commitments/{lee1,park1,moon1}.xlsx
        data/processed/press_releases_tokenized.parquet
        models/skku1111d.bin
Output: data/analysis/policy_commitments_with_vectors.parquet
        data/analysis/press_releases_analysis.parquet

Policy commitments
  The raw files contain the policy commitments (국정과제) announced by each
  presidential transition committee, restricted to the six ministries studied.
  The free-text fields describing each commitment (세부실천과제, 과제개요,
  주요 추진계획, 과제목표, 주요내용, 기대효과) are concatenated and grouped by
  administration x ministry x commitment, producing 170 commitment documents.
  Ministry names are harmonized across administrations (e.g., 농림축산식품부
  and 농림수산식품부 -> 농림부).

Vectors
  Each commitment text is tokenized with the same Mecab pipeline as step 1
  (the tokens are stored in the 'promise' column) and converted to a
  100-dimensional vector with Doc2Vec.infer_vector(alpha=0.025,
  min_alpha=0.025). Press-release vectors are the trained Doc2Vec document
  vectors.

Average cosine similarity (dependent variable)
  For each minister / vice-minister press release, avg_similarity is the mean
  of the cosine similarities between the press-release vector and every
  policy-commitment vector of the same administration and ministry.

Environment note: tokenizing the commitment texts requires konlpy + Mecab-ko
(see step 1). Vector inference is stochastic in gensim; the exact vectors used
in the published analysis are stored in the distributed analysis datasets.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from konlpy.tag import Mecab
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "policy_commitments"
PROCESSED = ROOT / "data" / "processed"
ANALYSIS = ROOT / "data" / "analysis"
MODELS = ROOT / "models"

# ---------------------------------------------------------------------------
# Policy commitments: merge, group, harmonize ministry names
# ---------------------------------------------------------------------------
df2 = pd.concat([pd.read_excel(f) for f in sorted(RAW.glob("*.xlsx"))],
                ignore_index=True)
for col in ["정권", "부처", "국정과제"]:
    df2[col] = df2[col].ffill()

text_fields = ["세부실천과제", "과제개요", "주요 추진계획", "과제목표", "주요내용", "기대효과"]
df2["merged"] = sum((df2[c].fillna("").astype(str) for c in text_fields[1:]),
                    df2[text_fields[0]].fillna("").astype(str))

df3 = (df2.groupby(["정권", "부처", "국정과제"])["merged"]
       .apply("".join).reset_index())
df3["정권"] = df3["정권"].replace("정부", "", regex=True)
df3["부처"] = df3["부처"].replace({
    "농림축산식품부": "농림부", "농림수산식품부": "농림부",
    "교육과학기술부": "교육부", "보건복지가족부": "보건복지부",
    "여성가족부": "여가부", "여성부": "여가부",
})
print(f"Policy commitments: {len(df3)}")

# ---------------------------------------------------------------------------
# Tokenize commitment texts (same pipeline as step 1)
# ---------------------------------------------------------------------------
mecab = Mecab()
stopwords = ["이번", "관련", "경우", "우리", "이상"]


def preprocess(text):
    text = re.sub(r"[\s+|\(|\)]", " ", text)
    words = re.split(r"\s", text)
    filtered = " ".join(w for w in words if not re.search(r"[^가-힣]", w))
    return [t for t in mecab.nouns(filtered) if len(t) > 1 and t not in stopwords]


df3["promise"] = df3["merged"].apply(preprocess)

# ---------------------------------------------------------------------------
# Vectors: infer commitment vectors, read press-release document vectors
# ---------------------------------------------------------------------------
d2v = Doc2Vec.load(str(MODELS / "skku1111d.bin"))
df3["Inferred_Vectors"] = df3["promise"].apply(
    lambda words: d2v.infer_vector(words, alpha=0.025, min_alpha=0.025))
df3.to_parquet(ANALYSIS / "policy_commitments_with_vectors.parquet", index=False)

df = pd.read_parquet(PROCESSED / "press_releases_tokenized.parquet")
df["Inferred_Vectors"] = [d2v.dv[i] for i in range(len(df))]

# ---------------------------------------------------------------------------
# Average cosine similarity per press release, within administration x ministry
# ---------------------------------------------------------------------------
name_map = {
    "농림축산식품부": "농림부", "농림수산식품부": "농림부",
    "교육과학기술부": "교육부", "보건복지가족부": "보건복지부",
    "여성가족부": "여가부", "여성부": "여가부",
}
df["부처"] = df["부처"].replace(name_map)


def avg_similarity(row):
    match = df3[(df3["정권"] == row["정권"]) & (df3["부처"] == row["부처"])]
    if match.empty:
        return np.nan
    sims = cosine_similarity([row["Inferred_Vectors"]],
                             np.stack(match["Inferred_Vectors"].values))
    return sims.mean()


df["avg_similarity"] = df.apply(avg_similarity, axis=1)

# English acronyms and labels used in the reported tables and figures
acronym = {"농림부": "AFF", "교육부": "EDU", "국방부": "DEF", "보건복지부": "HSW",
           "여가부": "GEF", "기획재정부": "MEF",
           "이명박": "LEE", "문재인": "MOON", "박근혜": "PARK"}
df["ministry"] = df["부처"].replace(acronym)
df["regime"] = df["정권"].replace(acronym)
df["type"] = df["분석대상"].replace({1: "minister", 2: "vice minister", 0: "general"})

df.to_parquet(ANALYSIS / "press_releases_analysis.parquet", index=False)
print(f"Analysis dataset saved: {len(df)} press releases "
      f"({(df['type'] != 'general').sum()} minister/vice-minister documents)")
