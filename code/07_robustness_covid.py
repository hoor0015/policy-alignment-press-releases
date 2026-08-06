# -*- coding: utf-8 -*-
"""Step 7 (Online Appendix E). Robustness check including COVID-19 press releases.

This script reruns the full pipeline while KEEPING the 447 COVID-19 daily
surveillance bulletins that are excluded from the main analysis (steps 1-3).
All 51,269 cleaned press releases enter the re-estimation of the Word2Vec /
Doc2Vec embeddings. The document-type coding then classifies 66 of the COVID-19
releases as minister or vice-minister documents (these enter the regressions,
N = 32,888) and 381 as general documents (excluded from the regressions).

Input : data/raw/press_releases/{lee,park,moon}.xlsx
        data/raw/policy_commitments/{lee1,park1,moon1}.xlsx
Output: models/skku1201.bin, models/skku1201d.bin (COVID-inclusive embeddings)
        output/robustness_* tables (Appendix Tables E1, E2, E3, E4)

Environment note: requires konlpy + Mecab-ko (see step 1) and retrains the
embeddings, which is not bit-reproducible with workers=4. The exact models used
for the published Appendix E results are distributed as models/skku1201.bin and
models/skku1201d.bin.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Mecab
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RAW_PRESS = ROOT / "data" / "raw" / "press_releases"
RAW_COMMIT = ROOT / "data" / "raw" / "policy_commitments"
MODELS = ROOT / "models"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Clean and tokenize, KEEPING COVID-19 surveillance bulletins
# ---------------------------------------------------------------------------
df = pd.concat([pd.read_excel(f) for f in sorted(RAW_PRESS.glob("*.xlsx"))],
               ignore_index=True)
df = df.dropna(subset=["제목", "내용"])
print(f"Corpus including COVID-19 bulletins: {len(df)}")

mecab = Mecab()
stopwords = ["이번", "관련", "경우", "우리", "이상"]


def preprocess(text):
    text = re.sub(r"[\s+|\(|\)]", " ", text)
    words = re.split(r"\s", text)
    filtered = " ".join(w for w in words if not re.search(r"[^가-힣]", w))
    return [t for t in mecab.nouns(filtered) if len(t) > 1 and t not in stopwords]


df["content"] = df["내용"].apply(preprocess)


def categorize_content(content):
    if "장관" in content and "차관" in content:
        return 1 if content.index("장관") < content.index("차관") else 2
    elif "장관" in content:
        return 1
    elif "차관" in content:
        return 2
    return 0


df["분석대상"] = df["내용"].apply(categorize_content)

# ---------------------------------------------------------------------------
# 2. Re-train embeddings on the COVID-inclusive corpus
# ---------------------------------------------------------------------------
w2v = Word2Vec(df["content"].tolist(), window=5, vector_size=100, min_count=5,
               sg=1, seed=42, workers=4, min_alpha=0.0001, negative=3)
w2v.save(str(MODELS / "skku1201.bin"))

tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["content"])]
d2v = Doc2Vec(window=5, vector_size=100, min_count=5, seed=42, workers=4)
d2v.build_vocab(tagged)
d2v.wv = w2v.wv
d2v.train(tagged, total_examples=d2v.corpus_count, epochs=10)
d2v.save(str(MODELS / "skku1201d.bin"))

# ---------------------------------------------------------------------------
# 3. Policy commitments and similarity measure (as in step 3)
# ---------------------------------------------------------------------------
df2 = pd.concat([pd.read_excel(f) for f in sorted(RAW_COMMIT.glob("*.xlsx"))],
                ignore_index=True)
for col in ["정권", "부처", "국정과제"]:
    df2[col] = df2[col].ffill()
text_fields = ["세부실천과제", "과제개요", "주요 추진계획", "과제목표", "주요내용", "기대효과"]
df2["merged"] = sum((df2[c].fillna("").astype(str) for c in text_fields[1:]),
                    df2[text_fields[0]].fillna("").astype(str))
df3 = (df2.groupby(["정권", "부처", "국정과제"])["merged"]
       .apply("".join).reset_index())
df3["정권"] = df3["정권"].replace("정부", "", regex=True)

name_map = {"농림축산식품부": "농림부", "농림수산식품부": "농림부",
            "교육과학기술부": "교육부", "보건복지가족부": "보건복지부",
            "여성가족부": "여가부", "여성부": "여가부"}
df3["부처"] = df3["부처"].replace(name_map)
df["부처"] = df["부처"].replace(name_map)

df3["promise"] = df3["merged"].apply(preprocess)
df3["Inferred_Vectors"] = df3["promise"].apply(
    lambda words: d2v.infer_vector(words, alpha=0.025, min_alpha=0.025))
df["Inferred_Vectors"] = [d2v.dv[i] for i in range(len(df))]


def avg_similarity(row):
    match = df3[(df3["정권"] == row["정권"]) & (df3["부처"] == row["부처"])]
    if match.empty:
        return np.nan
    sims = cosine_similarity([row["Inferred_Vectors"]],
                             np.stack(match["Inferred_Vectors"].values))
    return sims.mean()


df["avg_similarity"] = df.apply(avg_similarity, axis=1)

acronym = {"농림부": "AFF", "교육부": "EDU", "국방부": "DEF", "보건복지부": "HSW",
           "여가부": "GEF", "기획재정부": "MEF"}
df["ministry"] = df["부처"].replace(acronym)
df["type"] = df["분석대상"].replace({1: "minister", 2: "vice minister", 0: "general"})

d = df[df["type"] != "general"].copy()
print(f"Robustness regression sample: N = {len(d)}")

# ---------------------------------------------------------------------------
# 4. Appendix Tables E1-E4
# ---------------------------------------------------------------------------
mE1 = ols("avg_similarity ~ C(type)", data=d).fit()
print(mE1.summary())
mE1.summary2().tables[1].to_csv(OUT / "robustness_tableE1.csv", encoding="utf-8-sig")

mE2 = ols("avg_similarity ~ C(type) + C(ministry)", data=d).fit()
print(mE2.summary())
mE2.summary2().tables[1].to_csv(OUT / "robustness_tableE2.csv", encoding="utf-8-sig")

mE3 = ols("avg_similarity ~ 0 + C(ministry):C(type)", data=d).fit(cov_type="HC1")
print(mE3.summary())
mE3.summary2().tables[1].to_csv(OUT / "robustness_tableE3.csv", encoding="utf-8-sig")

rows = []
for g in sorted(d["ministry"].unique()):
    p1 = [n for n in mE3.params.index if g in n and "minister" in n
          and "vice" not in n][0]
    p2 = [n for n in mE3.params.index if g in n and "vice minister" in n][0]
    test = mE3.t_test(f"{p1} - {p2} = 0")
    diff = float(np.squeeze(test.effect))
    se = float(np.squeeze(test.sd))
    crit = stats.t.ppf(0.975, df=mE3.df_resid)
    rows.append({"ministry": g, "difference": diff, "se": se,
                 "ci_low": diff - crit * se, "ci_high": diff + crit * se,
                 "p_value": float(np.squeeze(test.pvalue))})
e4 = pd.DataFrame(rows)
e4["p_holm"] = multipletests(e4["p_value"], method="holm")[1]
e4.to_csv(OUT / "robustness_tableE4_contrasts.csv", index=False, encoding="utf-8-sig")
print(e4.round(4))
