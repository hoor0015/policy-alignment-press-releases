# -*- coding: utf-8 -*-
"""Step 1. Build the tokenized press-release corpus from the raw data.

Input : data/raw/press_releases/{lee,park,moon}.xlsx
         (51,389 ministry press releases, 2008-2022)
Output: data/processed/press_releases_tokenized.parquet
         (50,822 documents with tokenized nouns and document-type coding)

Cleaning rules
  1. Drop records whose title or body is missing (120 records).
  2. Drop COVID-19 daily surveillance bulletins, identified by the phrase
     "코로나바이러스감염증" in the title (447 records). COVID-19 policy press
     releases remain in the corpus.

Tokenization
  - Keep whitespace-delimited words consisting only of Hangul syllables.
  - Extract nouns with the Mecab-ko morphological analyzer.
  - Keep nouns of two or more characters; drop the five generic stopwords
    ['이번', '관련', '경우', '우리', '이상'].

Document-type coding (분석대상)
  1 = minister document      ("장관" appears before "차관", or alone)
  2 = vice-minister document ("차관" appears before "장관", or alone)
  0 = general document       (neither term appears)

Environment note: the original analysis ran in Google Colab with Mecab-ko
installed via https://github.com/SOMJANG/Mecab-ko-for-Google-Colab. This step
requires konlpy + Mecab-ko; all downstream steps (04-06) can be run from the
distributed processed data without this dependency.
"""
import re
from pathlib import Path

import pandas as pd
from konlpy.tag import Mecab

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "press_releases"
OUT = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Load and clean the raw press releases
# ---------------------------------------------------------------------------
df = pd.concat([pd.read_excel(f) for f in sorted(RAW.glob("*.xlsx"))],
               ignore_index=True)
print(f"Raw records: {len(df)}")

df = df.dropna(subset=["제목", "내용"])
print(f"After dropping missing title/body: {len(df)}")

df = df[~df["제목"].str.contains("코로나바이러스감염증")]
print(f"After dropping COVID-19 surveillance bulletins: {len(df)}")

# ---------------------------------------------------------------------------
# Tokenize: Hangul-only words -> Mecab nouns (length >= 2, minus stopwords)
# ---------------------------------------------------------------------------
mecab = Mecab()
stopwords = ["이번", "관련", "경우", "우리", "이상"]


def preprocess(text):
    text = re.sub(r"[\s+|\(|\)]", " ", text)
    words = re.split(r"\s", text)
    filtered = " ".join(w for w in words if not re.search(r"[^가-힣]", w))
    return [t for t in mecab.nouns(filtered) if len(t) > 1 and t not in stopwords]


df["content"] = df["내용"].apply(preprocess)

# ---------------------------------------------------------------------------
# Code the document type from the order of "장관" / "차관" in the body text
# ---------------------------------------------------------------------------
def categorize_content(content):
    if "장관" in content and "차관" in content:
        return 1 if content.index("장관") < content.index("차관") else 2
    elif "장관" in content:
        return 1
    elif "차관" in content:
        return 2
    return 0


df["분석대상"] = df["내용"].apply(categorize_content)
print(df["분석대상"].value_counts())

df.to_parquet(OUT / "press_releases_tokenized.parquet", index=False)
print(f"Saved {OUT / 'press_releases_tokenized.parquet'} ({len(df)} documents)")
