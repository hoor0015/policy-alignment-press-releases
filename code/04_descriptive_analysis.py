# -*- coding: utf-8 -*-
"""Descriptive analysis: Table 2 and Figures 2-4.

Reproduces, from data/analysis/press_releases_analysis.parquet and
data/analysis/policy_commitments_with_vectors.parquet:
  - Table 2   Frequency of documents, by administration, document and ministry type
  - Figure 2  Word frequency per document for ministers and vice ministers
  - Figure 3  Probability of word occurrence in minister and vice-minister documents
  - Figure 4  Top 20 keywords: frequency and TF-IDF

Table 2 is written to output/ as CSV; figures are written to figures/ at 300 dpi.

Korean-to-English keyword labels: the original analysis translated keywords with
deep-translator (GoogleTranslator) plus a manual override dictionary at run time.
For reproducibility without a network dependency, the resulting translations are
fixed here in TRANSLATION, matching the labels in the published figures.

Usage: python code/04_descriptive_analysis.py   (run from the repository root)
"""
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def embed_caption(fig, title, note=None, wrap=100, note_wrap=130):
    """Embed the manuscript caption as the figure title and note in the image.

    Manuscript formatting: title in Times New Roman 12 pt bold; the note label
    ("Note:"/"Notes:") in italic with the note text in Times New Roman 10 pt.
    """
    fig.text(0.0, 1.02, textwrap.fill(title, wrap), ha="left", va="bottom",
             fontsize=12, fontweight="bold", family="Times New Roman")
    if note:
        label, _, body = note.partition(":")
        t1 = fig.text(0.0, -0.03, label + ":", ha="left", va="top",
                      fontsize=10, style="italic", family="Times New Roman")
        fig.canvas.draw()
        x_after = fig.transFigure.inverted().transform(
            (t1.get_window_extent().x1, 0))[0] + 0.004
        fig.text(x_after, -0.03, textwrap.fill(body.strip(), note_wrap),
                 ha="left", va="top", fontsize=10, family="Times New Roman")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "analysis"
OUT = ROOT / "output"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

df = pd.read_parquet(DATA / "press_releases_analysis.parquet")
commitments = pd.read_parquet(DATA / "policy_commitments_with_vectors.parquet")

TRANSLATION = {
    "지원": "support", "교육": "education", "사업": "project", "기관": "agency",
    "식품": "food", "장관": "minister", "지역": "region", "추진": "proceed",
    "정책": "policy", "확대": "enlargement", "사회": "society", "대상": "target",
    "여성": "female", "학교": "school", "강화": "reinforcement", "관리": "management",
    "정부": "government", "한국": "korea", "운영": "operation", "협력": "cooperation",
    "참여": "participation", "연구": "research", "분야": "field", "개발": "development",
    "평가": "evaluation", "기술": "technology", "경제": "economy", "회의": "meeting",
    "재정": "finance", "금융": "finance", "시장": "market", "점검": "inspection",
    "기획": "plan", "방안": "countermeasure", "지속": "continuation",
    "대응": "reaction", "기업": "enterprise", "복지": "welfare", "산업": "industry",
    "서비스": "service", "집행": "execution", "혁신": "innovation",
}

# ---------------------------------------------------------------------------
# Table 2. Frequency of documents by administration, document and ministry type
# ---------------------------------------------------------------------------
ministry_cols = ["MEF", "DEF", "HSW", "EDU", "GEF", "AFF"]
admin_order = ["LEE", "PARK", "MOON"]

blocks = {}
blocks["Total"] = pd.crosstab(df["regime"], df["ministry"])
for doc_type in ["minister", "vice minister"]:
    sub = df[df["type"] == doc_type]
    blocks[doc_type] = pd.crosstab(sub["regime"], sub["ministry"])

acronym = {"기획재정부": "MEF", "국방부": "DEF", "보건복지부": "HSW", "교육부": "EDU",
           "여가부": "GEF", "농림부": "AFF", "이명박": "LEE", "박근혜": "PARK",
           "문재인": "MOON"}
c = commitments.copy()
c["ministry"] = c["부처"].map(acronym)
c["regime"] = c["정권"].str.strip().map(acronym)
blocks["policy commitment"] = pd.crosstab(c["regime"], c["ministry"])

rows = []
for label, block in blocks.items():
    block = block.reindex(index=admin_order, columns=ministry_cols, fill_value=0)
    block["Total"] = block.sum(axis=1)
    block.insert(0, "Document", label)
    rows.append(block)
table2 = pd.concat(rows)
table2.to_csv(OUT / "table2_document_frequencies.csv", encoding="utf-8-sig")
print("===== Table 2 =====")
print(table2)

# ---------------------------------------------------------------------------
# Figure 2. Word frequency per document by type (300 dpi)
# ---------------------------------------------------------------------------
category_labels = {1: "Minister", 2: "Vice Minister"}
df["Word Frequency"] = df["content"].apply(len)
df["Type"] = df["분석대상"].map(category_labels)

legend_stats = df.groupby("Type")["Word Frequency"].agg(["mean", "std", "count", "median"])
legend_stats["label"] = legend_stats.apply(
    lambda x: f"{x.name} (N={x['count']:.1f}, μ={x['mean']:.2f}, "
              f"σ={x['std']:.2f}, median={x['median']:.2f})", axis=1)
print("\nFigure 2 legend statistics:")
print(legend_stats[["mean", "std", "count", "median"]])

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Word Frequency", hue="Type", element="step", fill=True,
             common_norm=False, stat="density", palette=["#2ca02c", "#ff7f0e"],
             hue_order=["Vice Minister", "Minister"])
plt.legend(title="Type", labels=legend_stats["label"])
plt.xlabel("Word Frequency per Document")
plt.ylabel("Density")
plt.xlim(0, 500)
plt.tight_layout()
embed_caption(plt.gcf(),
              "Figure 2. Word Frequency Per Document for Ministers and Vice Ministers")
plt.savefig(FIG / "Figure2_word_frequency_histogram.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3. Probability of word occurrence (top 20 words overall) (300 dpi)
# ---------------------------------------------------------------------------
word_counts = {}
for category in df["분석대상"].unique():
    counts = Counter()
    for tokens in df.loc[df["분석대상"] == category, "content"]:
        counts.update(tokens)
    word_counts[category] = counts

all_counts = sum(word_counts.values(), Counter())
top_20_words = [w for w, _ in all_counts.most_common(20)]

top20 = {}
for category in [1, 2]:
    total = sum(word_counts[category].values())
    top20[category] = {TRANSLATION[w]: word_counts[category][w] * 100 / total
                       for w in top_20_words}

data = []
labels = {1: "Minister", 2: "Vice Minister"}
for category in [1, 2]:
    for word, percentage in top20[category].items():
        data.append({"Category": labels[category], "Word": word, "Percentage": percentage})
df_words = pd.DataFrame(data)

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x="Word", y="Percentage", hue="Category", data=df_words,
            palette=["#ff7f0e", "#2ca02c"])
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Percentage")
plt.legend(title="Category")
plt.tight_layout()
embed_caption(plt.gcf(),
              "Figure 3. Probability of Word Occurrence in Minister and "
              "Vice-Minister Documents")
plt.savefig(FIG / "Figure3_word_occurrence_probability.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 4. Top 20 keywords by frequency and TF-IDF (300 dpi)
# ---------------------------------------------------------------------------
stopwords = ['교육', '식품', '여성', '학교', '한국', '청소년', '가족', '대학', '환자',
             '장관', '차관', '재정부', '농업', '국방부', '학생', '의료', '국방']
df["content_str"] = df["content"].apply(" ".join)

def top_keywords(vectorizer_cls):
    result = {}
    for category in ["minister", "vice minister"]:
        vec = vectorizer_cls(stop_words=stopwords)
        X = vec.fit_transform(df.loc[df["type"] == category, "content_str"])
        sums = dict(zip(vec.get_feature_names_out(), X.toarray().sum(axis=0)))
        top = sorted(sums.items(), key=lambda kv: kv[1], reverse=True)[:20]
        # Duplicate English labels overwrite earlier entries, as in the original
        # run where two Korean words shared one translation (재정/금융 -> finance).
        result[category] = {TRANSLATION[w]: v for w, v in top}
    return result

top_freq = top_keywords(CountVectorizer)
top_tfidf = top_keywords(TfidfVectorizer)

def as_frame(top_dict):
    rows = [{"Category": cat, "Word": w, "Frequency": v}
            for cat, words in top_dict.items() for w, v in words.items()]
    return pd.DataFrame(rows)

df_plot1, df_plot2 = as_frame(top_freq), as_frame(top_tfidf)

base_colors = ["#ff7f0e", "#2ca02c"]

def color_gradient(base_color, n):
    return sns.light_palette(base_color, n_colors=n)[::-1]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for i, category in enumerate(["minister", "vice minister"]):
    for j, (frame, ylabel) in enumerate([(df_plot1, "Frequency"), (df_plot2, "TF-IDF")]):
        sub = frame[frame["Category"] == category].sort_values("Frequency", ascending=False)
        sns.barplot(x="Word", y="Frequency", data=sub, ax=axes[i][j],
                    palette=color_gradient(base_colors[i], len(sub)))
        axes[i][j].set_title(f"Top Keywords for {category}")
        axes[i][j].tick_params(axis="x", rotation=45)
        axes[i][j].set_xlabel("Words")
        axes[i][j].set_ylabel(ylabel)
plt.tight_layout()
embed_caption(fig,
              "Figure 4. Top 20 Keywords in Minister and Vice-Minister Documents: "
              "Frequency and TF-IDF", wrap=190)
plt.savefig(FIG / "Figure4_top_keywords_frequency_tfidf.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved Figures 2-4 to", FIG)
