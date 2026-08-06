# Replication materials — "Analyzing Policy Alignment Using Government Press Releases: A Machine Learning Approach"

This repository contains the full raw datasets, the refined datasets used in
the analysis, variable definitions and coding schemes, the trained embedding
models, and the final analysis code and figures for the article. It is
intended to allow editors and referees to verify and fully scrutinize the
methods and findings.

## Repository structure

```
data/
  raw/
    press_releases/        lee.xlsx, park.xlsx, moon.xlsx  (51,389 raw press releases)
    policy_commitments/    lee1.xlsx, park1.xlsx, moon1.xlsx  (transition-committee policy commitments)
  processed/
    press_releases_tokenized.parquet   (cleaned + tokenized corpus, 50,822 documents)
  analysis/
    press_releases_analysis.parquet    (analysis dataset: vectors + avg_similarity)
    policy_commitments_with_vectors.parquet / .xlsx  (170 commitments with vectors)
models/
  skku1111.bin, skku1111d.bin    (Word2Vec / Doc2Vec used in the main analysis)
  skku1201.bin, skku1201d.bin    (COVID-inclusive models, Online Appendix E)
code/
  01_build_corpus.py             (raw -> tokenized corpus; requires Mecab-ko)
  02_train_embeddings.py         (Word2Vec / Doc2Vec training)
  03_build_analysis_dataset.py   (commitment processing, vectors, avg_similarity)
  04_descriptive_analysis.py     (Table 2, Figures 2-4)
  05_similarity_regressions.py   (Tables 3-6, Appendix A1, E4 contrasts, Figure 5)
  06_umap_figure.py              (Figure 6)
  07_robustness_covid.py         (Online Appendix E pipeline)
figures/                         (all main-text figures as 300-dpi PNG)
output/                          (tables written by the analysis scripts, CSV)
docs/
  codebook.md                    (variable definitions, coding schemes, transformations)
```

## Data

The corpus consists of 51,389 press releases issued by six South Korean
ministries (MEF, DEF, HSW, EDU, GEF, AFF) across the Lee Myung-bak, Park
Geun-hye, and Moon Jae-in administrations (2008-2022), and 170 policy
commitments formulated by the three presidential transition committees.
After cleaning (dropping 120 records with missing title or body and 447
COVID-19 daily surveillance bulletins), 50,822 documents remain, of which
28,544 are minister documents and 4,278 are vice-minister documents. See
`docs/codebook.md` for all variable definitions, the document-type coding
rule, and every transformation applied.

## Reproducing the results

Steps 04-06 reproduce every table and figure reported in the paper directly
from the distributed datasets and require only the packages in
`requirements.txt`. Run them from the repository root.

```
pip install -r requirements.txt
python code/04_descriptive_analysis.py    # Table 2, Figures 2-4
python code/05_similarity_regressions.py  # Tables 3-6, A1, E4 contrasts, Figure 5
python code/06_umap_figure.py             # Figure 6
```

Steps 01-03 and 07 document the upstream pipeline (corpus construction,
embedding training, similarity computation, and the Appendix E robustness
run). They additionally require the Mecab-ko morphological analyzer via
konlpy; the original runs used Google Colab with Mecab-ko installed from
https://github.com/SOMJANG/Mecab-ko-for-Google-Colab. Because gensim training
and vector inference with multiple workers are not bit-reproducible, the
exact trained models and inferred vectors used in the published analysis are
distributed in `models/` and `data/analysis/`, and the downstream scripts
(04-06) are fully deterministic given those files.

## Verification

Running steps 04-05 on the distributed analysis dataset reproduces the
reported results, including the Table 2 document frequencies, the Figure 2
legend statistics, the Table 3 estimates (vice minister -0.0037, p = 0.069),
the Table 4 estimates, the Table 5 ANOVA decomposition, all Table 6
group-specific means, the Appendix Table A1 means, and the within-ministry
contrasts with Holm-adjusted p-values.

## Figures

All main-text figures are provided in `figures/` as PNG files at 300 dpi.
Figure 6 (UMAP) is deterministic within this repository (random_state = 42);
UMAP layouts may rotate or shift across library versions while the cluster
structure is stable.

## Software

Analysis was conducted in Python. The versions used for verification are
pinned in `requirements.txt` (Python 3.12; pandas 3.0.2, statsmodels 0.14.6,
scikit-learn 1.8.0, gensim 4.4.0, umap-learn 0.5.12, seaborn 0.13.2).

## Contact

For questions about the data or code, please contact the corresponding
author (kdkim@kw.ac.kr).
