# Codebook

Variable definitions, coding schemes, and transformations for all datasets in
this repository. Column names are kept exactly as they appear in the files
(Korean names are original; English glosses in parentheses).

## 1. Raw data

### 1.1 `data/raw/press_releases/{lee,park,moon}.xlsx`

Government press releases issued by six South Korean ministries during the
Lee Myung-bak (2008-2013), Park Geun-hye (2013-2017), and Moon Jae-in
(2017-2022) administrations, collected from the ministries' official press
release archives. 51,389 records in total (lee 16,582; park 17,368; moon
17,429).

| Column | Description |
|---|---|
| `연도` | Year of release |
| `월` | Month of release |
| `일` | Day of release |
| `정권` | Administration (이명박 / 박근혜 / 문재인) |
| `부처` | Issuing ministry, as named at the time of release |
| `제목` | Press release title |
| `내용` | Press release body text |

The six ministries and their acronyms used in the paper:

| Acronym | Ministry | Prestige type |
|---|---|---|
| MEF | Ministry of Economy and Finance (기획재정부) | High prestige |
| DEF | Ministry of National Defense (국방부) | High prestige |
| HSW | Ministry of Health and Social Welfare (보건복지부) | Political leverage |
| EDU | Ministry of Education (교육부) | Political leverage |
| GEF | Ministry of Gender Equality and Family (여가부) | Low profile |
| AFF | Ministry of Agriculture, Forestry, and Food (농림부) | Low profile |

Ministry names that changed across administrations are harmonized:
농림축산식품부, 농림수산식품부 → 농림부; 교육과학기술부 → 교육부;
보건복지가족부 → 보건복지부; 여성가족부, 여성부 → 여가부.

### 1.2 `data/raw/policy_commitments/{lee1,park1,moon1}.xlsx`

Policy commitments (국정과제) formulated by each presidential transition
committee before inauguration, restricted to commitments assigned to the six
ministries above. Merged cells in the source documents appear as blanks and
are forward-filled during processing.

| Column | Description |
|---|---|
| `정권` | Administration (이명박정부 / 박근혜정부 / 문재인정부) |
| `부처` | Ministry in charge |
| `국정과제` | Policy commitment title |
| `세부실천과제` | Detailed implementation task |
| `과제개요` | Commitment overview |
| `주요 추진계획` | Main implementation plan |
| `과제목표` | Commitment objective |
| `주요내용` | Main content |
| `기대효과` | Expected effect |

Not every text field exists in every file; the six free-text fields are
concatenated (missing fields as empty strings) into one text per commitment.
After grouping by administration x ministry x commitment there are 170
commitment documents (Lee 92, Park 50, Moon 28).

## 2. Processed data

### 2.1 `data/processed/press_releases_tokenized.parquet`

The cleaned and tokenized corpus (50,822 documents) produced by
`code/01_build_corpus.py`. Transformations applied to the raw press releases:

1. Drop records with missing `제목` or `내용` (120 records).
2. Drop COVID-19 daily surveillance bulletins, identified by the phrase
   "코로나바이러스감염증" in `제목` (447 records). COVID-19 policy press
   releases are retained.
3. Tokenize `내용`: keep whitespace-delimited words consisting only of Hangul
   syllables; extract nouns with Mecab-ko; keep nouns of length >= 2; remove
   stopwords ['이번', '관련', '경우', '우리', '이상'].

Additional columns:

| Column | Description |
|---|---|
| `content` | List of extracted noun tokens |
| `분석대상` | Document-type code (see below) |

Document-type coding (`분석대상`), applied to the body text `내용`:

| Code | Meaning | Rule |
|---|---|---|
| 1 | Minister document | "장관" appears before "차관", or only "장관" appears |
| 2 | Vice-minister document | "차관" appears before "장관", or only "차관" appears |
| 0 | General document | Neither term appears |

Resulting counts: 28,544 minister documents, 4,278 vice-minister documents,
18,000 general documents.

## 3. Analysis data

### 3.1 `data/analysis/press_releases_analysis.parquet`

The dataset used for all reported tables and figures (50,822 rows), produced
by `code/03_build_analysis_dataset.py`. Columns in addition to those of the
tokenized corpus:

| Column | Description |
|---|---|
| `Inferred_Vectors` | 100-dimensional Doc2Vec document vector (list of float32) |
| `avg_similarity` | Dependent variable; see formula below |
| `ministry` | Ministry acronym (MEF/DEF/HSW/EDU/GEF/AFF) |
| `regime` | Administration acronym (LEE/PARK/MOON) |
| `type` | Document type label (minister / vice minister / general) |

`avg_similarity` for press release *i* in administration *r* and ministry *m*
is the mean of cosine similarities between the press-release vector A_rmi and
each policy-commitment vector B_rmj of the same administration and ministry:

    avg_similarity(i) = (1/k) * sum_j cos(A_rmi, B_rmj),  j = 1..k

where k is the number of policy commitments for that administration-ministry
pair. Regressions use the 32,822 minister / vice-minister documents
(`type != 'general'`).

The ministry-type classification used in Table A1 and Hypothesis 2:
High-Prestige = {MEF, DEF}; Political-Leverage = {HSW, EDU};
Low-Profile = {GEF, AFF}.

### 3.2 `data/analysis/policy_commitments_with_vectors.parquet` (and `.xlsx`)

The 170 policy-commitment documents with their inferred vectors.

| Column | Description |
|---|---|
| `정권` | Administration (이명박 / 박근혜 / 문재인; suffix "정부" removed) |
| `부처` | Ministry (harmonized names) |
| `국정과제` | Policy commitment title |
| `merged` | Concatenated commitment text |
| `promise` | List of extracted noun tokens (same pipeline as press releases) |
| `Inferred_Vectors` | 100-dimensional Doc2Vec inferred vector |

The `.xlsx` file is the original stored version (vectors serialized as
strings); the `.parquet` file contains the same data with parsed vectors.

## 4. Models

| File | Description |
|---|---|
| `models/skku1111.bin` | Word2Vec trained on the 50,822-document corpus (100 dims, window 5, Skip-gram, min_count 5, negative 3, seed 42) |
| `models/skku1111d.bin` | Doc2Vec trained on the same corpus with word vectors replaced by the Word2Vec vectors (10 epochs) |
| `models/skku1201.bin` | Word2Vec retrained on the COVID-inclusive corpus (51,269 documents) for Appendix E |
| `models/skku1201d.bin` | Doc2Vec counterpart for Appendix E |

Load with `gensim.models.word2vec.Word2Vec.load()` /
`gensim.models.doc2vec.Doc2Vec.load()`.

## 5. Keyword translation

Figures 3 and 4 label Korean keywords in English. The original analysis
translated keywords at run time with deep-translator (GoogleTranslator) plus a
manual override dictionary; the resulting labels are fixed in the `TRANSLATION`
dictionary in `code/04_descriptive_analysis.py`. Two Korean words (재정, 금융)
share the label "finance"; in the published Figure 4 the later entry overwrote
the earlier one in the plotted dictionary, and the replication code reproduces
this behavior.
