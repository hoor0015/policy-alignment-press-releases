# -*- coding: utf-8 -*-
"""Step 2. Train the Word2Vec and Doc2Vec models on the press-release corpus.

Input : data/processed/press_releases_tokenized.parquet
Output: models/skku1111.bin   (Word2Vec)
        models/skku1111d.bin  (Doc2Vec)

Word2Vec parameters (Mikolov et al., 2013): 100-dimensional vectors, window
size 5, Skip-gram (sg=1), min_count 5, negative sampling 3, seed 42.

Doc2Vec: the document vectors are trained on the same corpus after replacing
the Doc2Vec word vectors with the trained Word2Vec vectors, so that document
embeddings aggregate (mean-pool) the Word2Vec semantic space (Le & Mikolov,
2014; Arora et al., 2017).

Reproducibility note: gensim training with workers=4 is not bit-reproducible
across runs (thread scheduling affects the result). The exact models used in
the published analysis are distributed in models/; rerun this script only if
you want to retrain from scratch.
"""
from pathlib import Path

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

df = pd.read_parquet(DATA / "press_releases_tokenized.parquet")
corpus = df["content"].tolist()

# ---------------------------------------------------------------------------
# Word2Vec
# ---------------------------------------------------------------------------
w2v = Word2Vec(corpus, window=5, vector_size=100, min_count=5, sg=1, seed=42,
               workers=4, min_alpha=0.0001, negative=3)
w2v.save(str(MODELS / "skku1111.bin"))
print(f"Word2Vec trained: vocabulary = {len(w2v.wv.index_to_key)} words")

# ---------------------------------------------------------------------------
# Doc2Vec initialized with the Word2Vec word vectors
# ---------------------------------------------------------------------------
tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
d2v = Doc2Vec(window=5, vector_size=100, min_count=5, seed=42, workers=4)
d2v.build_vocab(tagged)
d2v.wv = w2v.wv
d2v.train(tagged, total_examples=d2v.corpus_count, epochs=10)
d2v.save(str(MODELS / "skku1111d.bin"))
print(f"Doc2Vec trained: {len(d2v.dv)} document vectors")
