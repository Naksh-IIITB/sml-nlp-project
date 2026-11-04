from __future__ import annotations

from pathlib import Path

import joblib
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import MAX_BOW_FEATURES, MAX_TFIDF_FEATURES


def build_bow_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=MAX_BOW_FEATURES,
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 1),
    )


def build_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=MAX_TFIDF_FEATURES,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )


def fit_transform_cached(vectorizer, train_texts, test_texts, cache_prefix: Path):
    cache_prefix.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path = cache_prefix.with_suffix(".joblib")
    train_path = cache_prefix.with_name(cache_prefix.name + "_train.npz")
    test_path = cache_prefix.with_name(cache_prefix.name + "_test.npz")

    if vectorizer_path.exists() and train_path.exists() and test_path.exists():
        fitted_vectorizer = joblib.load(vectorizer_path)
        return fitted_vectorizer, load_npz(train_path), load_npz(test_path)

    train_matrix = vectorizer.fit_transform(train_texts)
    test_matrix = vectorizer.transform(test_texts)
    joblib.dump(vectorizer, vectorizer_path)
    save_npz(train_path, train_matrix)
    save_npz(test_path, test_matrix)
    return vectorizer, train_matrix, test_matrix
