from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from config import RANDOM_STATE, get_paths
from data import load_sampled_dataset
from evaluation import build_classification_report, build_confusion, results_to_frame
from models import reduce_tfidf_dimensions
from plots import (
    save_class_distribution,
    save_explained_variance,
    save_heatmap,
    save_model_comparison,
    save_text_length_distribution,
    save_tsne_projection,
)
from run_experiments import ensure_dirs, run_feature_space
from sparse_features import build_bow_vectorizer, build_tfidf_vectorizer, fit_transform_cached


def run_sparse_part() -> None:
    ensure_dirs()
    paths = get_paths()

    train_df, test_df = load_sampled_dataset(paths.data_dir / "dataset.csv")
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    save_class_distribution(full_df, paths.figures_dir / "class_distribution.png")
    save_text_length_distribution(full_df, paths.figures_dir / "text_length_distribution.png")

    bow, bow_train, bow_test = fit_transform_cached(
        build_bow_vectorizer(),
        train_df["text"],
        test_df["text"],
        paths.cache_dir / "bow_features",
    )
    tfidf, tfidf_train, tfidf_test = fit_transform_cached(
        build_tfidf_vectorizer(),
        train_df["text"],
        test_df["text"],
        paths.cache_dir / "tfidf_features",
    )

    joblib.dump(bow, paths.models_dir / "bow_vectorizer.joblib")
    joblib.dump(tfidf, paths.models_dir / "tfidf_vectorizer.joblib")

    all_results = []
    bow_results, _ = run_feature_space(
        bow_train,
        bow_test,
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="sparse",
        representation="unigram_bow",
    )
    all_results.extend(bow_results)

    tfidf_results, tfidf_predictions = run_feature_space(
        tfidf_train,
        tfidf_test,
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="sparse",
        representation="bigram_tfidf",
    )
    all_results.extend(tfidf_results)

    results_df = results_to_frame(all_results)
    results_df.to_csv(paths.tables_dir / "part1_sparse_results.csv", index=False)
    save_model_comparison(results_df, paths.figures_dir / "part1_sparse_model_comparison.png")

    variance_rows = []
    for n_components in [50, 100, 200, 300]:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        svd.fit(tfidf_train)
        variance_rows.append(
            {
                "feature_space": "tfidf_svd",
                "n_components": n_components,
                "cumulative_explained_variance": float(svd.explained_variance_ratio_.sum()),
            }
        )
    variance_df = pd.DataFrame(variance_rows)
    variance_df.to_csv(paths.tables_dir / "part1_sparse_explained_variance.csv", index=False)
    save_explained_variance(variance_df, paths.figures_dir / "part1_sparse_explained_variance.png")

    tfidf_reducer, tfidf_train_reduced, tfidf_test_reduced = reduce_tfidf_dimensions(
        tfidf_train, tfidf_test, n_components=300
    )
    save_tsne_projection(
        tfidf_test_reduced,
        test_df["label_name"].tolist(),
        paths.figures_dir / "part1_sparse_tfidf_tsne.png",
        title="t-SNE on TF-IDF SVD Features",
    )

    best_sparse = (
        results_df.query("representation == 'bigram_tfidf'")
        .sort_values("macro_f1", ascending=False)
        .iloc[0]
    )
    sparse_pred = tfidf_predictions[best_sparse["model_name"]]
    sparse_report = build_classification_report(test_df["label"], sparse_pred, label_names)
    sparse_report.to_csv(paths.tables_dir / "part1_sparse_best_report.csv")
    sparse_confusion = build_confusion(test_df["label"], sparse_pred, label_names)
    save_heatmap(
        sparse_confusion,
        paths.figures_dir / "part1_sparse_best_confusion.png",
        title=f"Part 1 Best Sparse Model: {best_sparse['model_name']}",
    )
