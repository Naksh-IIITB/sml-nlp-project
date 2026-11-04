from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / "results" / "cache" / "mpl"),
)

from config import RANDOM_STATE, get_paths
from data import load_sampled_dataset
from embeddings import FrozenEncoder
from evaluation import (
    build_classification_report,
    build_confusion,
    evaluate_predictions,
    results_to_frame,
)
from models import (
    build_classifiers,
    reduce_embedding_dimensions,
    reduce_tfidf_dimensions,
    tune_estimator,
)
from plots import (
    save_class_distribution,
    save_explained_variance,
    save_heatmap,
    save_model_comparison,
    save_text_length_distribution,
)
from sparse_features import build_bow_vectorizer, build_tfidf_vectorizer, fit_transform_cached


def ensure_dirs() -> None:
    paths = get_paths()
    for path in [
        paths.data_dir,
        paths.cache_dir,
        paths.figures_dir,
        paths.tables_dir,
        paths.models_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def run_feature_space(
    train_x,
    test_x,
    train_y,
    test_y,
    *,
    feature_space: str,
    representation: str,
) -> tuple[list, dict[str, np.ndarray]]:
    labels = pd.Series(train_y).unique()
    dense_features = not hasattr(train_x, "tocsr")
    models = build_classifiers(num_classes=len(labels), dense_features=dense_features)

    results = []
    predictions = {}
    for model_name, model in models.items():
        print(f"Tuning {feature_space}/{representation}/{model_name}...", flush=True)
        best_model, cv_macro_f1, best_params = tune_estimator(
            model_name,
            model,
            train_x,
            train_y,
            dense_features=dense_features,
            representation=representation,
        )
        pred = best_model.predict(test_x)
        predictions[model_name] = pred
        results.append(
            evaluate_predictions(
                test_y,
                pred,
                feature_space=feature_space,
                representation=representation,
                model_name=model_name,
                cv_macro_f1=cv_macro_f1,
                best_params=best_params,
            )
        )
    return results, predictions


def main() -> None:
    ensure_dirs()
    paths = get_paths()
    print("Loading dataset...", flush=True)

    train_df, test_df = load_sampled_dataset(paths.data_dir / "dataset.csv")
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    print("Creating EDA plots...", flush=True)
    save_class_distribution(full_df, paths.figures_dir / "class_distribution.png")
    save_text_length_distribution(full_df, paths.figures_dir / "text_length_distribution.png")

    print("Building cached sparse features...", flush=True)
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

    print("Loading cached embeddings...", flush=True)
    train_embedding_cache = paths.cache_dir / "train_embeddings.npz"
    test_embedding_cache = paths.cache_dir / "test_embeddings.npz"
    encoder = None
    if not train_embedding_cache.exists() or not test_embedding_cache.exists():
        encoder = FrozenEncoder()

    if encoder is None:
        from embeddings import encode_texts

        train_embeddings = encode_texts(
            train_df["text"].tolist(),
            train_embedding_cache,
        )
        test_embeddings = encode_texts(
            test_df["text"].tolist(),
            test_embedding_cache,
        )
    else:
        train_embeddings = encoder.encode_texts(
            train_df["text"].tolist(),
            train_embedding_cache,
        )
        test_embeddings = encoder.encode_texts(
            test_df["text"].tolist(),
            test_embedding_cache,
        )

    all_results = []

    print("Running sparse unigram experiments...", flush=True)
    sparse_results, sparse_predictions = run_feature_space(
        bow_train,
        bow_test,
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="sparse",
        representation="unigram_bow",
    )
    all_results.extend(sparse_results)

    print("Running sparse TF-IDF experiments...", flush=True)
    tfidf_results, tfidf_predictions = run_feature_space(
        tfidf_train,
        tfidf_test,
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="sparse",
        representation="bigram_tfidf",
    )
    all_results.extend(tfidf_results)

    print("Running contextual CLS experiments...", flush=True)
    cls_results, cls_predictions = run_feature_space(
        train_embeddings["cls"],
        test_embeddings["cls"],
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="contextual",
        representation="distilbert_cls",
    )
    all_results.extend(cls_results)

    print("Running contextual mean experiments...", flush=True)
    mean_results, mean_predictions = run_feature_space(
        train_embeddings["mean"],
        test_embeddings["mean"],
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="contextual",
        representation="distilbert_mean",
    )
    all_results.extend(mean_results)

    print("Saving comparison tables and plots...", flush=True)
    results_df = results_to_frame(all_results)
    results_df.to_csv(paths.tables_dir / "model_results.csv", index=False)
    save_model_comparison(results_df, paths.figures_dir / "model_comparison.png")

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

    for n_components in [10, 25, 50, 100]:
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        pca.fit(train_embeddings["mean"])
        variance_rows.append(
            {
                "feature_space": "embedding_pca",
                "n_components": n_components,
                "cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
            }
        )

    variance_df = pd.DataFrame(variance_rows)
    variance_df.to_csv(paths.tables_dir / "explained_variance.csv", index=False)
    save_explained_variance(variance_df, paths.figures_dir / "explained_variance.png")

    print("Running dimensionality reduction experiments...", flush=True)
    tfidf_reducer, tfidf_train_reduced, tfidf_test_reduced = reduce_tfidf_dimensions(
        tfidf_train, tfidf_test, n_components=300
    )
    embedding_reducer, emb_train_reduced, emb_test_reduced = reduce_embedding_dimensions(
        train_embeddings["mean"], test_embeddings["mean"], n_components=100
    )

    reduced_results = []
    reduced_tfidf_results, _ = run_feature_space(
        tfidf_train_reduced,
        tfidf_test_reduced,
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="reduced",
        representation="tfidf_svd_300",
    )
    reduced_results.extend(reduced_tfidf_results)

    reduced_embedding_results, _ = run_feature_space(
        emb_train_reduced,
        emb_test_reduced,
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="reduced",
        representation="embedding_pca_100",
    )
    reduced_results.extend(reduced_embedding_results)

    reduced_df = results_to_frame(reduced_results)
    reduced_df.to_csv(paths.tables_dir / "reduced_results.csv", index=False)

    best_sparse = (
        results_df.query("representation == 'bigram_tfidf'")
        .sort_values("macro_f1", ascending=False)
        .iloc[0]
    )
    best_contextual = (
        results_df.query("representation == 'distilbert_mean'")
        .sort_values("macro_f1", ascending=False)
        .iloc[0]
    )

    sparse_pred = tfidf_predictions[best_sparse["model_name"]]
    contextual_pred = mean_predictions[best_contextual["model_name"]]

    sparse_report = build_classification_report(
        test_df["label"], sparse_pred, label_names
    )
    contextual_report = build_classification_report(
        test_df["label"], contextual_pred, label_names
    )
    sparse_report.to_csv(paths.tables_dir / "best_sparse_report.csv")
    contextual_report.to_csv(paths.tables_dir / "best_contextual_report.csv")

    sparse_confusion = build_confusion(test_df["label"], sparse_pred, label_names)
    contextual_confusion = build_confusion(test_df["label"], contextual_pred, label_names)
    save_heatmap(
        sparse_confusion,
        paths.figures_dir / "best_sparse_confusion.png",
        title=f"Best Sparse Model: {best_sparse['model_name']}",
    )
    save_heatmap(
        contextual_confusion,
        paths.figures_dir / "best_contextual_confusion.png",
        title=f"Best Contextual Model: {best_contextual['model_name']}",
    )

    joblib.dump(tfidf_reducer, paths.models_dir / "tfidf_svd.joblib")
    joblib.dump(embedding_reducer, paths.models_dir / "embedding_pca.joblib")

    summary = pd.DataFrame(
        [
            {
                "best_sparse_model": best_sparse["model_name"],
                "best_sparse_macro_f1": best_sparse["macro_f1"],
                "best_contextual_model": best_contextual["model_name"],
                "best_contextual_macro_f1": best_contextual["macro_f1"],
            }
        ]
    )
    summary.to_csv(paths.tables_dir / "summary.csv", index=False)

    print("Experiment run complete.")
    print(f"Results saved to: {paths.root / 'results'}")


if __name__ == "__main__":
    main()
