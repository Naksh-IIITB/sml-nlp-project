from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA

from config import RANDOM_STATE, get_paths
from data import load_sampled_dataset
from embeddings import encode_texts
from encoder_registry import DEFAULT_ENCODER, safe_encoder_name
from evaluation import build_classification_report, build_confusion, results_to_frame
from models import reduce_embedding_dimensions
from part2_embeddings import build_embedding_cache_name
from plots import save_explained_variance, save_heatmap, save_model_comparison
from run_experiments import ensure_dirs, run_feature_space


def run_contextual_part(model_name: str = DEFAULT_ENCODER) -> dict[str, object]:
    ensure_dirs()
    paths = get_paths()
    encoder_slug = safe_encoder_name(model_name)
    train_df, test_df = load_sampled_dataset(paths.data_dir / "dataset.csv")
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    train_embeddings = encode_texts(
        train_df["text"].tolist(),
        build_embedding_cache_name(model_name, "train"),
        model_name=model_name,
    )
    test_embeddings = encode_texts(
        test_df["text"].tolist(),
        build_embedding_cache_name(model_name, "test"),
        model_name=model_name,
    )

    all_results = []
    cls_results, _ = run_feature_space(
        train_embeddings["cls"],
        test_embeddings["cls"],
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="contextual",
        representation=f"{model_name}_cls",
    )
    all_results.extend(cls_results)

    mean_results, mean_predictions = run_feature_space(
        train_embeddings["mean"],
        test_embeddings["mean"],
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="contextual",
        representation=f"{model_name}_mean",
    )
    all_results.extend(mean_results)

    results_df = results_to_frame(all_results)
    results_df.to_csv(paths.tables_dir / "part3_contextual_results.csv", index=False)
    results_df.to_csv(
        paths.tables_dir / f"part3_contextual_results__{encoder_slug}.csv",
        index=False,
    )
    save_model_comparison(results_df, paths.figures_dir / "part3_contextual_model_comparison.png")
    save_model_comparison(
        results_df,
        paths.figures_dir / f"part3_contextual_model_comparison__{encoder_slug}.png",
    )

    variance_rows = []
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
    variance_df.to_csv(paths.tables_dir / "part3_contextual_explained_variance.csv", index=False)
    variance_df.to_csv(
        paths.tables_dir / f"part3_contextual_explained_variance__{encoder_slug}.csv",
        index=False,
    )
    save_explained_variance(variance_df, paths.figures_dir / "part3_contextual_explained_variance.png")
    save_explained_variance(
        variance_df,
        paths.figures_dir / f"part3_contextual_explained_variance__{encoder_slug}.png",
    )

    emb_reducer, emb_train_reduced, emb_test_reduced = reduce_embedding_dimensions(
        train_embeddings["mean"], test_embeddings["mean"], n_components=100
    )
    reduced_results, _ = run_feature_space(
        emb_train_reduced,
        emb_test_reduced,
        train_df["label"].to_numpy(),
        test_df["label"].to_numpy(),
        feature_space="reduced_contextual",
        representation=f"{model_name}_embedding_pca_100",
    )
    reduced_df = results_to_frame(reduced_results)
    reduced_df.to_csv(paths.tables_dir / "part3_contextual_reduced_results.csv", index=False)
    reduced_df.to_csv(
        paths.tables_dir / f"part3_contextual_reduced_results__{encoder_slug}.csv",
        index=False,
    )

    best_contextual = (
        results_df.query(f"representation == '{model_name}_mean'")
        .sort_values("macro_f1", ascending=False)
        .iloc[0]
    )
    contextual_pred = mean_predictions[best_contextual["model_name"]]
    contextual_report = build_classification_report(test_df["label"], contextual_pred, label_names)
    contextual_report.to_csv(paths.tables_dir / "part3_contextual_best_report.csv")
    contextual_report.to_csv(
        paths.tables_dir / f"part3_contextual_best_report__{encoder_slug}.csv"
    )
    contextual_confusion = build_confusion(test_df["label"], contextual_pred, label_names)
    save_heatmap(
        contextual_confusion,
        paths.figures_dir / "part3_contextual_best_confusion.png",
        title=f"Part 2 Best Contextual Model: {best_contextual['model_name']}",
    )
    save_heatmap(
        contextual_confusion,
        paths.figures_dir / f"part3_contextual_best_confusion__{encoder_slug}.png",
        title=f"Part 3 Best Contextual Model: {best_contextual['model_name']}",
    )

    summary = {
        "encoder": model_name,
        "best_model": str(best_contextual["model_name"]),
        "accuracy": float(best_contextual["accuracy"]),
        "macro_f1": float(best_contextual["macro_f1"]),
    }
    pd.DataFrame([summary]).to_csv(
        paths.tables_dir / f"part3_contextual_summary__{encoder_slug}.csv",
        index=False,
    )
    return summary
