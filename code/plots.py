from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

from config import RANDOM_STATE, TSNE_MAX_SAMPLES


sns.set_theme(style="whitegrid")


def save_class_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=frame, x="label_name", order=frame["label_name"].value_counts().index)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_text_length_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(frame["text_length"], bins=40, kde=True)
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_model_comparison(results: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    plot_df = results.copy()
    plot_df["setting"] = plot_df["feature_space"] + " | " + plot_df["representation"]
    sns.barplot(data=plot_df, x="model_name", y="macro_f1", hue="setting")
    plt.title("Macro F1 by Model and Feature Space")
    plt.xlabel("Model")
    plt.ylabel("Macro F1")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_explained_variance(explained_variance: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=explained_variance, x="n_components", y="cumulative_explained_variance", hue="feature_space", marker="o")
    plt.title("Explained Variance vs Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_heatmap(matrix: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_tsne_projection(features, labels: list[str], output_path: Path, title: str) -> None:
    if len(labels) > TSNE_MAX_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = np.sort(rng.choice(len(labels), size=TSNE_MAX_SAMPLES, replace=False))
        features = features[selected]
        labels = [labels[idx] for idx in selected]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    projection = tsne.fit_transform(features)
    plot_df = pd.DataFrame(
        {
            "x": projection[:, 0],
            "y": projection[:, 1],
            "label": labels,
        }
    )
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="label",
        palette="deep",
        alpha=0.8,
        s=32,
    )
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
