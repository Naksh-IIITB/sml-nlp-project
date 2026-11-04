from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import get_paths
from data import load_sampled_dataset
from embeddings import FrozenEncoder
from encoder_registry import DEFAULT_ENCODER, safe_encoder_name
from plots import save_tsne_projection
from run_experiments import ensure_dirs


def build_embedding_cache_name(model_name: str, split_name: str) -> Path:
    safe_name = safe_encoder_name(model_name)
    return get_paths().cache_dir / f"{split_name}_{safe_name}_embeddings.npz"


def run_embedding_part(model_name: str = DEFAULT_ENCODER) -> None:
    ensure_dirs()
    paths = get_paths()
    train_df, test_df = load_sampled_dataset(paths.data_dir / "dataset.csv")

    print(f"Preparing encoder embeddings using: {model_name}", flush=True)
    print("This step is GPU-friendly and can be run on Kaggle or Colab.", flush=True)

    encoder = FrozenEncoder(model_name=model_name, show_progress=True)
    train_embeddings = encoder.encode_texts(
        train_df["text"].tolist(),
        build_embedding_cache_name(model_name, "train"),
    )
    test_embeddings = encoder.encode_texts(
        test_df["text"].tolist(),
        build_embedding_cache_name(model_name, "test"),
    )

    save_tsne_projection(
        test_embeddings["mean"],
        test_df["label_name"].tolist(),
        paths.figures_dir / f"part2_{safe_encoder_name(model_name)}_mean_tsne.png",
        title=f"t-SNE on {model_name} Mean Embeddings",
    )
    save_tsne_projection(
        test_embeddings["cls"],
        test_df["label_name"].tolist(),
        paths.figures_dir / f"part2_{safe_encoder_name(model_name)}_cls_tsne.png",
        title=f"t-SNE on {model_name} CLS Embeddings",
    )
    metadata = pd.DataFrame(
        [
            {
                "encoder": model_name,
                "train_cache": str(build_embedding_cache_name(model_name, "train")),
                "test_cache": str(build_embedding_cache_name(model_name, "test")),
                "embedding_dim": int(train_embeddings["mean"].shape[1]),
                "train_rows": int(train_embeddings["mean"].shape[0]),
                "test_rows": int(test_embeddings["mean"].shape[0]),
            }
        ]
    )
    metadata.to_csv(
        paths.tables_dir / f"part2_{safe_encoder_name(model_name)}_embedding_index.csv",
        index=False,
    )
    print("Embedding extraction complete.", flush=True)
    print(f"Train cache: {build_embedding_cache_name(model_name, 'train')}", flush=True)
    print(f"Test cache: {build_embedding_cache_name(model_name, 'test')}", flush=True)
