from __future__ import annotations

import pandas as pd
from tqdm.auto import tqdm

from config import get_paths
from encoder_registry import get_supported_encoders, safe_encoder_name
from part2_embeddings import build_embedding_cache_name
from part3_contextual import run_contextual_part
from plots import save_model_comparison
from run_experiments import ensure_dirs


def compare_supported_encoders() -> None:
    ensure_dirs()
    paths = get_paths()

    summaries = []
    for encoder_name in tqdm(get_supported_encoders(), desc="Comparing encoders", unit="encoder"):
        train_cache = build_embedding_cache_name(encoder_name, "train")
        test_cache = build_embedding_cache_name(encoder_name, "test")
        if not train_cache.exists() or not test_cache.exists():
            print(
                f"Skipping {encoder_name}: missing cached embeddings. Run part2 first.",
                flush=True,
            )
            continue
        summary = run_contextual_part(model_name=encoder_name)
        summaries.append(summary)

    if not summaries:
        raise RuntimeError("No encoder caches were available for comparison.")

    leaderboard = pd.DataFrame(summaries).sort_values(
        by=["macro_f1", "accuracy"],
        ascending=[False, False],
    )
    leaderboard.to_csv(paths.tables_dir / "encoder_leaderboard.csv", index=False)

    plot_df = leaderboard.rename(
        columns={
            "encoder": "representation",
            "best_model": "model_name",
        }
    )
    plot_df["feature_space"] = "encoder_search"
    save_model_comparison(plot_df, paths.figures_dir / "encoder_leaderboard.png")

    best_encoder = leaderboard.iloc[0]["encoder"]
    pd.DataFrame(
        [
            {
                "best_encoder": best_encoder,
                "best_encoder_safe_name": safe_encoder_name(best_encoder),
                "best_model": leaderboard.iloc[0]["best_model"],
                "macro_f1": leaderboard.iloc[0]["macro_f1"],
                "accuracy": leaderboard.iloc[0]["accuracy"],
            }
        ]
    ).to_csv(paths.tables_dir / "best_encoder_summary.csv", index=False)


if __name__ == "__main__":
    compare_supported_encoders()
