from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import DATASET_NAME, RANDOM_STATE, SAMPLE_SIZE, TRAIN_FRACTION


def load_sampled_dataset(cache_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    train_cache = cache_path.with_name("train_sample.csv")
    test_cache = cache_path.with_name("test_sample.csv")

    if train_cache.exists() and test_cache.exists():
        train_df = pd.read_csv(train_cache)
        test_df = pd.read_csv(test_cache)
        return train_df, test_df

    dataset = load_dataset(DATASET_NAME, split="train")
    full_frame = dataset.to_pandas()[["text", "label"]]
    frame, _ = train_test_split(
        full_frame,
        train_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=full_frame["label"],
    )
    frame["label_name"] = frame["label"].map(
        {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech",
        }
    )
    frame["text_length"] = frame["text"].str.split().str.len()

    train_df, test_df = train_test_split(
        frame,
        train_size=TRAIN_FRACTION,
        random_state=RANDOM_STATE,
        stratify=frame["label"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv(train_cache, index=False)
    test_df.to_csv(test_cache, index=False)
    return train_df, test_df
