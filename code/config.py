from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from encoder_registry import DEFAULT_ENCODER


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    cache_dir: Path
    figures_dir: Path
    tables_dir: Path
    models_dir: Path


def get_paths() -> ProjectPaths:
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    return ProjectPaths(
        root=root,
        data_dir=results_dir / "data",
        cache_dir=results_dir / "cache",
        figures_dir=results_dir / "figures",
        tables_dir=results_dir / "tables",
        models_dir=results_dir / "models",
    )


RANDOM_STATE = 42
DATASET_NAME = "ag_news"
SAMPLE_SIZE = 8_000
TRAIN_FRACTION = 0.8
CV_FOLDS = 3
EMBEDDING_MODEL_NAME = DEFAULT_ENCODER
MAX_BOW_FEATURES = 12_000
MAX_TFIDF_FEATURES = 20_000
EMBEDDING_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 192
TSNE_MAX_SAMPLES = 1_200
