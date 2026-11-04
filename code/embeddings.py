from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from config import (
    EMBEDDING_BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
)
from encoder_registry import DEFAULT_ENCODER


def _cache_is_current(data: np.lib.npyio.NpzFile) -> bool:
    if "cls" not in data or "mean" not in data:
        return False
    for key in ["cls", "mean"]:
        arr = data[key]
        if not np.isfinite(arr).all():
            return False
        sample = arr[: min(len(arr), 128)]
        norms = np.linalg.norm(sample, axis=1)
        if not np.all((norms > 0.98) & (norms < 1.02)):
            return False
    return True


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class FrozenEncoder:
    def __init__(self, model_name: str = DEFAULT_ENCODER, show_progress: bool = False) -> None:
        self.model_name = model_name
        self.show_progress = show_progress
        self.device = _select_device()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=True,
            ).to(self.device)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def encode_texts(self, texts: list[str], cache_file: Path) -> dict[str, np.ndarray]:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            data = np.load(cache_file)
            if _cache_is_current(data):
                return {
                    "cls": data["cls"].astype(np.float64),
                    "mean": data["mean"].astype(np.float64),
                }
            cache_file.unlink()

        cls_vectors: list[np.ndarray] = []
        mean_vectors: list[np.ndarray] = []

        starts = range(0, len(texts), EMBEDDING_BATCH_SIZE)
        if self.show_progress:
            starts = tqdm(starts, desc=f"Encoding with {self.model_name}", unit="batch")

        with torch.no_grad():
            for start in starts:
                batch_texts = texts[start : start + EMBEDDING_BATCH_SIZE]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQUENCE_LENGTH,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                last_hidden = outputs.last_hidden_state
                cls_batch = F.normalize(last_hidden[:, 0, :], p=2, dim=1)
                mean_batch = F.normalize(
                    _mean_pool(last_hidden, encoded["attention_mask"]),
                    p=2,
                    dim=1,
                )
                cls_vectors.append(cls_batch.cpu().numpy().astype(np.float64))
                mean_vectors.append(mean_batch.cpu().numpy().astype(np.float64))

        cls_array = np.concatenate(cls_vectors, axis=0)
        mean_array = np.concatenate(mean_vectors, axis=0)
        np.savez_compressed(cache_file, cls=cls_array, mean=mean_array)
        return {"cls": cls_array, "mean": mean_array}


def encode_texts(
    texts: list[str],
    cache_file: Path,
    model_name: str = DEFAULT_ENCODER,
) -> dict[str, np.ndarray]:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        data = np.load(cache_file)
        if _cache_is_current(data):
            return {
                "cls": data["cls"].astype(np.float64),
                "mean": data["mean"].astype(np.float64),
            }
        cache_file.unlink()
    encoder = FrozenEncoder(model_name=model_name)
    return encoder.encode_texts(texts, cache_file)
