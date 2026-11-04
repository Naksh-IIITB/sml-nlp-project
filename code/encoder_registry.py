from __future__ import annotations

ENCODER_CANDIDATES = {
    "distilbert-base-uncased": {
        "label": "DistilBERT Base",
        "hidden_size": 768,
        "notes": "Fast baseline encoder with 768-dim vectors.",
    },
    "bert-base-uncased": {
        "label": "BERT Base",
        "hidden_size": 768,
        "notes": "Strong general-purpose encoder with 768-dim vectors.",
    },
    "roberta-base": {
        "label": "RoBERTa Base",
        "hidden_size": 768,
        "notes": "Often stronger than BERT on classification-style tasks.",
    },
    "microsoft/deberta-v3-base": {
        "label": "DeBERTa-v3 Base",
        "hidden_size": 768,
        "notes": "Recommended high-accuracy encoder candidate with 768-dim vectors.",
    },
}

DEFAULT_ENCODER = "microsoft/deberta-v3-base"


def get_supported_encoders() -> list[str]:
    return list(ENCODER_CANDIDATES.keys())


def safe_encoder_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_")
