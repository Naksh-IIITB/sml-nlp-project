from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass
class EvaluationResult:
    feature_space: str
    representation: str
    model_name: str
    accuracy: float
    macro_f1: float
    cv_macro_f1: float | None = None
    best_params: str | None = None


def evaluate_predictions(
    y_true,
    y_pred,
    *,
    feature_space: str,
    representation: str,
    model_name: str,
    cv_macro_f1: float | None = None,
    best_params: str | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        feature_space=feature_space,
        representation=representation,
        model_name=model_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        cv_macro_f1=cv_macro_f1,
        best_params=best_params,
    )


def results_to_frame(results: list[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(result) for result in results]).sort_values(
        by=["feature_space", "representation", "macro_f1", "accuracy"],
        ascending=[True, True, False, False],
    )


def build_classification_report(y_true, y_pred, label_names: list[str]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def build_confusion(y_true, y_pred, label_names: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(matrix, index=label_names, columns=label_names)
