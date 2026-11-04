from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from config import CV_FOLDS, RANDOM_STATE


@dataclass
class KMeansClassifier:
    n_clusters: int
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        self.model = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
        self.cluster_to_label: dict[int, int] = {}

    def fit(self, features, labels):
        clusters = self.model.fit_predict(features)
        for cluster_id in range(self.n_clusters):
            mask = clusters == cluster_id
            if not np.any(mask):
                self.cluster_to_label[cluster_id] = int(np.bincount(labels).argmax())
                continue
            self.cluster_to_label[cluster_id] = int(np.bincount(labels[mask]).argmax())
        return self

    def predict(self, features):
        clusters = self.model.predict(features)
        return np.array([self.cluster_to_label[int(cluster)] for cluster in clusters])


def build_classifiers(num_classes: int, dense_features: bool) -> dict[str, BaseEstimator]:
    if dense_features:
        return {
            "logistic_regression": LogisticRegression(
                max_iter=4_000,
                solver="lbfgs",
                C=1.0,
                random_state=RANDOM_STATE,
            ),
            "linear_svm": LinearSVC(
                C=1.0,
                random_state=RANDOM_STATE,
                max_iter=8_000,
            ),
            "knn": KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
            ),
            "kmeans": KMeansClassifier(n_clusters=num_classes),
            "rbf_svm": SVC(
                kernel="rbf",
                gamma="scale",
                C=2.0,
                random_state=RANDOM_STATE,
            ),
        }

    scaler = StandardScaler(with_mean=False)
    models: dict[str, BaseEstimator] = {
        "logistic_regression": make_pipeline(
            scaler,
            LogisticRegression(
                max_iter=2_000,
                solver="saga",
                random_state=RANDOM_STATE,
            ),
        ),
        "linear_svm": make_pipeline(
            scaler,
            LinearSVC(random_state=RANDOM_STATE, max_iter=5_000),
        ),
        "knn": make_pipeline(
            scaler,
            KNeighborsClassifier(),
        ),
        "kmeans": KMeansClassifier(n_clusters=num_classes),
        "rbf_svm": make_pipeline(
            StandardScaler(with_mean=False),
            SVC(kernel="rbf", gamma="scale", random_state=RANDOM_STATE),
        ),
    }
    return models


def build_param_grids(dense_features: bool) -> dict[str, list[dict]]:
    return {
        "logistic_regression": [
            {
                "logisticregression__C": [0.5, 1.0, 2.0],
            }
        ],
        "linear_svm": [
            {
                "linearsvc__C": [0.5, 1.0, 2.0],
            }
        ],
        "knn": [
            {
                "kneighborsclassifier__n_neighbors": [3, 5, 7, 11],
                "kneighborsclassifier__weights": ["uniform", "distance"],
            }
        ],
        "rbf_svm": [
            {
                "svc__C": [1.0, 2.0, 4.0] if dense_features else [0.5, 1.0, 2.0],
                "svc__gamma": ["scale", 0.1],
            }
        ],
        "kmeans": [{}],
    }


def tune_estimator(
    model_name: str,
    estimator,
    features,
    labels,
    *,
    dense_features: bool,
    representation: str | None = None,
):
    if model_name == "kmeans":
        estimator.fit(features, labels)
        return estimator, None, None
    
    if dense_features:
        estimator.fit(features, labels)
        return estimator, None, "stable_dense_defaults"

    if model_name == "knn":
        estimator.fit(features, labels)
        return estimator, None, "default_params_for_efficiency"
    
    if model_name == "rbf_svm" and not dense_features:
        estimator.fit(features, labels)
        return estimator, None, "default_params_for_efficiency"

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=build_param_grids(dense_features)[model_name],
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(features, labels)
    return search.best_estimator_, float(search.best_score_), str(search.best_params_)


def reduce_tfidf_dimensions(train_x, test_x, n_components: int = 300):
    reducer = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    return reducer, reducer.fit_transform(train_x), reducer.transform(test_x)


def reduce_embedding_dimensions(train_x, test_x, n_components: int = 100):
    reducer = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return reducer, reducer.fit_transform(train_x), reducer.transform(test_x)
