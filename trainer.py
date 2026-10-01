"""
trainer.py
Stage 6: TRAINING
Trains every candidate model inside the full preprocessing pipeline using
cross-validation, and selects the best performer automatically.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer


class TrainingResult:
    def __init__(self, name: str, pipeline: Pipeline, cv_scores: np.ndarray, fit_time: float):
        self.name = name
        self.pipeline = pipeline
        self.cv_scores = cv_scores
        self.cv_mean = float(np.mean(cv_scores))
        self.cv_std = float(np.std(cv_scores))
        self.fit_time = fit_time


class Trainer:
    def __init__(self, preprocessor: ColumnTransformer, task_type: str, cv_folds: int = 5):
        self.preprocessor = preprocessor
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.scoring = "accuracy" if task_type == "classification" else "r2"
        self.results: List[TrainingResult] = []

    def train_all(self, candidates: Dict[str, object], X_train, y_train) -> List[TrainingResult]:
        self.results = []
        n_folds = min(self.cv_folds, max(2, y_train.value_counts().min())) \
            if self.task_type == "classification" and hasattr(y_train, "value_counts") \
            else self.cv_folds

        for name, model in candidates.items():
            pipeline = Pipeline([
                ("preprocessor", self.preprocessor),
                ("model", model),
            ])
            start = time.time()
            try:
                scores = cross_val_score(pipeline, X_train, y_train,
                                          cv=n_folds, scoring=self.scoring, n_jobs=-1)
            except Exception:
                # fall back to fewer folds if a class has too few samples
                scores = cross_val_score(pipeline, X_train, y_train,
                                          cv=2, scoring=self.scoring, n_jobs=-1)
            fit_time = time.time() - start
            pipeline.fit(X_train, y_train)  # fit on full training data for final model
            self.results.append(TrainingResult(name, pipeline, scores, fit_time))

        self.results.sort(key=lambda r: r.cv_mean, reverse=True)
        return self.results

    def best(self) -> TrainingResult:
        if not self.results:
            raise RuntimeError("No models have been trained yet.")
        return self.results[0]

    def leaderboard_markdown(self) -> str:
        metric = "Accuracy" if self.task_type == "classification" else "R\u00b2"
        lines = [f"| Model | CV {metric} (mean \u00b1 std) | Fit Time (s) |",
                 "|---|---|---|"]
        for r in self.results:
            lines.append(f"| {r.name} | {r.cv_mean:.4f} \u00b1 {r.cv_std:.4f} | {r.fit_time:.2f} |")
        return "\n".join(lines)
