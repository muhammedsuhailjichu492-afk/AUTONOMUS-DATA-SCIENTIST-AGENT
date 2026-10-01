from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)


class Evaluator:
    def __init__(self, output_dir: str, task_type: str):
        self.output_dir = output_dir
        self.task_type = task_type
        os.makedirs(output_dir, exist_ok=True)
        self.metrics: dict = {}
        self.plot_path: str | None = None

    def evaluate(self, pipeline, X_test, y_test, class_names=None) -> dict:
        y_pred = pipeline.predict(X_test)

        if self.task_type == "classification":
            self.metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
            self._plot_confusion_matrix(y_test, y_pred, class_names)
        else:
            self.metrics = {
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": r2_score(y_test, y_pred),
            }
            self._plot_residuals(y_test, y_pred)

        return self.metrics

    def _plot_confusion_matrix(self, y_test, y_pred, class_names=None):
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names if class_names is not None else "auto",
                    yticklabels=class_names if class_names is not None else "auto", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Test Set)")
        path = os.path.join(self.output_dir, "confusion_matrix.png")
        fig.savefig(path, bbox_inches="tight", dpi=110)
        plt.close(fig)
        self.plot_path = path

    def _plot_residuals(self, y_test, y_pred):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(y_test, y_pred, alpha=0.5, edgecolor="k", linewidth=0.3)
        lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
        axes[0].plot(lims, lims, "r--", linewidth=1)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Predicted vs Actual (Test Set)")

        residuals = np.array(y_test) - np.array(y_pred)
        sns.histplot(residuals, kde=True, ax=axes[1], color="#C44E52")
        axes[1].set_title("Residual Distribution")
        axes[1].set_xlabel("Residual")

        fig.tight_layout()
        path = os.path.join(self.output_dir, "residuals.png")
        fig.savefig(path, bbox_inches="tight", dpi=110)
        plt.close(fig)
        self.plot_path = path

    def metrics_markdown(self) -> str:
        lines = ["## Evaluation Metrics (Test Set)", "", "| Metric | Value |", "|---|---|"]
        for k, v in self.metrics.items():
            lines.append(f"| {k} | {v:.4f} |")
        return "\n".join(lines)
