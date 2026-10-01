from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from .profiler import DataProfile

sns.set_theme(style="whitegrid")


class EDAEngine:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.generated_plots: List[str] = []

    def run(self, df: pd.DataFrame, profile: DataProfile) -> List[str]:
        self.generated_plots = []
        numeric_cols = [c for c in profile.numeric_cols if c in df.columns]
        categorical_cols = [c for c in profile.categorical_cols
                             if c in df.columns and df[c].nunique() <= 20]

        # 1. Correlation heatmap (numeric features)
        if len(numeric_cols) >= 2:
            self._correlation_heatmap(df, numeric_cols)

        # 2. Distribution histograms for numeric columns (top 9 by variance)
        if numeric_cols:
            self._histograms(df, numeric_cols)

        # 3. Boxplots for numeric columns vs target (if classification target)
        if profile.target_col and profile.task_type == "classification" and numeric_cols:
            self._boxplots_vs_target(df, numeric_cols, profile.target_col)

        # 4. Target distribution
        if profile.target_col and profile.target_col in df.columns:
            self._target_distribution(df, profile.target_col, profile.task_type)

        # 5. Categorical count plots (top few)
        if categorical_cols:
            self._categorical_counts(df, categorical_cols)

        return self.generated_plots

    def _save(self, fig, name: str):
        path = os.path.join(self.output_dir, name)
        fig.savefig(path, bbox_inches="tight", dpi=110)
        plt.close(fig)
        self.generated_plots.append(path)

    def _correlation_heatmap(self, df, numeric_cols):
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(min(1 + 0.6 * len(numeric_cols), 14),
                                         min(1 + 0.6 * len(numeric_cols), 12)))
        sns.heatmap(corr, annot=len(numeric_cols) <= 15, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap")
        self._save(fig, "correlation_heatmap.png")

    def _histograms(self, df, numeric_cols):
        cols = sorted(numeric_cols, key=lambda c: -df[c].var())[:9]
        n = len(cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, c in enumerate(cols):
            sns.histplot(df[c].dropna(), kde=True, ax=axes[i], color="#4C72B0")
            axes[i].set_title(c)
        for j in range(len(cols), len(axes)):
            axes[j].axis("off")
        fig.suptitle("Feature Distributions", y=1.02)
        fig.tight_layout()
        self._save(fig, "distributions.png")

    def _boxplots_vs_target(self, df, numeric_cols, target_col):
        cols = numeric_cols[:6]
        n = len(cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, c in enumerate(cols):
            sns.boxplot(x=df[target_col], y=df[c], ax=axes[i])
            axes[i].set_title(f"{c} by {target_col}")
            axes[i].tick_params(axis="x", rotation=30)
        for j in range(len(cols), len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        self._save(fig, "boxplots_vs_target.png")

    def _target_distribution(self, df, target_col, task_type):
        fig, ax = plt.subplots(figsize=(6, 4))
        if task_type == "classification":
            sns.countplot(x=df[target_col], hue=df[target_col], ax=ax, palette="viridis", legend=False)
            ax.tick_params(axis="x", rotation=30)
        else:
            sns.histplot(df[target_col].dropna(), kde=True, ax=ax, color="#55A868")
        ax.set_title(f"Target Distribution: {target_col}")
        self._save(fig, "target_distribution.png")

    def _categorical_counts(self, df, categorical_cols):
        cols = categorical_cols[:6]
        n = len(cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, c in enumerate(cols):
            order = df[c].value_counts().index[:10]
            sns.countplot(y=df[c], hue=df[c], order=order, ax=axes[i], palette="mako", legend=False)
            axes[i].set_title(c)
        for j in range(len(cols), len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        self._save(fig, "categorical_counts.png")
