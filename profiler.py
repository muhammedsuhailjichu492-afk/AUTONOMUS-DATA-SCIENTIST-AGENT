

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_counts: Dict[str, int]
    missing_pct: Dict[str, float]
    n_duplicates: int
    numeric_cols: List[str]
    categorical_cols: List[str]
    datetime_cols: List[str]
    high_cardinality_cols: List[str]
    constant_cols: List[str]
    outlier_summary: Dict[str, int]
    target_col: str | None = None
    task_type: str | None = None  # "classification" | "regression"
    numeric_summary: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            "## Data Profiling Summary",
            f"- Rows: **{self.n_rows}**, Columns: **{self.n_cols}**",
            f"- Duplicate rows: **{self.n_duplicates}**",
            f"- Numeric columns ({len(self.numeric_cols)}): {', '.join(self.numeric_cols) or 'None'}",
            f"- Categorical columns ({len(self.categorical_cols)}): {', '.join(self.categorical_cols) or 'None'}",
            f"- Datetime columns ({len(self.datetime_cols)}): {', '.join(self.datetime_cols) or 'None'}",
        ]
        if self.high_cardinality_cols:
            lines.append(f"- High-cardinality columns (dropped/hashed): {', '.join(self.high_cardinality_cols)}")
        if self.constant_cols:
            lines.append(f"- Constant columns (no variance): {', '.join(self.constant_cols)}")
        missing = {k: v for k, v in self.missing_pct.items() if v > 0}
        if missing:
            top_missing = sorted(missing.items(), key=lambda x: -x[1])[:10]
            lines.append("- Top columns with missing data:")
            for col, pct in top_missing:
                lines.append(f"  - {col}: {pct:.1f}% missing")
        if self.target_col:
            lines.append(f"- Detected target column: **{self.target_col}** -> task type: **{self.task_type}**")
        return "\n".join(lines)


class DataProfiler:
    """Performs autonomous inspection of a raw dataset."""

    HIGH_CARDINALITY_RATIO = 0.9  # unique/nrows above this => treat as identifier-like
    HIGH_CARDINALITY_ABS = 50      # for categorical cols with more unique vals than this

    def profile(self, df: pd.DataFrame, target_col: str | None = None) -> DataProfile:
        n_rows, n_cols = df.shape
        dtypes = {c: str(df[c].dtype) for c in df.columns}
        missing_counts = df.isna().sum().to_dict()
        missing_pct = {c: (missing_counts[c] / n_rows * 100 if n_rows else 0) for c in df.columns}
        n_duplicates = int(df.duplicated().sum())

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
        # attempt to auto-detect date-like object columns
        for c in df.select_dtypes(include=["object"]).columns:
            if c in datetime_cols:
                continue
            sample = df[c].dropna().astype(str).head(20)
            if len(sample) == 0:
                continue
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample, errors="raise")
                if parsed.notna().mean() > 0.9:
                    datetime_cols.append(c)
            except Exception:
                pass

        categorical_cols = [
            c for c in df.columns
            if c not in numeric_cols and c not in datetime_cols
        ]

        high_cardinality_cols = []
        for c in categorical_cols:
            nunique = df[c].nunique(dropna=True)
            if nunique > self.HIGH_CARDINALITY_ABS or (n_rows and nunique / n_rows > self.HIGH_CARDINALITY_RATIO):
                high_cardinality_cols.append(c)

        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

        outlier_summary = {}
        for c in numeric_cols:
            series = df[c].dropna()
            if len(series) < 10:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((series < lower) | (series > upper)).sum())
            if n_out > 0:
                outlier_summary[c] = n_out

        numeric_summary = {}
        if numeric_cols:
            numeric_summary = df[numeric_cols].describe().to_dict()

        task_type = None
        if target_col and target_col in df.columns:
            task_type = self._infer_task_type(df[target_col])

        return DataProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            columns=df.columns.tolist(),
            dtypes=dtypes,
            missing_counts=missing_counts,
            missing_pct=missing_pct,
            n_duplicates=n_duplicates,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            datetime_cols=datetime_cols,
            high_cardinality_cols=high_cardinality_cols,
            constant_cols=constant_cols,
            outlier_summary=outlier_summary,
            target_col=target_col,
            task_type=task_type,
            numeric_summary=numeric_summary,
        )

    @staticmethod
    def _infer_task_type(target: pd.Series) -> str:
        """Autonomous decision: classification vs regression."""
        if target.dtype == object or str(target.dtype) == "category" or target.dtype == bool:
            return "classification"
        nunique = target.nunique(dropna=True)
        if pd.api.types.is_integer_dtype(target) and nunique <= max(20, int(len(target) * 0.05)):
            return "classification"
        if nunique <= 2:
            return "classification"
        return "regression"
