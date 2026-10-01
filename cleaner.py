
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from .profiler import DataProfile

if TYPE_CHECKING:
    from .llm_advisor import CleaningAdvice


class DataCleaner:
    def __init__(self, missing_threshold: float = 40.0, cap_outliers: bool = True):
        """
        missing_threshold: columns with more than this % missing are dropped.
        cap_outliers: whether to winsorize numeric outliers using the IQR rule.
        """
        self.missing_threshold = missing_threshold
        self.cap_outliers = cap_outliers
        self.log: List[str] = []

    def clean(self, df: pd.DataFrame, profile: DataProfile, advice: Optional["CleaningAdvice"] = None) -> pd.DataFrame:
        df = df.copy()
        self.log = []

        # Apply LLM advice if provided
        if advice and advice.source == "llm":
            if advice.missing_threshold is not None:
                self.missing_threshold = advice.missing_threshold
            if advice.cap_outliers is not None:
                self.cap_outliers = advice.cap_outliers

        # 1. Drop constant columns
        if profile.constant_cols:
            df = df.drop(columns=[c for c in profile.constant_cols if c in df.columns])
            self.log.append(f"Dropped {len(profile.constant_cols)} constant column(s): {profile.constant_cols}")

        # 2. Drop identifier-like high-cardinality categorical columns
        #    (keep the target column even if flagged)
        drop_high_card = [c for c in profile.high_cardinality_cols
                           if c in df.columns and c != profile.target_col]
        if drop_high_card:
            df = df.drop(columns=drop_high_card)
            self.log.append(f"Dropped {len(drop_high_card)} high-cardinality/identifier column(s): {drop_high_card}")

        # 3. Apply LLM extra_drop_cols advice if provided
        extra_drops = []
        if advice and advice.source == "llm" and advice.extra_drop_cols:
            extra_drops = [c for c in advice.extra_drop_cols if c in df.columns and c != profile.target_col]
            if extra_drops:
                df = df.drop(columns=extra_drops)
                self.log.append(f"Dropped {len(extra_drops)} column(s) per LLM advice: {extra_drops}")

        # 4. Drop columns with excessive missingness
        heavy_missing = [c for c, pct in profile.missing_pct.items()
                          if pct > self.missing_threshold and c in df.columns and c != profile.target_col]
        if heavy_missing:
            df = df.drop(columns=heavy_missing)
            self.log.append(f"Dropped {len(heavy_missing)} column(s) with >{self.missing_threshold}% missing: {heavy_missing}")

        # 5. Drop duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            self.log.append(f"Removed {removed} duplicate row(s)")

        # 6. Drop rows where target is missing (can't train/label on those)
        if profile.target_col and profile.target_col in df.columns:
            before = len(df)
            df = df.dropna(subset=[profile.target_col])
            removed = before - len(df)
            if removed:
                self.log.append(f"Removed {removed} row(s) with missing target")

        # 7. Impute remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        if profile.target_col in numeric_cols:
            numeric_cols.remove(profile.target_col)
        if profile.target_col in categorical_cols:
            categorical_cols.remove(profile.target_col)

        for c in numeric_cols:
            if df[c].isna().any():
                median_val = df[c].median()
                df[c] = df[c].fillna(median_val)
                self.log.append(f"Imputed numeric column '{c}' with median ({median_val:.3f})")

        for c in categorical_cols:
            if df[c].isna().any():
                mode_val = df[c].mode(dropna=True)
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                df[c] = df[c].fillna(fill_val)
                self.log.append(f"Imputed categorical column '{c}' with mode ('{fill_val}')")

        # 8. Cap outliers (winsorize) using IQR rule, numeric predictor columns only
        if self.cap_outliers:
            capped_cols = []
            for c in numeric_cols:
                series = df[c]
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_out = ((series < lower) | (series > upper)).sum()
                if n_out > 0:
                    df[c] = series.clip(lower, upper)
                    capped_cols.append(c)
            if capped_cols:
                self.log.append(f"Capped outliers (IQR method) in: {capped_cols}")

        # 9. Parse detected datetime columns and expand into features
        for c in profile.datetime_cols:
            if c in df.columns and c != profile.target_col:
                try:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    df[f"{c}_year"] = parsed.dt.year
                    df[f"{c}_month"] = parsed.dt.month
                    df[f"{c}_dayofweek"] = parsed.dt.dayofweek
                    df = df.drop(columns=[c])
                    self.log.append(f"Expanded datetime column '{c}' into year/month/dayofweek features")
                except Exception:
                    pass

        return df.reset_index(drop=True)
