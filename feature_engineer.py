

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from .profiler import DataProfile


class FeatureEngineer:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.log: List[str] = []
        self.preprocessor: ColumnTransformer | None = None
        self.label_encoder: LabelEncoder | None = None
        self.feature_names_: List[str] = []

    def build_preprocessor(self, df: pd.DataFrame, profile: DataProfile) -> ColumnTransformer:
        target = profile.target_col
        numeric_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c != target and c not in numeric_cols]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_pipeline, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_pipeline, categorical_cols))

        self.log.append(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
        self.log.append(f"Categorical features ({len(categorical_cols)}) one-hot encoded: {categorical_cols}")

        self.preprocessor = ColumnTransformer(transformers, remainder="drop")
        return self.preprocessor

    def split(self, df: pd.DataFrame, profile: DataProfile):
        target = profile.target_col
        X = df.drop(columns=[target])
        y = df[target]

        stratify = y if profile.task_type == "classification" and y.nunique() > 1 else None

        if profile.task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y), index=y.index, name=target)
            self.log.append(f"Label-encoded target classes: {list(self.label_encoder.classes_)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )
        self.log.append(f"Train/test split: {len(X_train)} train rows, {len(X_test)} test rows "
                         f"(test_size={self.test_size})")
        return X_train, X_test, y_train, y_test
