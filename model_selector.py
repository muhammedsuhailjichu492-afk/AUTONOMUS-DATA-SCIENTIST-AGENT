

from __future__ import annotations
from typing import Dict, List
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ModelSelector:
    """Chooses candidate models based on task type and dataset scale."""

    def get_candidates(self, task_type: str, n_rows: int, n_features: int) -> Dict[str, object]:
        candidates: Dict[str, object] = {}

        if task_type == "classification":
            candidates["LogisticRegression"] = LogisticRegression(max_iter=1000)
            candidates["RandomForestClassifier"] = RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1
            )
            if n_rows <= 20000:
                candidates["KNeighborsClassifier"] = KNeighborsClassifier()
                candidates["SVC"] = SVC(probability=True)
            if XGBOOST_AVAILABLE:
                candidates["XGBClassifier"] = XGBClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=5,
                    eval_metric="logloss", random_state=42, n_jobs=-1
                )
        else:  # regression
            candidates["LinearRegression"] = LinearRegression()
            candidates["Ridge"] = Ridge()
            candidates["RandomForestRegressor"] = RandomForestRegressor(
                n_estimators=300, random_state=42, n_jobs=-1
            )
            if n_rows <= 20000:
                candidates["KNeighborsRegressor"] = KNeighborsRegressor()
                candidates["SVR"] = SVR()
            if XGBOOST_AVAILABLE:
                candidates["XGBRegressor"] = XGBRegressor(
                    n_estimators=300, learning_rate=0.05, max_depth=5,
                    random_state=42, n_jobs=-1
                )

        return candidates

    @staticmethod
    def xgboost_available() -> bool:
        return XGBOOST_AVAILABLE
