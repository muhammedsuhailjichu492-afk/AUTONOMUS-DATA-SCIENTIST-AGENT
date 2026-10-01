

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from .llm_client import LLMClient
from .profiler import DataProfile


@dataclass
class CleaningAdvice:
    extra_drop_cols: List[str] = field(default_factory=list)
    missing_threshold: Optional[float] = None  # None -> use DataCleaner's default
    cap_outliers: Optional[bool] = None         # None -> use DataCleaner's default
    rationale: str = ""
    source: str = "default"  # "llm" | "default"


@dataclass
class ModelAdvice:
    selected_models: Optional[List[str]] = None  # None -> use all rule-selected candidates
    rationale: str = ""
    source: str = "default"


DEFAULT_CLEANING_RATIONALE = (
    "LLM advisor unavailable — using default rule-based cleaning "
    "(drop constant/high-cardinality/heavily-missing columns, "
    "median/mode imputation, IQR outlier capping)."
)
DEFAULT_MODEL_RATIONALE = (
    "LLM advisor unavailable — training every rule-selected candidate "
    "model appropriate for this task type and dataset size."
)


class LLMAdvisor:
    def __init__(self, client: LLMClient):
        self.client = client

    # ------------------------------------------------------------------
    def advise_cleaning(self, profile: DataProfile, business_objective: str) -> CleaningAdvice:
        if not self.client.available:
            return CleaningAdvice(rationale=DEFAULT_CLEANING_RATIONALE, source="default")

        system = (
            "You are a data-cleaning advisor inside an automated data science "
            "pipeline. Given a dataset profile and a business objective, "
            "recommend cleaning parameters. Be conservative: only suggest "
            "dropping a column if it is clearly an identifier, a leakage risk, "
            "or obviously irrelevant to the stated objective. Never suggest "
            "dropping the target column."
        )
        user = (
            f"Business objective: {business_objective or 'Not specified.'}\n\n"
            f"{profile.to_markdown()}\n\n"
            "Respond with JSON of the form:\n"
            '{"extra_drop_columns": [<column names to drop beyond the default '
            'rules, or empty list>], "missing_threshold_pct": <number 5-80, the '
            "% missing above which a column should be dropped>, "
            '"cap_outliers": <true or false>, "rationale": <one or two '
            "sentences explaining your choices in plain language>}"
        )
        data = self.client.complete_json(system, user)
        if not data:
            return CleaningAdvice(rationale=DEFAULT_CLEANING_RATIONALE, source="default")

        valid_cols = set(profile.columns) - {profile.target_col}
        extra_drop = [
            c for c in data.get("extra_drop_columns", []) or []
            if isinstance(c, str) and c in valid_cols
        ]

        threshold = data.get("missing_threshold_pct")
        if not isinstance(threshold, (int, float)):
            threshold = None
        else:
            threshold = max(5.0, min(80.0, float(threshold)))

        cap_outliers = data.get("cap_outliers")
        if not isinstance(cap_outliers, bool):
            cap_outliers = None

        rationale = data.get("rationale") or "No rationale provided."
        return CleaningAdvice(
            extra_drop_cols=extra_drop,
            missing_threshold=threshold,
            cap_outliers=cap_outliers,
            rationale=str(rationale),
            source="llm",
        )

    # ------------------------------------------------------------------
    def advise_models(
        self,
        profile: DataProfile,
        candidate_names: List[str],
        business_objective: str,
    ) -> ModelAdvice:
        if not self.client.available or len(candidate_names) <= 1:
            return ModelAdvice(rationale=DEFAULT_MODEL_RATIONALE, source="default")

        system = (
            "You are a model-selection advisor inside an automated data "
            "science pipeline. You will be given a shortlist of candidate "
            "scikit-learn/xgboost model names that have already been vetted "
            "as appropriate for the task type and dataset size. Choose the "
            "subset most likely to perform well and/or be worth the training "
            "time, given the dataset characteristics and business objective. "
            "You may choose all of them if that seems best. Never propose a "
            "model name that isn't in the given list."
        )
        user = (
            f"Business objective: {business_objective or 'Not specified.'}\n\n"
            f"Task type: {profile.task_type}\n"
            f"Rows: {profile.n_rows}, Columns: {profile.n_cols}\n"
            f"Candidate models: {candidate_names}\n\n"
            "Respond with JSON of the form:\n"
            '{"selected_models": [<subset of the candidate model names, '
            "at least one>], \"rationale\": <one or two sentences explaining "
            "your choice in plain language>}"
        )
        data = self.client.complete_json(system, user)
        if not data:
            return ModelAdvice(rationale=DEFAULT_MODEL_RATIONALE, source="default")

        candidate_set = set(candidate_names)
        selected = [
            m for m in data.get("selected_models", []) or []
            if isinstance(m, str) and m in candidate_set
        ]
        if not selected:
            # LLM returned nothing usable — fall back rather than train zero models.
            return ModelAdvice(rationale=DEFAULT_MODEL_RATIONALE, source="default")

        rationale = data.get("rationale") or "No rationale provided."
        return ModelAdvice(selected_models=selected, rationale=str(rationale), source="llm")
