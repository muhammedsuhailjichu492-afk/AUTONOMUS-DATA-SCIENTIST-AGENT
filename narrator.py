

from __future__ import annotations
from typing import List, Optional

from .llm_client import LLMClient


def default_summary(dataset_name: str, best_model_name: str) -> str:
    return (
        f"The agent inspected **{dataset_name}**, cleaned and transformed the data, "
        f"explored it visually, trained and compared multiple candidate models, and "
        f"selected **{best_model_name}** as the best-performing model for the stated "
        f"business objective. See the metrics and plots above for detail."
    )


class Narrator:
    def __init__(self, client: LLMClient):
        self.client = client

    def narrate(
        self,
        dataset_name: str,
        business_objective: str,
        profile_md: str,
        cleaning_log: List[str],
        feature_log: List[str],
        leaderboard_md: str,
        best_model_name: str,
        evaluation_md: str,
    ) -> str:
        fallback = default_summary(dataset_name, best_model_name)
        if not self.client.available:
            return fallback

        system = (
            "You are a data science communicator writing the executive-summary "
            "section of an automated pipeline report for a business stakeholder "
            "who is not a data scientist. Write in plain language. Be concrete: "
            "reference the actual dataset, model, and metrics given to you. "
            "Do not invent numbers that weren't provided. Keep it to 3-5 short "
            "paragraphs or a short paragraph plus a few bullet points. End with "
            "1-3 concrete next-step recommendations tied to the business "
            "objective. Output markdown, with no top-level heading (the caller "
            "adds its own)."
        )
        user = (
            f"Dataset: {dataset_name}\n"
            f"Business objective: {business_objective or 'Not specified.'}\n\n"
            f"{profile_md}\n\n"
            "Cleaning actions taken:\n" + ("\n".join(f"- {l}" for l in cleaning_log) or "- None") + "\n\n"
            "Feature engineering:\n" + ("\n".join(f"- {l}" for l in feature_log) or "- None") + "\n\n"
            f"{leaderboard_md}\n\n"
            f"Selected model: {best_model_name}\n\n"
            f"{evaluation_md}\n"
        )
        text = self.client.complete(system, user, max_tokens=700, temperature=0.4)
        return text if text else fallback
