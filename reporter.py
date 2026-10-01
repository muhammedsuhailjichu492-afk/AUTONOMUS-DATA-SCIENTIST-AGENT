

from __future__ import annotations
import os
import datetime
from typing import List


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(
        self,
        dataset_name: str,
        business_objective: str,
        profile_md: str,
        cleaning_log: List[str],
        feature_log: List[str],
        eda_plots: List[str],
        leaderboard_md: str,
        best_model_name: str,
        evaluation_md: str,
        evaluation_plot: str | None,
        summary: str | None = None,
    ) -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        def rel(p: str | None) -> str | None:
            if not p:
                return None
            return os.path.basename(p)

        lines = [
            f"# Autonomous Data Scientist Agent — Report",
            f"*Generated {ts}*",
            "",
            f"**Dataset:** {dataset_name}",
            f"**Business objective:** {business_objective}",
            "",
            "---",
            "",
            profile_md,
            "",
            "---",
            "",
            "## Data Cleaning Actions",
        ]
        if cleaning_log:
            lines += [f"- {line}" for line in cleaning_log]
        else:
            lines.append("- No cleaning actions were necessary.")
        lines += [
            "",
            "---",
            "",
            "## Feature Engineering",
            *(f"- {line}" for line in feature_log),
            "",
            "---",
            "",
            "## Exploratory Data Analysis",
        ]
        for p in eda_plots:
            name = rel(p)
            lines.append(f"![{name}]({name})")
            lines.append("")

        lines += [
            "---",
            "",
            "## Model Selection & Training Leaderboard",
            "",
            leaderboard_md,
            "",
            f"**Selected model:** `{best_model_name}` (highest cross-validated score)",
            "",
            "---",
            "",
            evaluation_md,
            "",
        ]
        if evaluation_plot:
            name = rel(evaluation_plot)
            lines.append(f"![{name}]({name})")
            lines.append("")

        # Use LLM-generated summary if available, otherwise use default
        if summary is None:
            summary = (
                f"The agent inspected **{dataset_name}**, cleaned and transformed the data, "
                f"explored it visually, trained and compared multiple candidate models, and "
                f"selected **{best_model_name}** as the best-performing model for the stated "
                f"business objective. See the metrics and plots above for detail."
            )

        lines += [
            "---",
            "",
            "## Summary",
            summary,
        ]

        report_md = "\n".join(str(l) for l in lines)
        path = os.path.join(self.output_dir, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_md)
        return path
