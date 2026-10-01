"""
chat.py
Interactive Q&A over a completed agent run. Give it the run's context
(profile, cleaning/feature logs, leaderboard, metrics, and a small sample of
the cleaned data) and it answers follow-up questions grounded in that
context — "why was column X dropped?", "which model would you pick if I
cared more about recall?", "what does the RMSE mean here?", etc.

If no LLM is configured, `ask()` returns a short explanatory message instead
of raising, so the UI can display it directly either way.
"""

from __future__ import annotations
from typing import List, Tuple

from .llm_client import LLMClient

NOT_CONFIGURED_MESSAGE = (
    "Chat isn't available right now — no Anthropic API key is configured. "
    "Set the ANTHROPIC_API_KEY environment variable (or enter a key in the "
    "app's LLM settings) to enable this feature."
)

MAX_CONTEXT_CHARS = 12000  # keep the system prompt bounded regardless of run size


class DataChatAgent:
    def __init__(self, client: LLMClient, context_md: str):
        self.client = client
        self.context_md = context_md[:MAX_CONTEXT_CHARS]

    def ask(self, question: str, history: List[Tuple[str, str]] | None = None) -> str:
        """`history` is a list of (question, answer) pairs from earlier turns
        in this session, oldest first."""
        if not self.client.available:
            return NOT_CONFIGURED_MESSAGE

        system = (
            "You are a data science assistant answering questions about ONE "
            "specific completed pipeline run. Ground every answer in the run "
            "context provided below — don't invent numbers, columns, or "
            "results that aren't in it. If the context doesn't contain what's "
            "needed to answer, say so plainly and suggest what additional "
            "analysis would be needed, rather than guessing. Keep answers "
            "concise and concrete.\n\n"
            "=== RUN CONTEXT ===\n" + self.context_md
        )

        messages = []
        for q, a in (history or []):
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": question})

        answer = self.client.chat(system, messages, max_tokens=700, temperature=0.3)
        return answer if answer else (
            "Sorry, I couldn't reach the LLM to answer that just now "
            f"({self.client.last_error or 'unknown error'}). Please try again."
        )


def build_context_md(
    dataset_name: str,
    business_objective: str,
    profile_md: str,
    cleaning_log: List[str],
    feature_log: List[str],
    leaderboard_md: str,
    best_model_name: str,
    evaluation_md: str,
    agent_decisions_md: str = "",
) -> str:
    """Assembles the same information the report contains into a single
    context blob suitable for grounding chat answers."""
    cleaning_lines = [f"- {l}" for l in cleaning_log] if cleaning_log else ["- No cleaning actions were necessary."]
    feature_lines = [f"- {l}" for l in feature_log]

    parts = [
        f"Dataset: {dataset_name}",
        f"Business objective: {business_objective or 'Not specified.'}",
        "",
        profile_md,
        "",
        "## Data Cleaning Actions",
        *cleaning_lines,
        "",
        "## Feature Engineering",
        *feature_lines,
        "",
        leaderboard_md,
        "",
        f"Selected model: {best_model_name}",
        "",
        evaluation_md,
    ]
    if agent_decisions_md:
        parts += ["", agent_decisions_md]
    return "\n".join(str(p) for p in parts)
