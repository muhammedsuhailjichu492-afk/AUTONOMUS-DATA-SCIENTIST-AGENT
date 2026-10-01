"""
llm_client.py
Thin wrapper around the Anthropic API used by every LLM-powered feature in
this project (agentic advisor, narrative report writer, and chat).

Design goals:
  - The rest of the codebase never talks to the `anthropic` package directly.
  - Every method degrades gracefully: if no API key is configured, the
    `anthropic` package isn't installed, or a call fails for any reason
    (network, auth, rate limit, malformed response), methods return None
    instead of raising. Callers are expected to fall back to the existing
    rule-based behavior when that happens, so the pipeline keeps working
    exactly as before even with zero LLM configuration.

Configuration:
  - API key: pass `api_key=...` explicitly, or set the ANTHROPIC_API_KEY
    environment variable.
  - Model: defaults to "claude-sonnet-4-6" (a solid, inexpensive default for
    this kind of structured-advice / report-writing work). Pass a different
    model string (e.g. "claude-opus-4-8" or "claude-haiku-4-5-20251001") to
    change it. Always double-check the current model ID in Anthropic's docs
    (https://docs.claude.com/en/docs/about-claude/models/overview) before
    relying on this in production, since model strings are updated over time.
"""

from __future__ import annotations
import os
import json
import re
from typing import List, Dict, Any, Optional

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False


DEFAULT_MODEL = "claude-sonnet-4-6"


class LLMClient:
    """Optional, fail-soft Anthropic API client.

    `available` tells callers whether LLM features can actually be used.
    Every public method is safe to call regardless of `available` — they
    just return None (or a default) when it's False.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        enabled: bool = True,
    ):
        self.model = model
        self.enabled = enabled
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        self.last_error: Optional[str] = None

        self.available = bool(
            self.enabled and self.api_key and ANTHROPIC_SDK_AVAILABLE
        )
        if self.available:
            try:
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:  # pragma: no cover - defensive
                self.available = False
                self.last_error = str(e)

    def status_message(self) -> str:
        """Human-readable explanation of why LLM features are/aren't active."""
        if not self.enabled:
            return "LLM features disabled by configuration."
        if not ANTHROPIC_SDK_AVAILABLE:
            return "The 'anthropic' package is not installed (pip install anthropic)."
        if not self.api_key:
            return "No API key configured (set ANTHROPIC_API_KEY or pass one in)."
        if not self.available:
            return f"LLM client failed to initialize: {self.last_error}"
        return f"LLM features active (model: {self.model})."

    # ------------------------------------------------------------------
    # Single-turn helpers
    # ------------------------------------------------------------------
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> Optional[str]:
        """Single-turn completion. Returns None on any failure."""
        if not self.available:
            return None
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
            return text.strip() or None
        except Exception as e:
            self.last_error = str(e)
            return None

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Single-turn completion that asks for and parses a JSON object.
        Returns None if the call fails or the response isn't valid JSON."""
        json_system = (
            system
            + "\n\nRespond with ONLY a single valid JSON object. No prose, "
            "no markdown code fences, no explanation outside the JSON."
        )
        text = self.complete(json_system, user, max_tokens=max_tokens, temperature=temperature)
        if not text:
            return None
        return self._extract_json(text)

    # ------------------------------------------------------------------
    # Multi-turn chat
    # ------------------------------------------------------------------
    def chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
        temperature: float = 0.4,
    ) -> Optional[str]:
        """Multi-turn chat. `messages` is a list of {"role": "user"|"assistant",
        "content": str} dicts. Returns None on any failure."""
        if not self.available:
            return None
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
            return text.strip() or None
        except Exception as e:
            self.last_error = str(e)
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of a JSON object from model output, even if
        it's wrapped in code fences or has stray leading/trailing text."""
        candidate = text.strip()
        candidate = re.sub(r"^```(json)?", "", candidate).strip()
        candidate = re.sub(r"```$", "", candidate).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None
