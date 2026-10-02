"""Provider-agnostic LLM client.

The rest of the system shouldn't care WHICH model vendor we use. This class
hides that behind one method, complete(system, user) -> str. Swapping OpenAI
for Anthropic is a config change, not a code change — that abstraction is the
"never lock yourself to one vendor" decision (D3) made concrete.

Keys come from .env (OPENAI_API_KEY / ANTHROPIC_API_KEY). SDKs are imported
lazily so you only need the one you actually use installed.
"""
from __future__ import annotations

import os

from gamefin.logging_conf import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 700,
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        """Send a system + user prompt, return the model's text reply."""
        if self.provider == "openai":
            return self._openai(system, user)
        if self.provider == "anthropic":
            return self._anthropic(system, user)
        raise ValueError(f"Unknown llm_provider: {self.provider!r}")

    def _openai(self, system: str, user: str) -> str:
        from openai import OpenAI

        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

    def _anthropic(self, system: str, user: str) -> str:
        import anthropic

        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        client = anthropic.Anthropic(api_key=key)
        # anthropic>=1.0 dropped `temperature` from create(); send it raw via
        # extra_body. Haiku 4.5 / Sonnet 4.6 accept it, but Opus 4.7+ and
        # Sonnet 5 reject it — remove this if you switch to those models.
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            extra_body={"temperature": self.temperature},
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
