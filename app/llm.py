"""LLM providers for answer generation -- bring your own key.

In a RAG system the LLM does exactly one job: read the chunks retrieval found
and write an answer from them. Everything before that (loading, chunking,
embedding, searching, reranking) is ours and runs locally. That makes the
provider a swappable detail: Google Gemini has a no-credit-card free tier,
Anthropic Claude is the paid upgrade, and nothing upstream knows which one ran.

Each provider returns the same `LLMAnswer`, so callers stay provider-free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from app.config import DEFAULT_MODELS, Provider, settings

_MAX_ATTEMPTS = 5
_BACKOFF_SECONDS = 10


@dataclass(frozen=True)
class LLMAnswer:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMRefused(RuntimeError):
    """The provider declined to answer (safety filter, blocked prompt)."""


class ProviderNotConfigured(RuntimeError):
    """No API key for the selected provider."""


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, prompt: str) -> LLMAnswer: ...


class GeminiProvider:
    """Google Gemini. Free tier needs no credit card and no billing account.

    Get a key at https://aistudio.google.com/apikey and put it in .env as
    GOOGLE_API_KEY. The free tier is rate-limited per minute and per day, which
    matters once we run the evaluation set (dozens of questions in a row).
    """

    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=8_000,
            # We pass no tools; leaving AFC on only produces a log warning.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        # The free tier regularly answers 503 ("high demand") or 429 (per-minute
        # quota). Both are temporary, so wait and retry instead of failing a whole
        # evaluation run halfway through.
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                break
            except errors.APIError as exc:
                if exc.code not in (429, 500, 503) or attempt == _MAX_ATTEMPTS - 1:
                    raise
                time.sleep(_BACKOFF_SECONDS * (attempt + 1))

        if not response.text:
            raise LLMRefused(_gemini_refusal_detail(response))

        usage = response.usage_metadata
        return LLMAnswer(
            text=response.text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )


def _gemini_refusal_detail(response: object) -> str:
    """Turn an empty Gemini response into something a user can act on."""
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        return f"Gemini blocked the prompt: {feedback.block_reason}."
    candidates = getattr(response, "candidates", None) or []
    if candidates and getattr(candidates[0], "finish_reason", None) is not None:
        return f"Gemini returned no text (finish_reason={candidates[0].finish_reason})."
    return "Gemini returned no text."


class AnthropicProvider:
    """Anthropic Claude. Paid API -- set ANTHROPIC_API_KEY.

    Server-side refusal fallbacks are enabled: if the model declines on policy
    grounds, the API re-runs the request on a fallback model inside the same call.
    """

    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16_000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "explanation", "") or ""
            raise LLMRefused(f"Claude declined to answer. {detail}".strip())

        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMAnswer(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class NvidiaProvider:
    """NVIDIA's hosted models (build.nvidia.com). Free for development -- set NVIDIA_API_KEY.

    The API speaks the OpenAI format, so the official `openai` client works with a
    different base URL. Nemotron is a reasoning model: it thinks first (returned
    separately as `reasoning_content`), then writes the answer in `content`.
    """

    name = "nvidia"
    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(base_url=self.BASE_URL, api_key=api_key, max_retries=_MAX_ATTEMPTS)

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=8_000,  # room for the thinking plus the answer
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMRefused(f"{self.model} returned no text (finish_reason={response.choices[0].finish_reason}).")
        return LLMAnswer(
            text=text,
            model=self.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )


_MISSING_KEY_HELP = {
    "gemini": (
        "GOOGLE_API_KEY is not set. Get a free key (no credit card) at "
        "https://aistudio.google.com/apikey and add it to .env"
    ),
    "anthropic": (
        "ANTHROPIC_API_KEY is not set. Get a key at https://console.anthropic.com "
        "and add it to .env, or set LLM_PROVIDER=gemini to use the free tier"
    ),
    "nvidia": (
        "NVIDIA_API_KEY is not set. Get a free key at https://build.nvidia.com "
        "and add it to .env"
    ),
}


PROVIDERS = {"gemini": GeminiProvider, "anthropic": AnthropicProvider, "nvidia": NvidiaProvider}


def get_llm(provider: Provider | None = None, model: str | None = None) -> LLMProvider:
    """Build the provider named by LLM_PROVIDER (or `provider`). Raises if its key is missing.

    The arguments let the evaluation build its judge from a different provider.
    """
    if provider is None:
        provider, model = settings.llm_provider, model or settings.llm_model_resolved
    key = settings.api_key(provider)
    if not key:
        raise ProviderNotConfigured(_MISSING_KEY_HELP[provider])
    return PROVIDERS[provider](key, model or DEFAULT_MODELS[provider])


def get_judge() -> LLMProvider:
    """The evaluation's grader (A7): JUDGE_PROVIDER / JUDGE_MODEL."""
    return get_llm(settings.judge_provider, settings.judge_model)
