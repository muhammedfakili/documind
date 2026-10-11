"""Application settings, loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Gemini 3.5 Flash-Lite is on the free tier and answers well from retrieved context.
# (gemini-2.5-flash allows only 20 free requests a day -- too few for the A7 exam.)
# Claude is the paid upgrade when answer quality matters more than cost.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-opus-5",
    # NVIDIA's hosted API (build.nvidia.com): free for development, OpenAI-compatible.
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",
}

Provider = Literal["gemini", "anthropic", "nvidia"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM --------------------------------------------------------------
    # The LLM only writes the final answer from the chunks we retrieve.
    # Retrieval (embeddings, search, reranking) runs locally and never calls
    # it, so the provider is a swappable detail.
    llm_provider: Provider = "gemini"
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    nvidia_api_key: str | None = None
    # Leave unset to take the default model for the selected provider.
    llm_model: str | None = None
    # A7: the model that grades answers in the evaluation. A different model family
    # than the one that wrote the answer, so it has no reason to favour its own style.
    judge_provider: Provider = "nvidia"
    judge_model: str | None = None  # unset = default model of judge_provider

    # --- Embeddings -------------------------------------------------------
    # Runs locally (ONNX, no GPU, no API key). Changing the model changes the
    # vector size, so the database table (A5) has to be rebuilt afterwards.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_cache_dir: Path = PROJECT_ROOT / ".cache" / "models"
    # A9: the cross-encoder that re-sorts search results. Also local, also no key.
    # Small (130 MB) and the best of five tried on our exam (scripts/a9_rerank.py).
    rerank_model: str = "jinaai/jina-reranker-v1-tiny-en"
    # A10: when the first answer is "not found", rewrite the question into the documents'
    # words (app/rewrite.py) and try once more. Only misses pay the 2 extra LLM calls.
    query_rewrite: bool = True

    # --- Storage ----------------------------------------------------------
    database_url: str = "postgresql://documind:documind@localhost:5433/documind"

    # --- API (A11) --------------------------------------------------------
    # Unlocks the support-agent mode (internal documents) and /ingest over HTTP.
    # Unset = both are switched off; customers can still ask questions.
    admin_api_key: str | None = None

    def api_key(self, provider: Provider) -> str | None:
        return {
            "gemini": self.google_api_key,
            "anthropic": self.anthropic_api_key,
            "nvidia": self.nvidia_api_key,
        }[provider]

    @property
    def llm_api_key(self) -> str | None:
        """The API key belonging to the selected provider."""
        return self.api_key(self.llm_provider)

    @property
    def llm_model_resolved(self) -> str:
        """`llm_model` if set, else the default model for the provider."""
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]


settings = Settings()
