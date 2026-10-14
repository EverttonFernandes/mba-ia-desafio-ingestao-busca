"""Estimativa local de custo, validacao de vetores e integracao OpenAI."""
from __future__ import annotations

import math

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import Settings
from .errors import IngestError

# Preco consultado em 22/09/2026; estimativa, nao controle de faturamento.
PRICE_PER_MILLION = {"text-embedding-3-small": 0.02}


class EmbeddingService:
    """Mantem o cliente OpenAI e valida os vetores retornados."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAIEmbeddings(
            api_key=settings.api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            max_retries=0,
            request_timeout=60,
        )

    def generate(self, texts: list[str]) -> list[list[float]]:
        vectors = self.client.embed_documents(texts)
        validate_vectors(vectors, len(texts), self.settings.embedding_dimensions)
        return vectors


def estimate_cost(chunks: list[Document], embedding_model: str) -> float:
    """Estima o custo conservador em dolares, sem acessar a API."""
    if embedding_model not in PRICE_PER_MILLION:
        raise IngestError(
            "Preco deste modelo nao revisado. Use text-embedding-3-small "
            "ou revise a estimativa antes de gerar embeddings."
        )
    # Um token BPE ocupa ao menos um byte. Contar bytes UTF-8 superestima o custo
    # e dispensa download de tokenizer; inclui texto repetido pelo overlap.
    byte_count = sum(len(chunk.page_content.encode("utf-8")) for chunk in chunks)
    return byte_count * PRICE_PER_MILLION[embedding_model] / 1_000_000


def validate_vectors(
    vectors: list[list[float]], expected_count: int, expected_dimensions: int
) -> None:
    """Confere quantidade, tamanho e numeros antes de enviar o lote ao banco."""
    invalid_message = (
        "Provedor retornou vetores invalidos ou dimensao inesperada. Confira o modelo "
        "e OPENAI_EMBEDDING_DIMENSIONS; nada foi persistido."
    )
    if len(vectors) != expected_count:
        raise IngestError(invalid_message)
    for vector in vectors:
        if len(vector) != expected_dimensions:
            raise IngestError(invalid_message)
        if not all(math.isfinite(value) for value in vector):
            raise IngestError(invalid_message)


def validate_api_budget(settings: Settings, chunks: list[Document], max_cost: float) -> None:
    """Valida chave e estimativa apenas quando novos embeddings sao necessarios."""
    if not settings.api_key:
        raise IngestError("Defina OPENAI_API_KEY no .env local antes da primeira ingestao.")
    estimated_cost = estimate_cost(chunks, settings.embedding_model)
    if not math.isfinite(max_cost) or max_cost <= 0 or estimated_cost > max_cost:
        raise IngestError(
            "Estimativa excede --max-cost-usd ou limite invalido. Use --dry-run "
            "e revise o volume antes de autorizar um limite maior."
        )
