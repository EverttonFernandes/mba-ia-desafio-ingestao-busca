"""Configuracao imutavel obtida do ambiente, sem abrir conexoes."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

from sqlalchemy import URL

from .errors import IngestError

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Agrupa a configuracao sem alteracoes posteriores (frozen=True).

    repr=False evita mostrar credenciais ao inspecionar este objeto.
    """

    embedding_model: str
    embedding_dimensions: int
    collection_name: str
    api_key: str = field(repr=False)
    database_url: URL = field(repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        embedding_dimensions, database_port = cls._read_numeric_settings()
        embedding_model, collection_name = cls._read_embedding_settings()
        return cls(
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            collection_name=collection_name,
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            database_url=cls._read_database_url(database_port),
        )

    @staticmethod
    def _read_numeric_settings() -> tuple[int, int]:
        try:
            embedding_dimensions = int(
                os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")
            )
            database_port = int(os.getenv("POSTGRES_PORT", "5432"))
            if embedding_dimensions <= 0 or not 1 <= database_port <= 65535:
                raise ValueError
        except ValueError:
            raise IngestError(
                "Confira OPENAI_EMBEDDING_DIMENSIONS e POSTGRES_PORT: "
                "use inteiros positivos validos."
            ) from None
        return embedding_dimensions, database_port

    @staticmethod
    def _read_embedding_settings() -> tuple[str, str]:
        embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ).strip()
        collection_name = os.getenv("PGVECTOR_COLLECTION", "document_pdf").strip()
        if not embedding_model or not collection_name:
            raise IngestError(
                "OPENAI_EMBEDDING_MODEL e PGVECTOR_COLLECTION nao podem estar vazios."
            )
        return embedding_model, collection_name

    @staticmethod
    def _read_database_url(database_port: int) -> URL:
        return URL.create(
            "postgresql+psycopg",
            username=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=database_port,
            database=os.getenv("POSTGRES_DB", "rag"),
        )
