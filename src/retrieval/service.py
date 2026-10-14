"""Recuperação de documentos similares e montagem do contexto."""
from __future__ import annotations

from langchain_core.documents import Document

from ingestion.config import Settings
from ingestion.embeddings import EmbeddingService
from ingestion.persistence import PgVectorRepository

from .errors import (
    CollectionNotReadyError,
    IncompatibleCollectionError,
    InvalidQuestionError,
    InvalidSearchConfigurationError,
)

TOP_K = 10
SearchResults = list[tuple[Document, float]]


class SemanticSearchService:
    """Confere a collection e recupera somente seus documentos similares."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def search(self, question: str) -> SearchResults:
        normalized_question = self._validate_question(question)
        self._validate_configuration()
        repository = PgVectorRepository(self.settings)
        try:
            collection_metadata = self._read_collection_metadata(repository)
            embeddings = EmbeddingService(self.settings)
            vector_store = repository.vector_store(
                embeddings.client, collection_metadata,
            )
            return vector_store.similarity_search_with_score(
                normalized_question, k=TOP_K,
            )
        finally:
            repository.close()

    def _validate_question(self, question: str) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise InvalidQuestionError(
                "Pergunta vazia. Informe o que deseja buscar no PDF."
            )
        return normalized_question

    def _validate_configuration(self) -> None:
        if not self.settings.database_url.password:
            raise InvalidSearchConfigurationError(
                "Defina POSTGRES_PASSWORD no .env local."
            )
        if not self.settings.api_key:
            raise InvalidSearchConfigurationError(
                "Defina OPENAI_API_KEY no .env local antes da busca."
            )

    def _read_collection_metadata(
        self, repository: PgVectorRepository,
    ) -> dict:
        with repository.engine.begin() as connection:
            repository.check_database(connection, self.settings)
            stored_collection = repository.stored_collection(
                connection, self.settings.collection_name,
            )
            if stored_collection is None:
                raise CollectionNotReadyError(
                    "Collection ausente. Execute a ingestão antes da busca."
                )
            self._validate_collection_metadata(stored_collection["cmetadata"])
            if repository.collection_size(
                connection, stored_collection["uuid"],
            ) == 0:
                raise CollectionNotReadyError(
                    "Collection vazia. Execute a ingestão antes da busca."
                )
            return stored_collection["cmetadata"]

    def _validate_collection_metadata(self, metadata: dict | None) -> None:
        expected_metadata = {
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
        }
        if not metadata or any(
            metadata.get(name) != value
            for name, value in expected_metadata.items()
        ):
            raise IncompatibleCollectionError(
                "Collection incompatível com o modelo ou a dimensão configurados. "
                "Use a mesma configuração empregada na ingestão."
            )


def concatenate_context(results: SearchResults) -> str:
    """Concatena somente textos recuperados; scores continuam dados separados."""
    return "\n\n".join(document.page_content for document, _score in results)
