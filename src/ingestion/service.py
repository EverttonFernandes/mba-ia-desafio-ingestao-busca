"""Orquestracao da ingestao, preservando a ordem dos efeitos externos."""
from langchain_core.documents import Document

from .config import Settings
from .errors import IngestError
from .pdf import collection_metadata, chunk_ids
from .embeddings import EmbeddingService, validate_api_budget
from .persistence import PgVectorRepository


class IngestionService:
    """Coordena as dependencias e garante a ordem dos efeitos externos."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def ingest(
        self,
        chunks: list[Document],
        pdf_hash: str,
        max_cost: float = 0.01,
    ) -> bool:
        settings = self.settings
        if not settings.database_url.password:
            raise IngestError("Defina POSTGRES_PASSWORD no .env local.")
        metadata = collection_metadata(settings, chunks, pdf_hash)
        ids = chunk_ids(settings.collection_name, metadata, chunks)
        repository = PgVectorRepository(settings)
        try:
            # O lock permanece ativo enquanto PGVector grava por outra conexao.
            with repository.engine.begin() as connection:
                repository.acquire_ingestion_lock(connection)
                repository.check_database(connection, settings)
                if repository.is_already_ingested(
                    connection, settings, metadata, ids, chunks,
                ):
                    return False

                validate_api_budget(settings, chunks, max_cost)
                self._generate_and_save(repository, metadata, ids, chunks)
                repository.confirm_saved(connection, ids, chunks)
                return True
        finally:
            repository.close()

    def _generate_and_save(
        self,
        repository: PgVectorRepository,
        metadata: dict,
        ids: list[str],
        chunks: list[Document],
    ) -> None:
        """Cria a integracao somente depois de aprovar o orcamento."""
        embeddings = EmbeddingService(self.settings)
        vector_store = repository.vector_store(embeddings.client, metadata)
        chunk_texts = [chunk.page_content for chunk in chunks]
        vectors = embeddings.generate(chunk_texts)
        repository.save(vector_store, chunk_texts, vectors, ids, chunks)
