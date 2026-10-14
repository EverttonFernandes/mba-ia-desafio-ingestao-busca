"""Persistencia PGVector, lock transacional e verificacao de integridade."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, RowMapping

from .config import Settings
from .errors import IngestError

INGESTION_LOCK_ID = 724001


class PgVectorRepository:
    """Possui o engine, o lock e as operacoes de integridade e persistencia."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = create_engine(
            settings.database_url, hide_parameters=True,
            connect_args={
                "connect_timeout": 5,
                "options": "-c search_path=public -c statement_timeout=30000",
            },
        )

    def acquire_ingestion_lock(self, connection: Connection) -> None:
        """Impede ingestao concorrente; o PostgreSQL libera o lock ao fim da transacao."""
        lock_query = text("SELECT pg_try_advisory_xact_lock(:lock_id)")
        acquired = connection.execute(lock_query, {"lock_id": INGESTION_LOCK_ID}).scalar_one()
        if not acquired:
            raise IngestError(
                "Outra ingestao esta em andamento neste banco. "
                "Aguarde antes de tentar novamente."
            )

    def check_database(self, connection: Connection, settings: Settings) -> None:
        """Recusa extensao ausente ou coluna incompatível antes de gerar embeddings."""
        self._check_vector_extension(connection)
        self._check_vector_column(connection, settings.embedding_dimensions)

    def _check_vector_extension(self, connection: Connection) -> None:
        extension_query = text(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')"
        )
        extension_installed = connection.execute(extension_query).scalar_one()
        if not extension_installed:
            raise IngestError(
                "Extensao vector ausente. Execute docker compose up -d "
                "e confira o bootstrap."
            )

    def _check_vector_column(
        self, connection: Connection, expected_dimensions: int,
    ) -> None:
        vector_type = connection.execute(text("""
            SELECT format_type(atttypid, atttypmod)
            FROM pg_attribute
            WHERE attrelid=to_regclass('public.langchain_pg_embedding')
              AND attname='embedding' AND NOT attisdropped
        """)).scalar_one_or_none()
        expected_vector_types = {
            None,
            "vector",
            f"vector({expected_dimensions})",
        }
        if vector_type not in expected_vector_types:
            raise IngestError(
                "Dimensao da coluna vector incompativel. Use um banco/schema compativel; "
                "mudar so a collection nao altera a coluna. Nenhum dado foi apagado."
            )

    def is_already_ingested(
        self,
        connection: Connection,
        settings: Settings,
        expected_metadata: dict,
        expected_ids: list[str],
        expected_chunks: list[Document],
    ) -> bool:
        """Reaproveita somente uma collection compativel com todos os chunks intactos."""
        stored = self.stored_collection(connection, settings.collection_name)
        if stored is None:
            return False
        if stored["cmetadata"] != expected_metadata:
            raise IngestError(
                "Collection incompativel com PDF, modelo, dimensao ou chunks atuais. "
                "Use outra PGVECTOR_COLLECTION; dimensao diferente exige banco/schema "
                "compativel. Nenhum dado foi apagado."
            )
        return self.verify_rows(
            connection,
            stored["uuid"],
            expected_ids,
            expected_chunks,
            settings.embedding_dimensions,
        )

    def vector_store(self, embeddings, metadata: dict) -> PGVector:
        return PGVector(
            embeddings=embeddings, connection=self.engine,
            embedding_length=self.settings.embedding_dimensions,
            collection_name=self.settings.collection_name,
            collection_metadata=metadata, use_jsonb=True,
            create_extension=False, pre_delete_collection=False,
        )

    def save(
        self,
        store: PGVector,
        texts: list[str],
        vectors: list[list[float]],
        ids: list[str],
        chunks: list[Document],
    ) -> None:
        # PGVector confirma o lote em uma transacao.
        store.add_embeddings(
            texts=texts, embeddings=vectors, ids=ids,
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def confirm_saved(
        self,
        connection: Connection,
        expected_ids: list[str],
        expected_chunks: list[Document],
    ) -> None:
        """Reler pela conexao do lock comprova o commit feito pelo PGVector."""
        stored = self.stored_collection(connection, self.settings.collection_name)
        if stored is None or not self.verify_rows(
            connection,
            stored["uuid"],
            expected_ids,
            expected_chunks,
            self.settings.embedding_dimensions,
        ):
            raise IngestError(
                "Persistencia nao confirmada. "
                "Confira o banco antes de repetir a ingestao."
            )

    def stored_collection(self, connection: Connection, name: str) -> RowMapping | None:
        """Consulta a collection; None significa que ela ainda nao foi criada."""
        if not self._collection_table_exists(connection):
            return None
        collection_query = text("""
            SELECT uuid, cmetadata FROM public.langchain_pg_collection WHERE name=:name
        """)
        return connection.execute(collection_query, {"name": name}).mappings().one_or_none()

    def collection_size(self, connection: Connection, collection_id) -> int:
        """Conta somente os embeddings pertencentes à collection informada."""
        count_query = text("""
            SELECT count(*) FROM public.langchain_pg_embedding
            WHERE collection_id=:collection_id
        """)
        return connection.execute(
            count_query, {"collection_id": collection_id},
        ).scalar_one()

    def _collection_table_exists(self, connection: Connection) -> bool:
        # Consultar o catalogo evita reutilizar a ausencia de uma relacao resolvida
        # antes de outra conexao (PGVector) criar e confirmar as tabelas.
        table_exists_query = text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_catalog.pg_tables
                WHERE schemaname='public' AND tablename='langchain_pg_collection'
            )
        """)
        return connection.execute(table_exists_query).scalar_one()

    def verify_rows(
        self,
        connection: Connection,
        collection_id,
        expected_ids: list[str],
        expected_chunks: list[Document],
        expected_dimensions: int,
    ) -> bool:
        """Retorna False se vazia; dados parciais ou diferentes geram erro."""
        rows = self._read_rows(connection, collection_id)
        if not rows:
            return False
        chunks_by_id = dict(zip(expected_ids, expected_chunks))
        self._validate_rows(rows, chunks_by_id, expected_dimensions)
        return True

    def _read_rows(self, connection: Connection, collection_id) -> list[RowMapping]:
        rows_query = text("""
            SELECT id, document, cmetadata, vector_dims(embedding) AS dimensions
            FROM public.langchain_pg_embedding WHERE collection_id=:collection_id
        """)
        return list(connection.execute(
            rows_query, {"collection_id": collection_id},
        ).mappings().all())

    def _validate_rows(
        self,
        rows: list[RowMapping],
        chunks_by_id: dict[str, Document],
        expected_dimensions: int,
    ) -> None:
        inconsistent_message = (
            "Collection parcial ou inconsistente. Revise os registros no pgAdmin; "
            "nenhuma reingestao ou exclusao automatica foi feita."
        )
        if len(rows) != len(chunks_by_id):
            raise IngestError(inconsistent_message)

        for row in rows:
            chunk = chunks_by_id.get(row["id"])
            if chunk is None:
                raise IngestError(inconsistent_message)
            if (
                row["document"] != chunk.page_content
                or row["cmetadata"] != chunk.metadata
                or row["dimensions"] != expected_dimensions
            ):
                raise IngestError(inconsistent_message)

    def close(self) -> None:
        self.engine.dispose()
