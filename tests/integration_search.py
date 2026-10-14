"""Busca em PostgreSQL real com vetores sintéticos e banco descartável."""
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from psycopg import sql
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ingestion import embeddings
from ingestion.config import Settings
from ingestion.service import IngestionService
from retrieval.service import SemanticSearchService, concatenate_context


class SyntheticEmbeddings:
    """Gera vetores determinísticos para comprovar ordenação sem API paga."""

    def __init__(self, **_kwargs):
        pass

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        vector = [0.0] * 1536
        if 'SuperTechIABrazil' in text or 'faturamento' in text.lower():
            vector[0] = 1.0
        else:
            vector[1] = 1.0
        return vector


def verify_search_with_fewer_than_ten_results(settings: Settings) -> None:
    """Dada collection com dois chunks, busca devolve dois e prioriza o pertinente."""
    # GIVEN
    chunks = [
        Document(
            page_content='SuperTechIABrazil R$ 10.000.000,00 2025',
            metadata={'page': 2, 'source': 'test.pdf'},
        ),
        Document(
            page_content='Outra empresa R$ 50,00 1990',
            metadata={'page': 1, 'source': 'test.pdf'},
        ),
    ]
    IngestionService(settings).ingest(chunks, 'synthetic-pdf-hash')

    # WHEN
    results = SemanticSearchService(settings).search(
        'Qual o faturamento da SuperTechIABrazil?',
    )

    # THEN
    assert len(results) == 2
    assert 'SuperTechIABrazil' in results[0][0].page_content
    assert '10.000.000,00' in concatenate_context(results)
    assert all(isinstance(score, float) for _document, score in results)
    print('PASS: k=10 devolveu somente os dois chunks existentes com scores.')


def main() -> None:
    # GIVEN: banco real exclusivo e provedor sintético.
    load_dotenv(ROOT / '.env')
    project_settings = Settings.from_env()
    database_url = project_settings.database_url
    test_database_name = 'rag_search_test_' + uuid.uuid4().hex
    admin_connection = psycopg.connect(
        host=database_url.host,
        port=database_url.port,
        user=database_url.username,
        password=database_url.password,
        dbname=database_url.database,
        autocommit=True,
    )
    database_created = False
    test_engine = None
    try:
        admin_connection.execute(
            sql.SQL('CREATE DATABASE {}').format(sql.Identifier(test_database_name))
        )
        database_created = True
        test_environment = {
            'POSTGRES_DB': test_database_name,
            'PGVECTOR_COLLECTION': 'search_integration',
            'OPENAI_API_KEY': 'test-local-only',
        }
        with patch.dict(os.environ, test_environment):
            settings = Settings.from_env()
        test_engine = create_engine(settings.database_url, hide_parameters=True)
        with test_engine.begin() as connection:
            connection.execute(text('CREATE EXTENSION vector'))

        # WHEN / THEN
        with patch.object(embeddings, 'OpenAIEmbeddings', SyntheticEmbeddings):
            verify_search_with_fewer_than_ten_results(settings)
    finally:
        # THEN: remove somente o banco criado para este teste.
        if test_engine is not None:
            test_engine.dispose()
        try:
            if database_created:
                admin_connection.execute(
                    sql.SQL('DROP DATABASE {}').format(
                        sql.Identifier(test_database_name)
                    )
                )
        finally:
            admin_connection.close()


if __name__ == '__main__':
    main()
