"""PostgreSQL real em banco descartavel; embeddings sinteticos, nenhuma API paga.

Executar explicitamente: venv/bin/python tests/integration_ingest.py
Cria/remove somente seu proprio banco rag_test_<uuid>; nunca o banco do projeto.
"""
import os
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from psycopg import sql
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import ingest
from ingestion import pdf, embeddings, persistence
from ingestion.service import IngestionService


class SyntheticEmbeddings:
    """Substitui a OpenAI por vetores previsiveis e conta chamadas."""

    calls = 0

    def __init__(self, **kwargs):
        # Aceita os mesmos argumentos usados para construir OpenAIEmbeddings.
        pass

    def embed_documents(self, texts):
        type(self).calls += 1
        return [[0.1] * 1536 for _ in texts]


def assert_ingestion_rejected(settings, chunks):
    """Uma configuracao ou collection invalida precisa gerar erro."""
    # WHEN
    try:
        IngestionService(settings).ingest(chunks, 'test-hash')
    except ingest.IngestError:
        # THEN
        return
    # THEN
    raise AssertionError('A ingestao deveria rejeitar os dados incompatíveis.')


def verify_new_ingestion_and_repetition(settings, chunks):
    # GIVEN: banco descartavel sem tabelas do LangChain.

    # WHEN
    created = IngestionService(settings).ingest(chunks, 'test-hash')

    # THEN
    assert created is True
    assert SyntheticEmbeddings.calls == 1
    print('PASS: primeira ingestao em banco sem tabelas confirma commit e conteudos.')

    # GIVEN: os mesmos chunks ja estao persistidos.

    # WHEN
    created_again = IngestionService(settings).ingest(chunks, 'test-hash')

    # THEN
    assert created_again is False
    assert SyntheticEmbeddings.calls == 1
    print('PASS: repeticao nao gera embeddings nem duplica registros.')


def verify_incompatible_settings(settings, chunks, engine):
    # GIVEN: copias da configuracao com modelo ou dimensao alterados.
    different_model = replace(settings, embedding_model='modelo-diferente')
    different_dimensions = replace(settings, embedding_dimensions=3)

    # WHEN
    assert_ingestion_rejected(different_model, chunks)
    assert_ingestion_rejected(different_dimensions, chunks)

    # THEN
    assert SyntheticEmbeddings.calls == 1

    with engine.connect() as connection:
        row_count = connection.execute(
            text('SELECT count(*) FROM langchain_pg_embedding')
        ).scalar_one()
    assert row_count == 1
    print('PASS: modelo/dimensao diferentes bloqueados sem API nem exclusao.')


def verify_modified_text_is_rejected(settings, chunks, engine):
    # GIVEN: o texto persistido no banco descartavel foi alterado.
    with engine.begin() as connection:
        connection.execute(text("UPDATE langchain_pg_embedding SET document='alterado'"))

    # WHEN
    assert_ingestion_rejected(settings, chunks)

    # THEN
    assert SyntheticEmbeddings.calls == 1
    print('PASS: texto inconsistente bloqueia sucesso e nao regenera embeddings.')


def main():
    # GIVEN: PostgreSQL real e um banco exclusivo para esta execucao.
    load_dotenv(ROOT / '.env')
    project_settings = ingest.Settings.from_env()
    database_url = project_settings.database_url
    test_database_name = 'rag_test_' + uuid.uuid4().hex
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
        create_database = sql.SQL('CREATE DATABASE {}').format(
            sql.Identifier(test_database_name)
        )
        admin_connection.execute(create_database)
        database_created = True

        test_environment = {
            'POSTGRES_DB': test_database_name,
            'PGVECTOR_COLLECTION': 'integration',
            'OPENAI_API_KEY': 'test-local-only',
        }
        with patch.dict(os.environ, test_environment):
            test_settings = ingest.Settings.from_env()

        test_engine = create_engine(test_settings.database_url, hide_parameters=True)
        with test_engine.begin() as connection:
            connection.execute(text('CREATE EXTENSION vector'))

        chunks = [
            Document(
                page_content='Empresa teste: faturamento 10 milhoes.',
                metadata={'page': 0, 'source': 'test.pdf'},
            )
        ]
        SyntheticEmbeddings.calls = 0

        # WHEN
        # THEN: cada funcao executa e verifica um cenario de integracao.
        with patch.object(embeddings, 'OpenAIEmbeddings', SyntheticEmbeddings):
            verify_new_ingestion_and_repetition(test_settings, chunks)
            verify_incompatible_settings(test_settings, chunks, test_engine)
            verify_modified_text_is_rejected(test_settings, chunks, test_engine)
    finally:
        # THEN: remove somente o banco descartavel criado no GIVEN.
        if test_engine is not None:
            test_engine.dispose()
        try:
            if database_created:
                drop_database = sql.SQL('DROP DATABASE {}').format(
                    sql.Identifier(test_database_name)
                )
                admin_connection.execute(drop_database)
        finally:
            admin_connection.close()


if __name__ == '__main__':
    main()
