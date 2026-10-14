"""Confere a ingestao real com chamadas OpenAI bloqueadas pelo teste."""
import sys
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import ingest
from ingestion import pdf, embeddings, persistence
from ingestion.service import IngestionService


def verify_existing_document(settings: ingest.Settings) -> None:
    # GIVEN
    chunks, file_hash = pdf.prepare_pdf(ROOT / 'document.pdf')
    forbidden_call = AssertionError('API proibida neste teste')

    # WHEN
    with patch.object(embeddings, 'OpenAIEmbeddings', side_effect=forbidden_call):
        created = IngestionService(settings).ingest(chunks, file_hash)

        # THEN
        assert created is False
    print('PASS: todos os textos/metadata/dimensoes conferem; nenhuma chamada OpenAI.')


def read_database_summary(settings: ingest.Settings) -> tuple:
    """Resume os dados para comparar antes/depois sem imprimir textos ou vetores."""
    # GIVEN
    summary_query = text("""
        SELECT count(*), min(vector_dims(e.embedding)), max(vector_dims(e.embedding)),
               md5(string_agg(
                   e.id || e.document || e.cmetadata::text || e.embedding::text,
                   '' ORDER BY e.id
               ))
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid=e.collection_id
        WHERE c.name=:name
    """)
    engine = create_engine(settings.database_url, hide_parameters=True)
    try:
        # WHEN
        with engine.connect() as connection:
            summary = connection.execute(
                summary_query, {'name': settings.collection_name}
            ).one()

            # THEN: devolve apenas numeros e checksum, sem texto ou vetores.
            return tuple(summary)
    finally:
        engine.dispose()


def main() -> None:
    # GIVEN
    load_dotenv(ROOT / '.env')
    settings = ingest.Settings.from_env()

    # WHEN
    verify_existing_document(settings)
    summary = read_database_summary(settings)

    # THEN
    print('chunks / min_dim / max_dim / checksum:', summary)


if __name__ == '__main__':
    main()
