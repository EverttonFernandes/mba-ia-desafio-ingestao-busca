"""Contratos locais da busca semântica; nenhuma chamada paga ou banco real."""
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from sqlalchemy import URL
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ingestion.config import Settings
from retrieval.errors import (
    CollectionNotReadyError,
    IncompatibleCollectionError,
    InvalidQuestionError,
    InvalidSearchConfigurationError,
)
from retrieval.service import SemanticSearchService, concatenate_context
import search


def make_settings(**overrides) -> Settings:
    values = {
        'embedding_model': 'text-embedding-3-small',
        'embedding_dimensions': 1536,
        'collection_name': 'document_pdf',
        'api_key': 'local-test-only',
        'database_url': URL.create(
            'postgresql+psycopg', password='local-test-only',
        ),
    }
    values.update(overrides)
    return Settings(**values)


class SemanticSearchTest(unittest.TestCase):
    def setUp(self):
        repository_patcher = patch('retrieval.service.PgVectorRepository')
        embeddings_patcher = patch('retrieval.service.EmbeddingService')
        self.addCleanup(repository_patcher.stop)
        self.addCleanup(embeddings_patcher.stop)
        self.repository_class = repository_patcher.start()
        self.embeddings_class = embeddings_patcher.start()
        self.repository = self.repository_class.return_value
        self.connection = (
            self.repository.engine.begin.return_value.__enter__.return_value
        )
        self.collection_metadata = {
            'embedding_model': 'text-embedding-3-small',
            'embedding_dimensions': 1536,
        }
        self.repository.stored_collection.return_value = {
            'uuid': 'collection-id',
            'cmetadata': self.collection_metadata,
        }
        self.repository.collection_size.return_value = 2
        self.vector_store = self.repository.vector_store.return_value

    def test_busca_solicita_dez_resultados_e_preserva_documentos_e_scores(self):
        """Deve buscar k=10 e devolver apenas os resultados existentes."""
        # GIVEN
        question = 'Qual o faturamento da SuperTechIABrazil?'
        expected_results = [
            (Document(page_content='SuperTechIABrazil R$ 10.000.000,00'), 0.12),
            (Document(page_content='Outro trecho do PDF'), 0.45),
        ]
        self.vector_store.similarity_search_with_score.return_value = expected_results

        # WHEN
        results = SemanticSearchService(make_settings()).search(question)

        # THEN
        self.assertEqual(results, expected_results)
        self.vector_store.similarity_search_with_score.assert_called_once_with(
            question, k=10,
        )
        self.embeddings_class.assert_called_once_with(make_settings())
        self.repository.vector_store.assert_called_once_with(
            self.embeddings_class.return_value.client,
            self.collection_metadata,
        )
        self.repository.stored_collection.assert_called_once_with(
            self.connection, 'document_pdf',
        )
        self.repository.close.assert_called_once()

    def test_collection_com_dados_pode_retornar_nenhum_contexto(self):
        """Deve representar ausência de similaridade como lista vazia."""
        # GIVEN
        self.vector_store.similarity_search_with_score.return_value = []

        # WHEN
        results = SemanticSearchService(make_settings()).search('pergunta')

        # THEN
        self.assertEqual(results, [])

    def test_contexto_contem_somente_texto_dos_documentos_recuperados(self):
        """Deve concatenar chunks sem transformar score em probabilidade."""
        # GIVEN
        results = [
            (Document(page_content='Primeiro trecho.'), 0.2),
            (Document(page_content='Segundo trecho.'), 0.8),
        ]

        # WHEN
        context = concatenate_context(results)

        # THEN
        self.assertEqual(context, 'Primeiro trecho.\n\nSegundo trecho.')
        self.assertNotIn('0.2', context)
        self.assertNotIn('0.8', context)

    def test_pergunta_vazia_e_rejeitada_antes_de_acessar_banco_ou_api(self):
        """Deve diagnosticar pergunta vazia sem criar integrações externas."""
        # GIVEN
        empty_question = '   '

        # WHEN
        # THEN
        with self.assertRaisesRegex(InvalidQuestionError, 'vazia'):
            SemanticSearchService(make_settings()).search(empty_question)
        self.repository_class.assert_not_called()
        self.embeddings_class.assert_not_called()

    def test_collection_ausente_ou_vazia_e_distinguida_de_resultado_sem_contexto(self):
        """Deve orientar a ingestão quando a collection ainda não possui chunks."""
        for stored_collection, size in [(None, 0), ({
            'uuid': 'collection-id', 'cmetadata': self.collection_metadata,
        }, 0)]:
            with self.subTest(stored_collection=stored_collection):
                # GIVEN
                self.repository.stored_collection.return_value = stored_collection
                self.repository.collection_size.return_value = size

                # WHEN
                # THEN
                with self.assertRaisesRegex(CollectionNotReadyError, 'ingest'):
                    SemanticSearchService(make_settings()).search('pergunta')
                self.embeddings_class.assert_not_called()

    def test_collection_incompativel_e_rejeitada_antes_da_api(self):
        """Deve impedir busca com modelo ou dimensão diferentes da ingestão."""
        # GIVEN
        self.repository.stored_collection.return_value = {
            'uuid': 'collection-id',
            'cmetadata': {
                'embedding_model': 'outro-modelo',
                'embedding_dimensions': 3,
            },
        }

        # WHEN
        # THEN
        with self.assertRaisesRegex(IncompatibleCollectionError, 'incompatível'):
            SemanticSearchService(make_settings()).search('pergunta')
        self.embeddings_class.assert_not_called()

    def test_configuracao_ausente_e_rejeitada_antes_de_integracoes(self):
        """Deve exigir senha do banco e chave da API antes de buscar."""
        invalid_settings = [
            make_settings(
                database_url=URL.create('postgresql+psycopg', password=''),
            ),
            make_settings(api_key=''),
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                # GIVEN
                service = SemanticSearchService(settings)

                # WHEN
                # THEN
                with self.assertRaisesRegex(
                    InvalidSearchConfigurationError, 'Defina',
                ):
                    service.search('pergunta')
                self.repository_class.assert_not_called()

    def test_falha_na_busca_ainda_libera_conexao(self):
        """Deve liberar o engine quando o provedor ou o banco falhar."""
        # GIVEN
        self.vector_store.similarity_search_with_score.side_effect = RuntimeError(
            'falha externa',
        )

        # WHEN
        # THEN
        with self.assertRaises(RuntimeError):
            SemanticSearchService(make_settings()).search('pergunta')
        self.repository.close.assert_called_once()


class SearchCLITest(unittest.TestCase):
    def test_cli_identifica_indisponibilidade_do_postgresql_sem_expor_detalhes(self):
        """Deve distinguir falha do banco e ocultar os detalhes da conexão."""
        # GIVEN
        error_output = StringIO()
        database_error = SQLAlchemyError('password=secret-data host=internal')

        # WHEN
        with patch.object(search, 'run', side_effect=database_error):
            with redirect_stderr(error_output):
                exit_code = search.main(['pergunta'])

        # THEN
        self.assertEqual(exit_code, 1)
        self.assertIn('Falha no PostgreSQL', error_output.getvalue())
        self.assertNotIn('secret-data', error_output.getvalue())
        self.assertNotIn('internal', error_output.getvalue())

    def test_cli_oculta_detalhes_de_falha_inesperada(self):
        """Deve apresentar diagnóstico seguro sem imprimir erro externo."""
        # GIVEN
        error_output = StringIO()

        # WHEN
        with patch.object(search, 'run', side_effect=RuntimeError('secret-data')):
            with redirect_stderr(error_output):
                exit_code = search.main(['pergunta'])

        # THEN
        self.assertEqual(exit_code, 1)
        self.assertIn('Falha inesperada', error_output.getvalue())
        self.assertNotIn('secret-data', error_output.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
