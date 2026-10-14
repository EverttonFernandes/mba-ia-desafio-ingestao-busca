"""Testes locais: nenhuma chamada paga e nenhum banco necessario."""
import hashlib
import runpy
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from pypdf import PdfWriter
from sqlalchemy import URL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import ingest
from ingestion import pdf, embeddings, persistence
from ingestion.service import IngestionService


class IngestCLITest(unittest.TestCase):
    def test_pdf_conhecido_prepara_chunks_sem_api(self):
        # GIVEN
        command = [sys.executable, str(ROOT / 'src/ingest.py'), '--dry-run']

        # WHEN
        result = subprocess.run(
            command,
            cwd=ROOT, capture_output=True, text=True,
        )

        # THEN
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('chunks', result.stdout)
        self.assertIn('1000/150', result.stdout)
        self.assertIn('Nenhuma chamada', result.stdout)


class PreparationTest(unittest.TestCase):
    def test_refatoracao_preserva_identidade_dos_dados_ja_ingeridos(self):
        # GIVEN
        settings = make_settings(collection_name='document_pdf')

        # WHEN
        chunks, file_hash = pdf.prepare_pdf(ROOT / 'document.pdf')
        metadata = pdf.collection_metadata(settings, chunks, file_hash)
        ids = pdf.chunk_ids('document_pdf', metadata, chunks)
        ids_digest = hashlib.sha256('\n'.join(ids).encode()).hexdigest()

        # THEN: valores anteriores impedem reingestao acidental.
        self.assertEqual(
            metadata['chunks_sha256'],
            '07bb81e26a14b7490969a9d27db9f4f6d9a7c5a9fc894320f2f5009435675f30',
        )
        self.assertEqual(
            ids_digest,
            'df6ee56bda2b3d2acbf7dc0f9a23df2db91394ec38058092a00fbdaaf82daa8a',
        )

    def test_pdf_real_tem_origem_pagina_texto_e_chunks_limitados(self):
        # GIVEN
        document_path = ROOT / 'document.pdf'

        # WHEN
        chunks, digest = pdf.prepare_pdf(document_path)

        # THEN
        self.assertGreater(len(chunks), 34)
        self.assertEqual(digest, '487adc82839c525365caa5feef4c7de01be10d3375ee43c9a2dfd24331274ca8')
        self.assertTrue(all(0 < len(chunk.page_content) <= 1000 for chunk in chunks))
        self.assertTrue(all(chunk.metadata['source'] == 'document.pdf' for chunk in chunks))
        self.assertEqual({chunk.metadata['page'] for chunk in chunks}, set(range(34)))
        company_on_page_three = any(
            'SuperTechIABrazil' in chunk.page_content and chunk.metadata['page'] == 2
            for chunk in chunks
        )
        self.assertTrue(company_on_page_three)

    def test_overlap_em_texto_sem_separadores_e_150(self):
        # GIVEN
        content = ''.join(chr(0x4E00 + i) for i in range(2100))
        with patch.object(pdf, 'PyPDFLoader') as loader:
            loader.return_value.load.return_value = [
                Document(page_content=content, metadata={'page': 0})
            ]

            # WHEN
            chunks, _ = pdf.prepare_pdf(ROOT / 'document.pdf')

        # THEN
        self.assertEqual([len(chunk.page_content) for chunk in chunks], [1000, 1000, 400])
        self.assertEqual(chunks[0].page_content[-150:], chunks[1].page_content[:150])
        self.assertEqual(chunks[1].metadata['start_index'], 850)

    def test_paginas_vazias_sao_ignoradas_e_metadata_do_loader_nao_e_persistida(self):
        # GIVEN
        pages = [
            Document(page_content='   ', metadata={'page': 0}),
            Document(
                page_content='Texto da segunda pagina.',
                metadata={'page': 1, 'source': '/local/privado/document.pdf', 'author': 'autor'},
            ),
        ]
        with patch.object(pdf, 'PyPDFLoader') as loader:
            loader.return_value.load.return_value = pages

            # WHEN
            chunks, pdf_hash = pdf.prepare_pdf(ROOT / 'document.pdf')

        # THEN
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, 'Texto da segunda pagina.')
        self.assertEqual(chunks[0].metadata, {
            'source': 'document.pdf',
            'page': 1,
            'pdf_sha256': pdf_hash,
            'start_index': 0,
            'chunk_index': 0,
        })

    def test_pdf_ausente(self):
        # GIVEN
        missing_pdf = ROOT / 'arquivo-inexistente.pdf'

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'ausente'):
            pdf.prepare_pdf(missing_pdf)

    def test_pdf_invalido_e_sem_texto(self):
        with tempfile.TemporaryDirectory() as folder:
            # GIVEN: primeiro um arquivo invalido.
            path = Path(folder) / 'test.pdf'
            path.write_bytes(b'%PDF-1.7\ninvalid')

            # WHEN
            # THEN
            with self.assertRaisesRegex(ingest.IngestError, 'ler o PDF'):
                pdf.prepare_pdf(path)

            # GIVEN: depois um PDF valido, mas sem texto.
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.write(path)

            # WHEN
            # THEN
            with self.assertRaisesRegex(ingest.IngestError, 'sem texto'):
                pdf.prepare_pdf(path)


def make_settings(**overrides):
    """Configuracao de teste sem ler credenciais reais do ambiente."""
    values = {
        'embedding_model': 'text-embedding-3-small',
        'embedding_dimensions': 1536,
        'collection_name': 'test',
        'api_key': 'local-test-only',
        'database_url': URL.create('postgresql+psycopg', password='local-test-only'),
    }
    values.update(overrides)
    return ingest.Settings(**values)


def make_repository():
    """Instancia o repositorio sem criar conexoes reais."""
    with patch.object(persistence, 'create_engine'):
        return persistence.PgVectorRepository(make_settings())


class ValidationTest(unittest.TestCase):
    def test_configuracao_preserva_valores_personalizados_do_ambiente(self):
        # GIVEN
        test_environment = {
            'OPENAI_EMBEDDING_MODEL': ' modelo-local ',
            'OPENAI_EMBEDDING_DIMENSIONS': '3',
            'PGVECTOR_COLLECTION': ' collection-local ',
            'OPENAI_API_KEY': ' chave-local ',
            'POSTGRES_HOST': 'banco-local',
            'POSTGRES_PORT': '6543',
            'POSTGRES_DB': 'database-local',
            'POSTGRES_USER': 'usuario-local',
            'POSTGRES_PASSWORD': 'senha-local',
        }

        # WHEN
        with patch.dict('os.environ', test_environment, clear=True):
            settings = ingest.Settings.from_env()

        # THEN
        self.assertEqual(settings.embedding_model, 'modelo-local')
        self.assertEqual(settings.embedding_dimensions, 3)
        self.assertEqual(settings.collection_name, 'collection-local')
        self.assertEqual(settings.api_key, 'chave-local')
        self.assertEqual(settings.database_url, URL.create(
            'postgresql+psycopg',
            host='banco-local', port=6543, database='database-local',
            username='usuario-local', password='senha-local',
        ))

    def test_config_esconde_senha_e_aceita_caracteres_especiais(self):
        # GIVEN
        test_environment = {
            'POSTGRES_PASSWORD': 'a@/:?#secret',
            'OPENAI_API_KEY': 'secret-key',
        }

        # WHEN
        with patch.dict('os.environ', test_environment, clear=True):
            config = ingest.Settings.from_env()

        # THEN
        self.assertEqual(config.database_url.password, 'a@/:?#secret')
        self.assertNotIn('secret', repr(config))

    def test_config_invalida(self):
        # GIVEN
        invalid_values = [
            ('POSTGRES_PORT', 'x'),
            ('POSTGRES_PORT', '0'),
            ('POSTGRES_PORT', '65536'),
            ('OPENAI_EMBEDDING_DIMENSIONS', '-1'),
            ('PGVECTOR_COLLECTION', ''),
            ('OPENAI_EMBEDDING_MODEL', '   '),
        ]
        for variable, value in invalid_values:
            with self.subTest(variable=variable, value=value):
                # WHEN
                # THEN
                with patch.dict('os.environ', {variable: value}, clear=True):
                    with self.assertRaises(ingest.IngestError):
                        ingest.Settings.from_env()

    def test_vetores_invalidos(self):
        # GIVEN
        invalid_vectors = [[], [[1.0]], [[float('nan'), 1]], [[float('inf'), 0]]]

        for vectors in invalid_vectors:
            # WHEN
            # THEN
            with self.subTest(vectors=vectors), self.assertRaises(ingest.IngestError):
                embeddings.validate_vectors(vectors, 1, 2)

        # WHEN
        # THEN: um vetor valido nao gera erro.
        embeddings.validate_vectors([[0.1, 0.2]], 1, 2)

    def test_sem_extensao_e_dimensao_incompativel(self):
        # GIVEN: banco sem a extensao vector.
        repository = make_repository()
        connection = Mock()
        connection.execute.return_value.scalar_one.return_value = False

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'Extensao'):
            repository.check_database(connection, make_settings())
        connection.execute.assert_called_once()

        # GIVEN: extensao presente e coluna com outra dimensao.
        connection.reset_mock()
        connection.execute.return_value.scalar_one.return_value = True
        connection.execute.return_value.scalar_one_or_none.return_value = 'vector(3)'

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'Dimensao'):
            repository.check_database(connection, make_settings())

    def test_coluna_ausente_ou_com_dimensao_compativel_e_aceita(self):
        # GIVEN
        repository = make_repository()
        for vector_type in [None, 'vector', 'vector(1536)']:
            with self.subTest(vector_type=vector_type):
                connection = Mock()
                connection.execute.return_value.scalar_one.return_value = True
                connection.execute.return_value.scalar_one_or_none.return_value = vector_type

                # WHEN
                # THEN: configuracoes compativeis nao geram erro.
                repository.check_database(connection, make_settings())

    def test_confirma_textos_metadata_e_dimensao_nao_so_contagem(self):
        # GIVEN
        repository = make_repository()
        chunk = Document(page_content='texto', metadata={'page': 0})
        row = dict(id='id', document='texto', cmetadata={'page': 0}, dimensions=1536)
        connection = Mock()
        result = connection.execute.return_value.mappings.return_value
        result.all.return_value = [row]

        # WHEN
        rows_are_valid = repository.verify_rows(
            connection, 'uuid', ['id'], [chunk], 1536,
        )

        # THEN
        self.assertTrue(rows_are_valid)

        # GIVEN
        inconsistent_values = [
            ('document', 'outro'),
            ('id', 'outro'),
            ('dimensions', 3),
            ('cmetadata', {}),
        ]
        for field, changed in inconsistent_values:
            result.all.return_value = [{**row, field: changed}]

            # WHEN
            # THEN
            with self.subTest(field=field), self.assertRaises(ingest.IngestError):
                repository.verify_rows(connection, 'uuid', ['id'], [chunk], 1536)

        # GIVEN
        result.all.return_value = []

        # WHEN
        rows_are_valid = repository.verify_rows(
            connection, 'uuid', ['id'], [chunk], 1536,
        )

        # THEN
        self.assertFalse(rows_are_valid)

    def test_quantidade_parcial_de_registros_bloqueia_sucesso(self):
        # GIVEN
        repository = make_repository()
        connection = Mock()
        connection.execute.return_value.mappings.return_value.all.return_value = [
            dict(id='primeiro', document='texto', cmetadata={}, dimensions=1536),
        ]
        chunks = [Document(page_content='texto'), Document(page_content='segundo')]

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'parcial ou inconsistente'):
            repository.verify_rows(
                connection, 'uuid', ['primeiro', 'segundo'], chunks, 1536,
            )

    def test_banco_sem_tabela_de_collection_retorna_ausencia(self):
        # GIVEN
        repository = make_repository()
        connection = Mock()
        connection.execute.return_value.scalar_one.return_value = False

        # WHEN
        stored_collection = repository.stored_collection(connection, 'test')

        # THEN
        self.assertIsNone(stored_collection)
        # A ausencia no catalogo evita consultar uma tabela que ainda nao existe.
        connection.execute.assert_called_once()

    def test_fingerprint_e_ids_distinguem_documento_modelo_e_collection(self):
        # GIVEN
        chunks = [Document(page_content='teste', metadata={'page': 0})]

        # WHEN
        original = pdf.collection_metadata(make_settings(), chunks, 'hash')
        altered = pdf.collection_metadata(
            make_settings(embedding_model='other'), chunks, 'hash'
        )

        # THEN
        self.assertNotEqual(original, altered)
        self.assertNotEqual(
            pdf.chunk_ids('a', original, chunks),
            pdf.chunk_ids('b', original, chunks),
        )
        self.assertNotEqual(
            pdf.chunk_ids('a', original, chunks),
            pdf.chunk_ids('a', altered, chunks),
        )


class PipelineTest(unittest.TestCase):
    """Falhas e ordem de efeitos; mocks nao substituem integracao PostgreSQL real."""
    def setUp(self):
        # GIVEN compartilhado pelos cenarios do pipeline.
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.mock_engine_factory = self.stack.enter_context(
            patch.object(persistence, 'create_engine')
        )
        self.mock_check_database = self.stack.enter_context(
            patch.object(persistence.PgVectorRepository, 'check_database')
        )
        self.mock_stored_collection = self.stack.enter_context(
            patch.object(persistence.PgVectorRepository, 'stored_collection')
        )
        self.mock_verify_rows = self.stack.enter_context(
            patch.object(persistence.PgVectorRepository, 'verify_rows')
        )
        self.mock_embeddings = self.stack.enter_context(
            patch.object(embeddings, 'OpenAIEmbeddings')
        )
        self.mock_vector_store = self.stack.enter_context(patch.object(persistence, 'PGVector'))
        engine = self.mock_engine_factory.return_value
        self.connection = engine.begin.return_value.__enter__.return_value
        self.connection.execute.return_value.scalar_one.return_value = True
        self.chunks = [Document(page_content='texto', metadata={'page': 0})]
        self.collection_metadata = pdf.collection_metadata(make_settings(), self.chunks, 'hash')
        self.mock_stored_collection.return_value = {
            'uuid': 'uuid',
            'cmetadata': self.collection_metadata,
        }
        self.mock_verify_rows.return_value = True

    def test_repetir_pdf_persistido_nao_instancia_api(self):
        # GIVEN
        settings_without_api_key = make_settings(api_key='')

        # WHEN
        created = IngestionService(settings_without_api_key).ingest(
            self.chunks, 'hash',
        )

        # THEN
        self.assertFalse(created)
        self.mock_embeddings.assert_not_called()
        self.mock_vector_store.assert_not_called()

    def test_collection_incompativel_bloqueia_api_e_escrita(self):
        # GIVEN
        self.mock_stored_collection.return_value = {'uuid': 'uuid', 'cmetadata': {}}

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'Collection incompativel'):
            IngestionService(make_settings()).ingest(self.chunks, 'hash')

        # THEN
        self.mock_embeddings.assert_not_called()
        self.mock_vector_store.assert_not_called()

    def test_concorrencia_bloqueia_api(self):
        # GIVEN
        self.connection.execute.return_value.scalar_one.return_value = False

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'Outra ingestao'):
            IngestionService(make_settings()).ingest(self.chunks, 'hash')

        # THEN
        self.mock_embeddings.assert_not_called()

    def test_limite_custo_bloqueia_antes_da_api(self):
        # GIVEN
        self.mock_stored_collection.return_value = None

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'Estimativa'):
            IngestionService(make_settings()).ingest(self.chunks, 'hash', max_cost=0.000000001)

        # THEN
        self.mock_embeddings.assert_not_called()

    def test_api_falha_sem_persistir(self):
        # GIVEN
        self.mock_stored_collection.return_value = None
        self.mock_embeddings.return_value.embed_documents.side_effect = RuntimeError('failure')

        # WHEN
        # THEN
        with self.assertRaises(RuntimeError):
            IngestionService(make_settings()).ingest(self.chunks, 'hash')

        # THEN
        self.mock_vector_store.return_value.add_embeddings.assert_not_called()
        self.mock_engine_factory.return_value.dispose.assert_called_once()

    def test_vetores_invalidos_bloqueiam_gravacao_e_liberam_conexao(self):
        # GIVEN
        self.mock_stored_collection.return_value = None
        self.mock_embeddings.return_value.embed_documents.return_value = [[0.1]]

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'vetores invalidos'):
            IngestionService(make_settings()).ingest(self.chunks, 'hash')

        # THEN
        self.mock_vector_store.return_value.add_embeddings.assert_not_called()
        self.mock_engine_factory.return_value.dispose.assert_called_once()

    def test_sem_chave_bloqueia_cliente_e_escrita(self):
        # GIVEN
        self.mock_stored_collection.return_value = None

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'OPENAI_API_KEY'):
            IngestionService(make_settings(api_key='')).ingest(self.chunks, 'hash')

        # THEN
        self.mock_embeddings.assert_not_called()
        self.mock_vector_store.assert_not_called()

    def test_persistir_lote_e_confirmar_antes_de_sucesso(self):
        # GIVEN
        self.mock_stored_collection.side_effect = [
            None,
            {'uuid': 'uuid', 'cmetadata': self.collection_metadata},
        ]
        self.mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536]

        # WHEN
        created = IngestionService(make_settings()).ingest(self.chunks, 'hash')

        # THEN
        self.assertTrue(created)
        self.mock_vector_store.return_value.add_embeddings.assert_called_once()
        self.mock_verify_rows.assert_called_once()
        self.assertEqual(self.mock_embeddings.call_args.kwargs['max_retries'], 0)

    def test_nao_anuncia_sucesso_se_releitura_falhar(self):
        # GIVEN
        self.mock_stored_collection.side_effect = [None, None]
        self.mock_embeddings.return_value.embed_documents.return_value = [[0.1] * 1536]

        # WHEN
        # THEN
        with self.assertRaisesRegex(ingest.IngestError, 'Persistencia nao confirmada'):
            IngestionService(make_settings()).ingest(self.chunks, 'hash')


class TestScriptImportTest(unittest.TestCase):
    def test_cli_oculta_detalhes_de_falha_inesperada(self):
        # GIVEN
        output = StringIO()

        # WHEN
        with patch.object(ingest, 'run', side_effect=RuntimeError('secret-key')):
            with redirect_stderr(output):
                result = ingest.main(['--dry-run'])

        # THEN
        self.assertEqual(result, 1)
        self.assertIn('Falha inesperada', output.getvalue())
        self.assertNotIn('secret-key', output.getvalue())

    def test_importar_scripts_nao_executa_integracao(self):
        # GIVEN
        for script_name in ['integration_ingest.py', 'verify_persistence.py']:
            with self.subTest(script=script_name):
                with patch.object(ingest.Settings, 'from_env') as load_settings:
                    # WHEN
                    runpy.run_path(str(ROOT / 'tests' / script_name))

                # THEN
                load_settings.assert_not_called()


if __name__ == '__main__':
    unittest.main()
