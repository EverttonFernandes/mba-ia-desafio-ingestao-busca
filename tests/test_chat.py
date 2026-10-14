"""Contratos da sessão CLI sem chamadas OpenAI ou PostgreSQL."""
import sys
import subprocess
import textwrap
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, call

from openai import APIConnectionError
from sqlalchemy.exc import SQLAlchemyError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import chat
from answering.model import AnswerError
from answering.prompt import REFUSAL


class ChatSessionTest(unittest.TestCase):
    def test_branco_saida_e_eof_nao_criam_servico(self):
        """Deve ignorar espaços e sair por comando ou EOF sem integrações."""
        # GIVEN
        for last_input in [' SAIR ', EOFError()]:
            output = StringIO()
            # WHEN
            with (
                self.subTest(last_input=type(last_input).__name__),
                patch('builtins.input', side_effect=['', '   ', last_input]),
                patch('chat.create_answer_service') as create,
                redirect_stdout(output),
            ):
                exit_code = chat.main()
            # THEN
            self.assertEqual(exit_code, 0)
            create.assert_not_called()
            self.assertEqual(output.getvalue().count('Informe uma pergunta'), 2)
            self.assertIn('Chat encerrado.', output.getvalue())

    def test_ctrl_c_na_entrada_ou_durante_resposta_encerra_sem_traceback(self):
        """Deve tratar interrupção do teclado em ambos os momentos da sessão."""
        # GIVEN
        for during_answer in [False, True]:
            service = Mock()
            service.answer.side_effect = KeyboardInterrupt()
            output = StringIO()
            # WHEN
            with (
                self.subTest(during_answer=during_answer),
                patch('chat.create_answer_service', return_value=service),
                patch('builtins.input', side_effect=['Pergunta'] if during_answer else KeyboardInterrupt()),
                redirect_stdout(output),
            ):
                exit_code = chat.main()
            # THEN
            self.assertEqual(exit_code, 0)
            self.assertIn('Chat encerrado.', output.getvalue())
            self.assertNotIn('RESPOSTA:', output.getvalue())

    def test_falha_tecnica_encerra_sem_retry_ou_expor_detalhes(self):
        """Deve distinguir erro de recusa e orientar banco/API/configuração."""
        # GIVEN
        cases = [
            (SQLAlchemyError('password=local-secret'), 'PostgreSQL'),
            (APIConnectionError(request=Mock()), 'OpenAI'),
            (AnswerError('Confira OPENAI_CHAT_MODEL.'), 'OPENAI_CHAT_MODEL'),
            (RuntimeError('local-secret'), 'Falha inesperada'),
        ]
        for error, expected_message in cases:
            service = Mock()
            service.answer.side_effect = error
            output, diagnostics = StringIO(), StringIO()
            # WHEN
            with (
                self.subTest(error=type(error).__name__),
                patch('chat.create_answer_service', return_value=service),
                patch('builtins.input', side_effect=['Pergunta', 'Não repetir']),
                redirect_stdout(output), redirect_stderr(diagnostics),
            ):
                exit_code = chat.main()
            # THEN
            self.assertEqual(exit_code, 1)
            service.answer.assert_called_once_with('Pergunta')
            self.assertIn(expected_message, diagnostics.getvalue())
            self.assertNotIn('local-secret', diagnostics.getvalue())
            self.assertNotIn('Traceback', diagnostics.getvalue())
            self.assertNotIn(REFUSAL, output.getvalue())
            self.assertNotIn('RESPOSTA:', output.getvalue())

    def test_configuracao_invalida_nao_anuncia_resposta(self):
        """Deve mostrar falha de inicialização sem simular resposta do PDF."""
        # GIVEN
        output, diagnostics = StringIO(), StringIO()
        # WHEN
        with (
            patch('builtins.input', return_value='Pergunta'),
            patch('chat.create_answer_service', side_effect=AnswerError('Confira OPENAI_API_KEY.')),
            redirect_stdout(output), redirect_stderr(diagnostics),
        ):
            exit_code = chat.main()
        # THEN
        self.assertEqual(exit_code, 1)
        self.assertIn('OPENAI_API_KEY', diagnostics.getvalue())
        self.assertNotIn('RESPOSTA:', output.getvalue())

    def test_varias_perguntas_recebem_respostas_sem_reiniciar(self):
        """Deve processar duas perguntas na mesma sessão e sair sem nova chamada."""
        # GIVEN
        service = Mock()
        service.answer.side_effect = ['R$ 10 milhões.', 'Resposta negativa.']
        output = StringIO()
        # WHEN
        with (
            patch('chat.create_answer_service', return_value=service) as create,
            patch('builtins.input', side_effect=['Qual receita?', 'Clientes?', 'sair']),
            redirect_stdout(output),
        ):
            exit_code = chat.main()
        # THEN
        self.assertEqual(exit_code, 0)
        self.assertEqual(service.answer.call_args_list, [call('Qual receita?'), call('Clientes?')])
        create.assert_called_once()
        self.assertIn('RESPOSTA: R$ 10 milhões.', output.getvalue())
        self.assertIn('RESPOSTA: Resposta negativa.', output.getvalue())


class ChatSubprocessTest(unittest.TestCase):
    def test_script_original_aceita_branco_saida_e_eof_sem_api(self):
        """Deve funcionar como python src/chat.py sem configuração de integrações."""
        # GIVEN
        for inputs in ['\n   \n SAIR \n', '']:
            # WHEN
            with self.subTest(inputs=inputs):
                result = subprocess.run(
                    [sys.executable, 'src/chat.py'], input=inputs,
                    text=True, capture_output=True, cwd=ROOT, timeout=20,
                )
            # THEN
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, '')
            self.assertIn('PERGUNTA:', result.stdout)
            self.assertIn('Chat encerrado.', result.stdout)
            self.assertNotIn('RESPOSTA:', result.stdout)

    def test_subprocesso_mostra_duas_respostas_e_oculta_falha_posterior(self):
        """Deve usar stdin/stdout reais com serviço controlado e erro seguro."""
        # GIVEN: provedor substituído no processo; nenhuma API ou banco.
        script = textwrap.dedent('''
            import sys
            from unittest.mock import Mock
            from sqlalchemy.exc import SQLAlchemyError
            sys.path.insert(0, 'src')
            import chat
            service = Mock()
            service.answer.side_effect = ['Primeira resposta.', 'Segunda resposta.', SQLAlchemyError('local-secret')]
            chat.create_answer_service = lambda: service
            raise SystemExit(chat.main())
        ''')
        # WHEN
        result = subprocess.run(
            [sys.executable, '-c', script], input='\nPrimeira?\nSegunda?\nTerceira?\n',
            text=True, capture_output=True, cwd=ROOT, timeout=20,
        )
        # THEN
        self.assertEqual(result.returncode, 1)
        self.assertIn('RESPOSTA: Primeira resposta.', result.stdout)
        self.assertIn('RESPOSTA: Segunda resposta.', result.stdout)
        self.assertEqual(result.stdout.count('RESPOSTA:'), 2)
        self.assertIn('PostgreSQL', result.stderr)
        self.assertNotIn('local-secret', result.stderr)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
