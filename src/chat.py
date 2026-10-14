"""Chat sobre o PDF: cada pergunta recupera seu próprio contexto."""
import sys

from dotenv import load_dotenv
from answering.cli import describe_error
from answering.model import ChatModel
from answering.service import AnswerService
from ingestion.config import ROOT, Settings
from retrieval.service import SemanticSearchService


def main() -> int:
    try:
        run_session()
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nChat encerrado.")
        return 0
    except Exception as error:
        print(f"Erro: {describe_error(error)}", file=sys.stderr)
    return 1


def run_session() -> None:
    print("Faça sua pergunta sobre o PDF. Digite sair para encerrar.")
    service = None
    while True:
        question = input("PERGUNTA: ").strip()
        if question.casefold() == "sair":
            print("Chat encerrado.")
            return
        if not question:
            print("Informe uma pergunta com texto.")
            continue
        # Inicialização adiada: branco e saída não exigem banco ou chave de API.
        if service is None:
            service = create_answer_service()
        answer = service.answer(question)
        print(f"RESPOSTA: {answer}\n")


def create_answer_service() -> AnswerService:
    load_dotenv(ROOT / ".env", override=False)
    settings = Settings.from_env()
    model = ChatModel.from_env(settings.api_key)
    return AnswerService(SemanticSearchService(settings), model)


if __name__ == "__main__":
    raise SystemExit(main())
