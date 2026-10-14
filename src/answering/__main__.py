"""Responde uma pergunta por execução; o loop de chat pertence à próxima história."""
import argparse
import sys

from dotenv import load_dotenv

from ingestion.config import ROOT, Settings
from retrieval.service import SemanticSearchService

from .cli import describe_error
from .model import ChatModel
from .service import AnswerService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Pergunta sobre o PDF.")
    arguments = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    try:
        settings = Settings.from_env()
        model = ChatModel.from_env(settings.api_key)
        service = AnswerService(SemanticSearchService(settings), model)
        print(service.answer(" ".join(arguments.question)))
        return 0
    except Exception as error:
        print(f"Erro: {describe_error(error)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
