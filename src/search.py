"""Busca os dez chunks do PDF mais similares a uma pergunta."""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError
from sqlalchemy.exc import SQLAlchemyError

from ingestion.config import ROOT, Settings
from retrieval.errors import SearchError
from retrieval.service import SemanticSearchService, concatenate_context


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    load_dotenv(ROOT / ".env", override=False)
    try:
        run(arguments)
        return 0
    except SearchError as error:
        message = str(error)
    except AuthenticationError:
        message = "Autenticação OpenAI falhou. Confira OPENAI_API_KEY no .env."
    except RateLimitError:
        message = "OpenAI recusou a busca por cota ou limite. Confira Usage/Billing."
    except APIConnectionError:
        message = "Falha de conexão com OpenAI. Confira a rede antes de repetir."
    except APIError:
        message = "OpenAI recusou a vetorização da pergunta. Confira o modelo."
    except SQLAlchemyError:
        message = (
            "Falha no PostgreSQL. Confira Docker, variáveis POSTGRES_* e a "
            "collection no pgAdmin."
        )
    except Exception:
        message = (
            "Falha inesperada na busca. Confira dependências e configuração; "
            "detalhes externos foram ocultados."
        )
    print(f"Erro: {message}", file=sys.stderr)
    return 1


def run(arguments: argparse.Namespace) -> None:
    settings = Settings.from_env()
    question = " ".join(arguments.question)
    results = SemanticSearchService(settings).search(question)
    print(f"Resultados recuperados: {len(results)}")
    for position, (document, score) in enumerate(results, start=1):
        page = document.metadata.get("page", "desconhecida")
        print(f"\n[{position}] página={page} score={score}")
        print(document.page_content)
    print("\nCONTEXTO RECUPERADO:")
    print(concatenate_context(results))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question",
        nargs="+",
        help="Pergunta usada para recuperar contexto do PDF.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
