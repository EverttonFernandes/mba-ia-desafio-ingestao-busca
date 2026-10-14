"""Ingestao de PDF: configuracao -> chunks -> embeddings -> PostgreSQL."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError
from sqlalchemy.exc import SQLAlchemyError

from ingestion.config import ROOT, Settings
from ingestion.errors import IngestError
from ingestion.pdf import CHUNK_SIZE, CHUNK_OVERLAP, prepare_pdf
from ingestion.embeddings import estimate_cost
from ingestion.service import IngestionService


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    load_dotenv(ROOT / ".env", override=False)
    try:
        run(args)
        return 0
    except IngestError as exc:
        message = str(exc)
    except AuthenticationError:
        message = "Autenticacao OpenAI falhou. Confira OPENAI_API_KEY no .env."
    except RateLimitError:
        message = (
            "OpenAI recusou por cota ou limite. Confira Usage/Billing e aguarde "
            "se necessario; nao houve repeticao automatica."
        )
    except APIConnectionError:
        message = (
            "Falha de conexao com OpenAI. Confira a rede; "
            "a API pode ter processado a solicitacao."
        )
    except APIError:
        message = "OpenAI recusou a solicitacao. Confira modelo, dimensao e acesso da conta."
    except SQLAlchemyError:
        message = (
            "Falha no PostgreSQL. Confira Docker, variaveis POSTGRES_*, permissoes "
            "e schema no pgAdmin. Persistencia nao confirmada; nao apague volumes."
        )
    except Exception:
        # Nunca imprimir excecoes de SDK/banco que possam conter SQL, texto ou chaves.
        message = (
            "Falha inesperada na ingestao. Confira arquivo, dependencias "
            "e configuracao; persistencia nao confirmada."
        )
    print(f"Erro: {message}", file=sys.stderr)
    return 1


def run(args: argparse.Namespace) -> None:
    """Executa o fluxo da CLI; main() transforma erros em mensagens seguras."""
    settings = Settings.from_env()
    chunks, pdf_hash = prepare_pdf(args.pdf)
    print(f"PDF preparado: {len(chunks)} chunks; tamanho/overlap {CHUNK_SIZE}/{CHUNK_OVERLAP}.")
    if args.dry_run:
        estimated_cost = estimate_cost(chunks, settings.embedding_model)
        print(f"Estimativa conservadora de embeddings: US$ {estimated_cost:.6f}.")
        print("Nenhuma chamada de API ou conexao ao banco foi realizada.")
        return

    created = IngestionService(settings).ingest(chunks, pdf_hash, args.max_cost_usd)
    if created:
        print("Ingestao concluida: textos, metadata e vetores confirmados no PostgreSQL.")
    else:
        print("PDF ja ingerido: dados conferidos; nenhuma chamada de API realizada.")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "document.pdf")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida PDF e estima custo sem banco ou API.",
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=0.01,
        help="Limite da estimativa por execucao (padrao: 0.01 USD).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
