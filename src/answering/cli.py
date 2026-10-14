"""Diagnósticos seguros compartilhados pelas duas entradas de resposta."""
from openai import APIError
from sqlalchemy.exc import SQLAlchemyError

from ingestion.errors import IngestError
from retrieval.errors import SearchError

from .model import AnswerError


def describe_error(error: Exception) -> str:
    """Expõe erros controlados e oculta detalhes de serviços externos."""
    if isinstance(error, (AnswerError, SearchError, IngestError)):
        return str(error)
    if isinstance(error, APIError):
        return "Falha na OpenAI. Confira chave, saldo, modelo e conexão antes de repetir."
    if isinstance(error, SQLAlchemyError):
        return "Falha no PostgreSQL. Confira Docker e as variáveis POSTGRES_* do .env."
    return "Falha inesperada ao responder. Detalhes externos foram ocultados."
