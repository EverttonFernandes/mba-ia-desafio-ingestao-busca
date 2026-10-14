"""Cliente de chat com orçamento conservador e validação da resposta."""
import math
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .prompt import SYSTEM_RULES, build_prompt

DEFAULT_CHAT_MODEL = "gpt-4.1-mini-2025-04-14"
MAX_OUTPUT_TOKENS = 160
MAX_CHAT_COST_USD = 0.01
# Preços standard conferidos em 11/10/2026, por milhão de tokens.
CHAT_PRICES = {DEFAULT_CHAT_MODEL: (0.40, 1.60)}


class AnswerError(Exception):
    """Erro técnico seguro; nunca representa ausência de informação no PDF."""


class ChatModel:
    """Mantém a configuração da LLM; cria o cliente só após conferir o orçamento."""

    def __init__(self, api_key: str, model: str):
        if not api_key or model not in CHAT_PRICES:
            raise AnswerError(
                "Confira OPENAI_API_KEY e OPENAI_CHAT_MODEL. "
                f"O modelo com preço revisado é {DEFAULT_CHAT_MODEL}."
            )
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_env(cls, api_key: str) -> "ChatModel":
        return cls(api_key, os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip())

    def answer(self, context: str, question: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_RULES),
            HumanMessage(content=build_prompt(context, question)),
        ]
        self._validate_budget(messages)
        client = ChatOpenAI(
            api_key=self.api_key, model=self.model,
            temperature=0, max_tokens=MAX_OUTPUT_TOKENS,
            max_retries=0, timeout=60,
        )
        response = client.invoke(messages)
        return self._read_answer(response)

    def _validate_budget(self, messages: list) -> None:
        # Bytes UTF-8 superestimam tokens BPE; reserva adicional cobre o envelope.
        input_bound = sum(len(message.content.encode("utf-8")) for message in messages) + 256
        input_price, output_price = CHAT_PRICES[self.model]
        estimated_cost = (
            input_bound * input_price + MAX_OUTPUT_TOKENS * output_price
        ) / 1_000_000
        if not math.isfinite(estimated_cost) or estimated_cost > MAX_CHAT_COST_USD:
            raise AnswerError(
                "Estimativa da chamada de chat excede US$ 0,01. "
                "Revise o tamanho da pergunta e do contexto antes de continuar."
            )

    def _read_answer(self, response) -> str:
        if response.response_metadata.get("finish_reason") != "stop":
            raise AnswerError(
                "A LLM não concluiu uma resposta textual. "
                "Confira o limite de saída e o modelo antes de repetir."
            )
        if not isinstance(response.content, str) or not response.content.strip():
            raise AnswerError("A LLM retornou resposta vazia ou inesperada.")
        return response.content.strip()
