"""Coordena a busca do contexto e a geração da resposta."""
from retrieval.service import SemanticSearchService, concatenate_context

from .model import ChatModel
from .prompt import REFUSAL


class AnswerService:
    """Responde uma pergunta usando os documentos da busca configurada."""

    def __init__(self, search: SemanticSearchService, model: ChatModel):
        self.search = search
        self.model = model

    def answer(self, question: str) -> str:
        results = self.search.search(question)
        context = concatenate_context(results)
        if not context.strip():
            return REFUSAL
        return self.model.answer(context, question)
