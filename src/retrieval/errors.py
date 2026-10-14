"""Erros seguros e distinguíveis do fluxo de recuperação."""


class SearchError(Exception):
    """Erro esperado que pode ser apresentado pelo CLI sem detalhes sensíveis."""


class InvalidQuestionError(SearchError):
    """A pergunta não contém texto pesquisável."""


class CollectionNotReadyError(SearchError):
    """A collection configurada ainda não possui chunks consultáveis."""


class IncompatibleCollectionError(SearchError):
    """A collection foi criada com outro modelo ou dimensão."""


class InvalidSearchConfigurationError(SearchError):
    """Falta uma configuração necessária para consultar banco ou provedor."""
