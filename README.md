# Ingestão e busca semântica com LangChain e PostgreSQL

Desafio do MBA: ingerir um PDF no PostgreSQL com pgVector e responder perguntas
no terminal usando somente o conteúdo recuperado. Baseado no
[repositório oficial](https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca).

O PDF é dividido com `RecursiveCharacterTextSplitter` em chunks de até 1.000
caracteres, com overlap configurado em 150. Os embeddings são armazenados via
`PGVector`. Cada pergunta é vetorizada e recupera até dez chunks com
`similarity_search_with_score(..., k=10)` para compor o prompt da LLM.

## Preparar o ambiente

Requisitos: Python 3.10+ (validado com 3.14.4), Docker e Docker Compose ativos.
Na raiz do projeto:

```sh
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Se a criação da venv falhar por ausência de `ensurepip`, instale o pacote
`python3-venv` correspondente à sua versão do Python e repita o comando.

Edite o `.env` e preencha:

- `POSTGRES_PASSWORD`: senha local do PostgreSQL.
- `PGADMIN_DEFAULT_EMAIL` e `PGADMIN_DEFAULT_PASSWORD`: login local do pgAdmin.
- `OPENAI_API_KEY`: chave da [plataforma OpenAI](https://platform.openai.com/api-keys),
  com acesso à API e saldo disponível.

O comando de cópia preserva um `.env` existente. Não versione suas credenciais.
Os demais valores padrão estão em [.env.example](.env.example): banco `rag`,
usuário `postgres`, host `127.0.0.1` e porta `5432` para a aplicação Python.

| Uso | Modelo configurado |
| --- | --- |
| Embeddings | `text-embedding-3-small`, com 1.536 dimensões |
| Respostas | `gpt-4.1-mini-2025-04-14` |

A primeira ingestão e as perguntas à OpenAI geram custo. Os testes unitários
e as integrações com embeddings sintéticos não fazem chamadas pagas.

## Executar o desafio

### 1. Subir o banco

```sh
docker compose up -d
docker compose ps -a
```

Aguarde `postgres` ficar saudável e `bootstrap_vector_ext` terminar com código 0.
Esse serviço habilita a extensão `vector` no banco.

### 2. Ingerir o PDF

```sh
python src/ingest.py
```

O `document.pdf` de exemplo produz 67 chunks, armazenados com texto, metadata e
vetores. Repetir o comando com o mesmo PDF e configuração confere os dados sem
gerar novos embeddings. Para apenas ler e dividir o PDF, sem banco ou API:

```sh
python src/ingest.py --dry-run
```

Mantenha o modelo e as dimensões de embeddings usados na ingestão. Mudá-los exige
uma collection e uma tabela compatíveis e nova ingestão; não há exclusão automática.

### 3. Conversar no terminal

```sh
python src/chat.py
```

Exemplo com o PDF fornecido:

```text
PERGUNTA: Qual o faturamento da SuperTechIABrazil?
RESPOSTA: O faturamento da SuperTechIABrazil é de R$ 10.000.000,00.

PERGUNTA: Quantos clientes a SuperTechIABrazil teve em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: sair
Chat encerrado.
```

As respostas positivas podem variar na redação. Perguntas fora do contexto,
como “Qual é a capital da França?” ou pedidos de opinião, devem retornar a
frase literal de recusa mostrada acima. O prompt exigido pelo desafio está em
[src/answering/prompt.py](src/answering/prompt.py).

Cada pergunta é independente, sem memória da conversa. Digite `sair` ou pressione
Ctrl+C para encerrar. Para inspecionar somente os chunks recuperados e seus scores:

```sh
python src/search.py "Qual o faturamento da SuperTechIABrazil?"
```

## Testes

### Unitários: sem Docker ou chave de API

Com as dependências instaladas e a venv ativa:

```sh
python -m unittest discover -s tests -p "test_*.py" -v
```

Resultado esperado: **56 testes, `OK`**. A suíte usa dependências controladas
e pode ser executada antes de criar o `.env`.

| Arquivo | O que verifica |
| --- | --- |
| [test_ingest.py](tests/test_ingest.py) | PDF, chunking, IDs, idempotência, lock, orçamento e persistência. |
| [test_search.py](tests/test_search.py) | k=10, compatibilidade, documentos, scores e contexto. |
| [test_answering.py](tests/test_answering.py) | Prompt literal, recusa, configuração do modelo e erros. |
| [test_chat.py](tests/test_chat.py) | Perguntas sucessivas, saída, entrada vazia e tratamento de falhas. |

O aviso `EOF marker not found` é esperado no teste de PDF inválido; confira o
`OK` final. O prompt de referência está em [tests/fixtures/answering_prompt.txt](tests/fixtures/answering_prompt.txt).
Os testes do PDF usam o arquivo de exemplo; substituí-lo exige revisar as expectativas.

### Integração: PostgreSQL real, sem OpenAI

Com o banco saudável e o `.env` configurado, mantendo os padrões de embeddings:

```sh
python tests/integration_ingest.py
python tests/integration_search.py
```

Os scripts usam vetores sintéticos para validar gravação, idempotência,
incompatibilidades e busca. Cada execução cria e remove somente seu próprio
banco temporário. O usuário precisa de permissão para criar bancos e habilitar
`vector`; o usuário padrão do Compose possui essas permissões.

Após a ingestão do PDF, confira os dados sem chamar a OpenAI:

```sh
python tests/verify_persistence.py
```

Essa verificação compara IDs, textos, metadata e dimensões e confirma a
idempotência. Para o PDF fornecido, espera-se 67 chunks com 1.536 dimensões.
Para validar o provedor real, execute o chat com as perguntas do exemplo acima.

## Inspecionar pelo pgAdmin

Abra [http://localhost:5050](http://localhost:5050) e entre com
`PGADMIN_DEFAULT_EMAIL` e `PGADMIN_DEFAULT_PASSWORD` do `.env`.
Em **Servers → Register → Server**, escolha um nome e preencha a aba **Connection**:

| Campo | Valor |
| --- | --- |
| Host | `postgres` — nome do serviço na rede Docker, não `localhost` |
| Porta | `5432` |
| Maintenance database | `POSTGRES_DB` do `.env`; padrão `rag` |
| Username | `POSTGRES_USER` do `.env`; padrão `postgres` |
| Password | `POSTGRES_PASSWORD` do `.env` |

PostgreSQL é o banco relacional; pgVector é uma extensão dentro dele que adiciona
o tipo `vector` e operações de similaridade. Não são dois bancos separados.
Em **Databases → rag → Extensions**, procure `vector`. No **Query Tool**, execute:

```sql
SELECT * FROM pg_extension;

SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';

SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public';

SELECT name, cmetadata FROM public.langchain_pg_collection;

SELECT document, cmetadata, vector_dims(embedding) AS dimensoes, embedding
FROM public.langchain_pg_embedding
LIMIT 5;
```

As tabelas do LangChain aparecem após a ingestão. Em **Schemas → public → Tables**,
use **View/Edit Data → First 100 Rows** para visualizar collections, chunks,
metadata e embeddings. Os dados persistem em volumes; `docker compose down`
para os serviços sem apagá-los. Alterar a senha no `.env` não troca a senha de
um PostgreSQL já inicializado.
