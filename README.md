# Ingestão e busca semântica com LangChain e PostgreSQL

Desafio do MBA: ingerir um PDF em PostgreSQL/pgVector e responder via terminal somente
com base no documento. Fonte: [repositório oficial](https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca).

## Estado atual

Ambiente local, ingestão do PDF, recuperação semântica, resposta com LLM e chat
contínuo no terminal implementados.
O PDF foi selecionado com permissão de uso e redistribuição confirmada pelo usuário.
OpenAI é o provedor escolhido; embeddings e busca foram validados com 1.536 dimensões.
Nenhuma chave de modelo é necessária para subir o banco e o pgAdmin.

O GitHub informa que o repositório atual não é um fork do exemplo. Regularizar a
origem conforme o enunciado é uma pendência manual que bloqueia a entrega final.

## Pré-requisitos e Python

Python 3.10+ (validação local com 3.14.4), Docker e Docker Compose com daemon ativo.
Na raiz do projeto:

```sh
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Se a distribuição não incluir `ensurepip`, instale o pacote de venv correspondente
(por exemplo `python3.14-venv` no Debian/Ubuntu). Alternativamente, o método
[oficial do pip](https://pip.pypa.io/en/stable/installation/) funciona dentro da venv:

```sh
python3 -m venv --without-pip venv
curl -fsSLo /tmp/rag-get-pip.py https://bootstrap.pypa.io/get-pip.py
venv/bin/python /tmp/rag-get-pip.py
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

As dependências diretas estão fixadas em `requirements.txt`. A integração OpenAI está incluída com `langchain-openai`.

## Configurar e subir o banco

Na primeira execução, copie o template (preserve seu `.env` se já existir):

```sh
cp -n .env.example .env
```

Preencha `POSTGRES_PASSWORD` no `.env` com uma senha local. Usuário, banco e porta
padrão são `postgres`, `rag` e `5432`. `POSTGRES_HOST=127.0.0.1` é o endereço para o
Python executado na máquina; o bootstrap usa o hostname interno `postgres`.
Preencha também `PGADMIN_DEFAULT_EMAIL` com um e-mail válido para o login local
(por exemplo `admin@example.com`), `PGADMIN_DEFAULT_PASSWORD` com uma senha própria
e `PGADMIN_PORT=5050`. Esse usuário é criado apenas no pgAdmin local; não exige conta
externa. Se seu `.env` já existia, acrescente essas três variáveis do template.
O Compose rejeita credenciais obrigatórias ausentes ou vazias. Não versione `.env`.

```sh
docker compose config --quiet
docker compose up -d
docker compose ps -a
docker compose logs bootstrap_vector_ext
```

O serviço `postgres` deve ficar saudável e `bootstrap_vector_ext` deve terminar com
código 0. O bootstrap executa `CREATE EXTENSION IF NOT EXISTS vector` e pode ser
repetido. Aguarde sua conclusão antes de executar a aplicação.

```sh
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT extname, extversion FROM pg_extension WHERE extname = '\''vector'\'';"'
```

Para verificar a conexão TCP do Python e uma operação vetorial, com a venv ativa:

```sh
python - <<'PY'
from dotenv import dotenv_values
import psycopg

config = dotenv_values('.env')
with psycopg.connect(
    host=config['POSTGRES_HOST'], port=config['POSTGRES_PORT'],
    user=config['POSTGRES_USER'], password=config['POSTGRES_PASSWORD'],
    dbname=config['POSTGRES_DB'], connect_timeout=5,
) as conn:
    print(conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone())
    print(conn.execute("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector").fetchone())
PY
```

Espera-se a versão da extensão e distância `1.0`.

## Adaptações do Compose oficial

Mantidos PostgreSQL 17 com pgVector, volume persistente e serviço de bootstrap.
Credenciais saíram do YAML e são compartilhadas pelos serviços via `.env`;
o healthcheck acompanha usuário/banco configurados. A porta fica vinculada a
`127.0.0.1` e pode mudar com `POSTGRES_PORT` se 5432 estiver ocupada. O nome fixo de
container foi removido para evitar colisões entre projetos Compose.

O volume `postgres_data` preserva os dados. Alterar usuário, senha ou banco no `.env`
não altera as credenciais de um volume já inicializado. Para parar sem apagar dados:

```sh
docker compose down
```

## pgAdmin: administrar pelo navegador

O serviço `pgadmin` usa a imagem oficial `dpage/pgadmin4:9.18`, com versão fixada,
conforme a [documentação de containers do pgAdmin](https://www.pgadmin.org/docs/pgadmin4/9.18/container_deployment.html).
Ele compartilha a rede padrão do Compose com `postgres` e tem volume próprio
`pgadmin_data`, montado em `/var/lib/pgadmin`, para manter usuários, conexões e configurações.

```sh
docker compose up -d
docker compose ps -a
```

Aguarde a inicialização e abra [http://localhost:5050](http://localhost:5050).
Se alterar `PGADMIN_PORT`, use essa porta no navegador. A interface está exposta
somente na máquina local. Entre com o e-mail de `PGADMIN_DEFAULT_EMAIL` e a senha
de `PGADMIN_DEFAULT_PASSWORD` do seu `.env`. As senhas locais não são publicadas
neste README; consulte seu `.env` no editor. O login do pgAdmin e o usuário do banco
são credenciais independentes.

### Registrar o PostgreSQL

No painel esquerdo, clique com o botão direito em **Servers → Register → Server**.
Na aba **General**, escolha um nome descritivo, como **RAG local**. Na aba
**Connection**, preencha:

| Campo | Valor |
| --- | --- |
| Host name/address | `postgres` (nome do serviço no Compose) |
| Port | `5432` (porta interna do PostgreSQL) |
| Maintenance database | Valor de `POSTGRES_DB` no `.env`; padrão `rag` |
| Username | Valor de `POSTGRES_USER` no `.env`; padrão `postgres` |
| Password | Valor de `POSTGRES_PASSWORD` no `.env` |

Clique em **Save**. Dentro do pgAdmin, **não use `localhost` como host do banco**:
isso apontaria para o próprio container do pgAdmin. `POSTGRES_HOST=127.0.0.1` serve
para o Python na máquina. Mesmo que `POSTGRES_PORT` mude no host, a conexão entre
os containers continua usando **postgres:5432**. A porta **5050** é da interface web.

O volume do pgAdmin é independente de `postgres_data`. Recriar o container preserva
as configurações. `PGADMIN_DEFAULT_EMAIL` e `PGADMIN_DEFAULT_PASSWORD` inicializam
o primeiro usuário; mudar esses valores no `.env` não redefine um usuário já
salvo no volume. Para alterar sua senha depois, use a opção de troca de senha
na interface do pgAdmin. `docker compose down` preserva ambos os volumes; não use
`down -v` se quiser manter dados e configurações.

### PostgreSQL e pgVector: um único banco

**PostgreSQL** é o banco de dados relacional. **pgVector** é uma extensão instalada
dentro dele que adiciona o tipo `vector` e operações de similaridade. Portanto,
não existem dois bancos diferentes: os embeddings serão colunas `vector` em
tabelas PostgreSQL, consultáveis pelo próprio pgAdmin.

Para confirmar visualmente a extensão, expanda **Servers → RAG local → Databases →
rag → Extensions** (substitua `rag` se configurou outro banco) e procure **vector**.
Use **Refresh** se necessário. Selecione o banco e abra **Tools → Query Tool**;
execute as consultas com o botão de execução (F5):

```sql
-- Listar extensões instaladas
SELECT * FROM pg_extension;

-- Confirmar pgVector: deve retornar uma linha com nome e versão
SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';

-- Listar tabelas do schema public
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public';

-- Descobrir colunas e tipos sem pressupor nomes de tabelas
SELECT table_name, column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;

-- Experimentar distância entre vetores; resultado esperado: 1
SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector AS distancia;
```

### Visualizar os dados da ingestão

Depois de executar a ingestão, atualize **Databases → rag → Schemas → public →
Tables**. O `PGVector` de `langchain-postgres==0.0.18` cria
`langchain_pg_collection` (collections e configuração) e `langchain_pg_embedding`
(chunks e vetores). Os nomes foram conferidos no código da versão instalada.

Em cada tabela identificada, abra **View/Edit Data → First 100 Rows** para visualizar
collections, registros/chunks, conteúdo textual, metadata e embeddings. Expanda
**Columns** para conferir os tipos: embeddings aparecem como `vector`, com sua
sequência de números na grade. O pgAdmin permite inspecionar esses valores; ele
não os transforma automaticamente em um gráfico de similaridade.

No **Query Tool** do banco `rag`, experimente:

```sql
-- Collections e modelo/dimensão usados na ingestão
SELECT name, cmetadata FROM public.langchain_pg_collection;

-- Quantidade de chunks por collection
SELECT c.name, count(e.id) AS chunks
FROM public.langchain_pg_collection c
LEFT JOIN public.langchain_pg_embedding e ON e.collection_id = c.uuid
GROUP BY c.name;

-- Texto, metadata e embedding; page é zero-based (some 1 para ler no PDF)
SELECT e.document, e.cmetadata, vector_dims(e.embedding) AS dimensoes, e.embedding
FROM public.langchain_pg_embedding e
JOIN public.langchain_pg_collection c ON c.uuid = e.collection_id
WHERE c.name = 'document_pdf'
ORDER BY (e.cmetadata->>'chunk_index')::integer
LIMIT 10;
```

Troque `document_pdf` na consulta se configurar outra `PGVECTOR_COLLECTION`.

## Configuração OpenAI (RAGT-002)

Preencha `OPENAI_API_KEY` apenas no `.env` local. Crie sua chave na
[plataforma OpenAI](https://platform.openai.com/api-keys); não cole chaves no chat,
README ou Git. Configure faturamento e limites na sua conta antes de usar a API.

```dotenv
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
OPENAI_CHAT_MODEL=gpt-4.1-mini-2025-04-14
```

O modelo de embeddings transforma texto em vetores de 1.536 números; o modelo de
chat produzirá respostas usando o contexto recuperado. A ingestão usa somente
embeddings; ela não chama o modelo de chat.

Preços standard consultados em 21/09/2026, em dólares por milhão de tokens:
embeddings US$ 0,02; chat US$ 0,40 de entrada e US$ 1,60 de saída. Limites e acesso
dependem da conta. Fontes: [embeddings](https://developers.openai.com/api/docs/models/text-embedding-3-small),
[dimensões](https://developers.openai.com/api/docs/guides/embeddings) e
[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Smoke real validado em 22/09/2026: um embedding de frase sintética retornou 1.536
dimensões e o chat respondeu “OK”, usando 16 tokens de entrada e 1 de saída.
O PDF não foi enviado nesse teste. O acesso depende da chave e do saldo de cada conta.
Para economizar, testes locais devem ser priorizados e chamadas reais mantidas
pequenas, com saída limitada e sem repetições automáticas. Saldo comprado e limite
mensal são controles diferentes; manter auto-reload desligado evita recargas automáticas.

Se trocar o modelo de embeddings ou suas dimensões, use uma nova collection e
refaça a ingestão. Mesmo dimensões iguais não tornam modelos intercambiáveis.
Se a tabela tiver uma coluna com dimensão fixa incompatível, será necessária uma
tabela/schema compatível novo; mudar só a collection não altera a coluna.
Nunca apague o volume automaticamente para corrigir incompatibilidades.

O `document.pdf` selecionado contém 34 páginas textuais com empresas, faturamento
e ano de fundação. Uso e redistribuição foram confirmados pelo usuário. Por exemplo,
a página 3 informa faturamento de R$ 10.000.000,00 e fundação em 2025 para a
SuperTechIABrazil. O ano de fundação não deve ser interpretado como ano do faturamento.
O arquivo foi fornecido pelo responsável pelo projeto; sua inclusão se baseia
nessa autorização declarada. Não foi atribuída uma licença pública específica ao
PDF. Antes de substituí-lo por outro documento, confirme também a permissão para
publicá-lo e enviar seu texto ao provedor.

## Ingerir o PDF (RAG-001)

Com a venv ativa, execute na raiz do projeto:

```sh
# Apenas ler e dividir o PDF: não acessa banco nem API
python src/ingest.py --dry-run

# Subir PostgreSQL, bootstrap e pgAdmin; aguarde postgres saudável/bootstrap exit 0
docker compose up -d
docker compose ps -a

# Envia o texto dos chunks à OpenAI e confirma a gravação no PostgreSQL
python src/ingest.py
```

O script lê `document.pdf` usando `PyPDFLoader` e divide cada página com
`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`. Separadores
podem produzir chunks menores e reduzir o overlap efetivo; páginas não são
misturadas. O PDF de referência produz 67 chunks. Cada registro conserva `source`,
`page` (a primeira é 0), `start_index`, `chunk_index` e `pdf_sha256`.

`PGVECTOR_COLLECTION=document_pdf` é o padrão mesmo em um `.env` antigo sem essa
variável. O `.env` é localizado pela raiz do projeto; variáveis já exportadas no
terminal têm prioridade. Para outro PDF textual, use `--pdf caminho/arquivo.pdf`.
PDFs digitalizados sem texto precisam de OCR externo, fora deste desafio.

Para economizar, a primeira execução usa apenas embeddings `text-embedding-3-small`,
com retries desativados. `--dry-run` estima conservadoramente o custo contando
bytes UTF-8 dos chunks, inclusive overlap, como limite superior de tokens. Para o
PDF atual, a estimativa é US$ 0,001020, com preço standard de US$ 0,02/milhão de
tokens consultado em 22/09/2026 na
[documentação oficial](https://developers.openai.com/api/docs/models/text-embedding-3-small).
É uma estimativa, não o valor faturado. O script bloqueia uma estimativa acima de
US$ 0,01 por execução; `--max-cost-usd VALOR` permite alterar esse limite após
revisar o volume. Modelos sem preço revisado no código são bloqueados antes da API.
Esse controle local não limita outros programas usando a mesma conta.

### Repetir a ingestão e tratar falhas

Reexecutar `python src/ingest.py` com o mesmo PDF e configuração confere os dados
existentes e termina sem gerar embeddings novamente. IDs determinísticos e a
verificação de textos, metadata e dimensões evitam duplicação silenciosa. Um lock
no PostgreSQL impede duas execuções simultâneas deste script no mesmo banco.

A collection registra hash do PDF, hash dos chunks, modelo, dimensão e parâmetros
de divisão. Se algum deles mudar, o script recusa reutilizar essa collection:
escolha outra `PGVECTOR_COLLECTION` e refaça a ingestão conscientemente. Uma coluna
`vector(1536)` continua incompatível com outra dimensão mesmo em nova collection;
nesse caso, prepare um banco/schema compatível com orientação explícita. Este
script usa o schema `public` e nunca apaga collections, tabelas ou volumes.

O lote de chunks é persistido em uma transação e relido antes de anunciar sucesso.
Uma falha na API pode deixar uma collection vazia; corrigir a causa e repetir
permite completá-la. Estado parcial ou inconsistente exige inspeção no pgAdmin.
Se a API já respondeu e a gravação falhar, uma nova tentativa pode gerar custo
novamente: não há cache local de embeddings. Após erro de conexão, confira os
dados antes de repetir; falhas encerram com código 1 e mensagem sem credenciais.

Para comprovar persistência, anote a contagem SQL acima, execute
`docker compose restart postgres`, aguarde o banco saudável e repita a consulta.
Os dados devem continuar presentes. Não use `docker compose down -v`.

## Buscar contexto por similaridade (RAG-002)

Depois de ingerir o PDF, passe uma pergunta ao script de busca:

```sh
python src/search.py "Qual o faturamento da Empresa SuperTechIABrazil?"
```

O script valida se a collection configurada existe, contém dados e foi criada com
o mesmo modelo e a mesma dimensão definidos no `.env`. Em seguida, gera somente o
embedding da pergunta e executa `similarity_search_with_score(..., k=10)` na
`PGVECTOR_COLLECTION` configurada. Ele mostra cada chunk com página e score e, ao
final, concatena apenas o texto dos chunks para formar o contexto.

O score exibido é o valor bruto devolvido pelo PGVector e **não é uma porcentagem
nem uma probabilidade**. Se a collection tiver menos de dez chunks, somente os
registros existentes serão retornados. Uma lista vazia significa que nenhum
contexto foi recuperado; collection ausente ou vazia, configuração incompatível e
falhas no PostgreSQL/OpenAI produzem erros distintos e encerram o comando com código
1. A busca nunca consulta outra collection para completar resultados.

Na validação funcional do PDF de referência, a pergunta do exemplo recuperou dez
chunks e encontrou o trecho `SuperTechIABrazil R$ 10.000.000,00 2025` na posição 2.
Essa execução usa uma chamada pequena à API de embeddings. Os testes unitários e a
integração sintética não acessam a OpenAI e devem ser preferidos durante o estudo.

## Responder com base no PDF (RAG-003)

Com o PostgreSQL ativo, PDF já ingerido e venv ativa, execute na raiz:

```sh
PYTHONPATH=src python -m answering "Qual o faturamento da SuperTechIABrazil?"
PYTHONPATH=src python -m answering "Quantos clientes temos em 2024?"
```

Validação real em 11/10/2026: faturamento e fundação da SuperTechIABrazil
respondidos com suporte no PDF; informação ausente, opinião e tentativas de
injeção pela pergunta e por texto sintético recuperado retornaram a recusa literal.

A execução recupera até dez chunks, preenche o prompt literal do desafio com os
textos e a pergunta e chama `ChatOpenAI`. O modelo é definido por
`OPENAI_CHAT_MODEL`; o padrão revisado é `gpt-4.1-mini-2025-04-14`. A resposta
para ausência de suporte explícito ou pedido de opinião é:

```text
Não tenho informações necessárias para responder sua pergunta.
```

O prompt do desafio é preservado integralmente na mensagem do usuário. Uma
mensagem de sistema adicional estabelece que instruções presentes na pergunta
ou nos chunks são dados não confiáveis e não podem substituir as regras.
Essa orientação reduz o risco, mas não garante matematicamente a ausência de
alucinações ou de sucesso de uma injeção de prompt. Confira respostas importantes
com o texto original do PDF; scores não comprovam que a informação esteja nele.

Cada pergunta normalmente usa uma chamada de embeddings e uma de chat. O chat
usa temperatura zero, saída limitada a 160 tokens e nenhuma repetição automática.
Uma estimativa conservadora inclui bytes UTF-8 da entrada, reserva de envelope
e o limite completo de saída; chamadas estimadas acima de US$ 0,01 são bloqueadas
antes de criar o cliente de chat. Preços standard revisados em 11/10/2026:
US$ 0,40/milhão de tokens de entrada e US$ 1,60/milhão de saída, conforme a
[documentação do modelo](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Outro modelo exige revisar seus preços no código. O limite é por chamada de chat,
não um teto acumulado da conta nem do custo de embeddings.

Contexto vazio retorna a recusa sem chamada de chat. Falhas de banco/API,
configuração inválida ou saída vazia/truncada são erros técnicos, com código de
saída 1; elas não são apresentadas como informação ausente no PDF. O texto dos
chunks recuperados é enviado à OpenAI para gerar a resposta.

## Conversar no terminal (RAG-004)

Na raiz, com a venv ativa e as configurações do `.env` preenchidas:

```sh
docker compose up -d
python src/ingest.py
python src/chat.py
```

Se o PDF já estiver completamente ingerido com a mesma configuração, a ingestão
confere os dados e não gera novos embeddings. Aguarde PostgreSQL saudável e o
bootstrap concluído antes de iniciar o chat.

Exemplo de sessão:

```text
Faça sua pergunta sobre o PDF. Digite sair para encerrar.
PERGUNTA: Qual o faturamento da SuperTechIABrazil?
RESPOSTA: O faturamento da SuperTechIABrazil é de R$ 10.000.000,00.

PERGUNTA: Quantos clientes a SuperTechIABrazil teve em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: sair
Chat encerrado.
```

Você pode fazer várias perguntas sem reiniciar o programa. Cada pergunta recupera
seu próprio contexto: não há memória das mensagens anteriores. Informe nomes
completos, em vez de depender de referências como “e essa empresa?”. A redação de
uma resposta positiva pode variar; a recusa deve conservar a frase literal.

Linhas vazias ou com espaços mostram uma orientação e não chamam banco/API.
`sair` (sem diferenciar maiúsculas/minúsculas), Ctrl+D/EOF ou Ctrl+C encerram sem
traceback e com código 0. O serviço só é inicializado após a primeira pergunta
com texto. Uma falha técnica apresenta diagnóstico seguro e encerra com código 1,
sem tentar novamente automaticamente; confira banco, configuração ou saldo antes
de reiniciar. Se a interrupção ocorrer durante uma chamada já enviada, ela pode
ter sido processada pelo provedor.

Cada pergunta válida usa o fluxo e os limites de custo descritos acima. O limite
do chat é por chamada, não um orçamento acumulado da sessão. Nenhum comando de
saída nem entrada em branco gera uma chamada paga.

## Como ler o código

Em `src/ingest.py`, comece pelo fluxo principal:

1. `main()` lê os argumentos e transforma erros em mensagens da CLI.
2. `run()` carrega a configuração, prepara o PDF e escolhe simulação ou ingestão.
3. `IngestionService.ingest()` em `src/ingestion/service.py` coordena a verificação,
   o orçamento, a geração dos vetores e a confirmação da persistência.
4. `EmbeddingService` em `embeddings.py` mantém o cliente OpenAI e valida os vetores.
5. `PgVectorRepository` em `persistence.py` concentra conexões, lock, PGVector
   e verificação de integridade.

`config.py` contém `Settings`, a configuração imutável com credenciais ocultadas
na representação do objeto. `pdf.py` contém funções para leitura, chunking,
metadata e IDs determinísticos. `errors.py` define o erro seguro da ingestão.
A CLI continua em `src/ingest.py`, com os mesmos argumentos e comandos.

Para a busca, `src/search.py` interpreta a pergunta e apresenta os resultados.
`SemanticSearchService` em `src/retrieval/service.py` valida a entrada e a
collection, cria o mesmo cliente de embeddings usado pela ingestão e solicita os
dez vizinhos mais similares. `retrieval/errors.py` mantém separados os erros
esperados que a CLI pode mostrar com segurança.

Para responder, `AnswerService` em `src/answering/service.py` coordena busca e
geração. `prompt.py` contém o prompt e a recusa literal; `model.py` concentra
orçamento, cliente de chat e validação da saída. `answering/__main__.py` recebe
uma pergunta e transforma falhas técnicas em mensagens seguras.

Em `src/chat.py`, `main()` trata encerramento e falhas; `run_session()` lê uma
pergunta, trata branco/saída e apresenta a resposta. `create_answer_service()`
conecta as dependências existentes na primeira pergunta válida. As duas CLIs de
resposta compartilham os diagnósticos seguros de `answering/cli.py`.

Leia `prepare_pdf()` de cima para baixo: carregar o PDF, selecionar páginas com
texto, preparar metadata e dividir em chunks. Os detalhes estão nas funções
auxiliares logo abaixo. Em `Settings.from_env()` e nos métodos do repositório,
o mesmo padrão separa as etapas de leitura e validação. O prefixo `_` indica
uma função ou método de uso interno, pela convenção do Python.

As anotações como `-> bool` indicam o tipo retornado; em `ingest()`, `True` significa dados novos e `False`
significa que o documento já estava completo. Elas ajudam a leitura, mas não
substituem as validações executadas pelo código.

## Verificações e solução de problemas

Para conferir o preparo do PDF e a entrada da CLI sem pagar por API:

```sh
python src/ingest.py --dry-run
printf '\nsair\n' | python src/chat.py
python -m pip check
docker compose config --quiet
```

O dry-run não acessa banco nem API. O chat ignora o branco e encerra sem criar os
serviços. Já executar ingestão pela primeira vez ou enviar uma pergunta válida
pode gerar custo: essas verificações reais são separadas das verificações locais.

| Situação | Ação |
| --- | --- |
| `python` ausente ou módulo não encontrado | Ative a venv na raiz com `source venv/bin/activate`; instale `requirements.txt`. Use `python3` para criar a venv. |
| Venv falha por ausência de `ensurepip` | Siga a alternativa de bootstrap do pip na seção de instalação; não instale pacotes no Python global. |
| Docker não responde | Inicie Docker Desktop/daemon. No WSL, confira a integração da distribuição nas configurações do Docker Desktop. Repita `docker compose ps -a`. |
| Porta 5432 ou 5050 ocupada | Ajuste `POSTGRES_PORT` ou `PGADMIN_PORT` no `.env` e recrie os serviços com `docker compose up -d`; a porta interna do banco continua 5432. |
| Compose recusa variável obrigatória | Copie `.env.example` para `.env` sem sobrescrever valores existentes e preencha as senhas e o e-mail local. |
| PostgreSQL recusa a senha | Confira a senha usada quando o volume foi inicializado; alterar `.env` não troca a senha de um usuário já existente. Não apague o volume para tentar corrigir. |
| Extension `vector` ausente | Confira `docker compose logs bootstrap_vector_ext`; espere o bootstrap concluir com código 0. |
| Collection ausente/vazia | Execute `python src/ingest.py` depois de confirmar o banco e o custo. |
| Collection ou dimensão incompatível | Mantenha o modelo/dimensão originais ou prepare outra collection e, se necessário, outro banco/schema compatível. Não há exclusão automática. |
| Falha OpenAI, saldo/cota ou autenticação | Confira a chave local, acesso ao modelo e Billing/Usage. Não publique a chave nem repita chamadas automaticamente. |
| LLM retorna saída truncada | O programa encerra com erro técnico. Revise o limite de saída no código e o tamanho da pergunta antes de repetir uma chamada paga. |

Validação do zero em 14/10/2026: venv nova com Python 3.14.4, instalação limpa,
56 testes locais aprovados e banco Compose isolado. Foram ingeridos 67 chunks
com vetores de 1.536 dimensões; o checksum foi preservado após reinício. A CLI
respondeu aos três fatos do gabarito e recusou literalmente informação ausente e
opinião. A prova real usou um lote de embeddings e cinco perguntas; unitários e
integrações sintéticas com PostgreSQL foram executados separadamente, sem OpenAI.
Nenhuma porcentagem de cobertura foi medida.

As dependências diretas estão fixadas e a instalação deve terminar com
`python -m pip check` sem conflitos. Dependências transitivas são resolvidas pelo
pip: `requirements.txt` não é um lock completo nem garante versões transitivas
idênticas em qualquer data ou sistema. A validação do zero registra o Python e os
pacotes efetivamente instalados; ela não comprova todas as versões de Python e
plataformas possíveis. Não atualize modelos/dimensões para resolver um problema
de instalação sem revisar a compatibilidade dos dados existentes.

O conjunto público contém a estrutura do desafio e `.gitignore`. `.env`, venv,
caches, logs, testes e documentos internos permanecem locais. A verificação de
segredos cobre o conjunto rastreado, o histórico Git alcançável, padrões de chave
OpenAI e valores locais conhecidos; não substitui uma auditoria de segurança
universal nem comprova ausência de toda vulnerabilidade nas dependências.

## Próximas etapas

A próxima história preparará a entrega pública. A exigência de fork do
repositório oficial e a submissão no canal do MBA continuam pendentes para a
entrega final.
