# AuditorFidelidade

Sistema de auditoria factual que avalia a veracidade de afirmações em um texto, utilizando LLMs e embeddings vetoriais.

## Características

- **Extração de Afirmações**: Usa LLM para extrair afirmações factuais atômicas de um texto
- **Busca por Evidências**: Recupera passagens relevantes usando FAISS + similaridade de cosseno
- **Verificação**: Classifica afirmações como `SUPPORTED`, `PARTIAL` ou `NOT_SUPPORTED`
- **Pontuação**: Calcula score de 0 a 1 baseado nas verificações
- **Embeddings Locais**: Usa `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
- **Deep Research**: Integração com Perplexity e OpenAI para pesquisas aprofundadas
- **Barra de Progresso**: Visualização do andamento da auditoria
- **Resultados em JSON**: Salva resultados em arquivos JSON estruturados
- **Múltiplos Providers LLM**: Suporte a OpenAI (inclui LM Studio) e Ollama
- **POO**: Código estruturado com classes e serviços separados
- **Type Hints**: Sistema fortemente tipado
- **FAISS**: Banco vetorial para persistência e busca de embeddings

## Requisitos

- Python 3.8+
- API Key da OpenAI (para modelos OpenAI/LM Studio) - opcional para Ollama
- API Key da Perplexity (para deep research)
- Ollama ou LM Studio instalados localmente (opcional)

### Dependências

```
langchain
langchain-community
langchain-core
langchain-huggingface
langchain-ollama
langchain-openai
faiss-cpu
numpy
tqdm
requests
perplexity
openai
sentence-transformers
dotenv
```

## Instalação

```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

```
AuditorFidelidade/
├── main.py                    # Entry point
├── teste.py                   # Exemplo de uso do ReferenceExtractor
├── requirements.txt
├── .env                       # Variáveis de ambiente
├── .env.example               # Exemplo de variáveis de ambiente
├── README.md
└── src/
    └── auditor/
        ├── __init__.py       # Versão e exports
        ├── config/
        │   └── settings.py  # Configurações (Settings)
        ├── models/
        │   ├── claim.py           # Modelo Claim
        │   └── verification_result.py  # Modelo VerificationResult
        ├── services/
        │   ├── llm_service.py       # Serviço LLM (OpenAI/Ollama)
        │   ├── embedding_service.py  # Serviço de Embeddings (HuggingFace)
        │   ├── vector_store.py      # VectorStore (FAISS)
        │   ├── perplexity_service.py # Deep Research via Perplexity
        │   ├── openai_service.py    # Deep Research via OpenAI
        │   ├── reference_extractor.py # Extração de referências de markdown
        │   └── file_converter.py    # Conversão de arquivos para markdown
        ├── core/
        │   ├── claim_extractor.py    # Extrator de Claims
        │   ├── retriever.py          # Recuperador de evidências (FAISS)
        │   ├── verifier.py           # Verificador
        │   └── scorer.py             # Calculador de score
        ├── pipeline/
        │   └── audit_pipeline.py    # Pipeline principal
        └── utils/
            └── helpers.py            # Utilitários (print_result, save_result_json)
```

## Uso

### Executar o teste padrão

```bash
python main.py
```

### Usar como módulo

```python
from src.auditor import AuditPipeline

pipeline = AuditPipeline.create_from_env()

answer = """
Brazil was colonized by Portugal in the 16th century.
It became independent in 1822.
The capital of Brazil is Rio de Janeiro.
"""

documents = [
    "Brazil was colonized by Portugal during the 1500s. It gained independence in 1822.",
    "The capital of Brazil is Brasília, not Rio de Janeiro."
]

result = pipeline.audit(answer, documents)

print(f"Score: {result.score}")
print(f"Supported: {result.total_supported}")
print(f"Not Supported: {result.total_not_supported}")

for r in result.results:
    print(f"Claim: {r.claim.text}")
    print(f"Label: {r.label.value}")
```

### Exemplo de Resultado

```json
{
  "timestamp": "20260408_182957",
  "score": 0.8,
  "total_supported": 4,
  "total_partial": 0,
  "total_not_supported": 1,
  "is_fully_supported": false,
  "claims": [
    {
      "index": 0,
      "text": "Brazil was colonized by Portugal.",
      "source_answer": "..."
    }
  ],
  "results": [
    {
      "claim_index": 0,
      "claim_text": "Brazil was colonized by Portugal.",
      "label": "SUPPORTED",
      "justification": "The evidence directly confirms Brazil was colonized by Portugal during the 1500s.",
      "passages": ["Brazil was colonized by Portugal during the 1500s."]
    }
  ]
}
```

### Salvar e carregar vector store

```python
pipeline.audit(answer, documents)
pipeline.save_vector_store("./vector_store")
```

```python
pipeline = AuditPipeline.create_from_env()
pipeline.load_vector_store("./vector_store")
result = pipeline.audit(answer, documents)
```

### Extração de Referências de Markdown

```python
from src.auditor.services import LLMService, ReferenceExtractor

llm = LLMService(
    api_key="",
    model="google/gemma-4-e4b",
    provider="openai",
    base_url="http://localhost:1234"
)

extractor = ReferenceExtractor(llm_service=llm)
result = extractor.extract_from_markdown("./documento.md")
```

### Deep Research com OpenAI

```python
from src.auditor.services import OpenAIService

client = OpenAIService()
response = client.deep_research("Who is Sam Bankman-Fried?")
client.save_response_json(response, "./results/pesquisa.json")
client.save_text(response, "./results/pesquisa.txt")
```

### Deep Research com Perplexity

```python
from src.auditor.services import PerplexityService

perplexity = PerplexityService()
results = perplexity.search("latest AI developments 2024")
perplexity.save_results_to_json(results, "./results/pesquisa.json")
```

## Cálculo do Score

O score é calculado com base nas verificações das afirmações:

```
score = (suportadas + 0.5 * parciais) / total
```

**Exemplo:**
```
resposta parcialmente fundamentada
10 claims
5 suportadas
3 parciais
2 não suportadas
score = (5 + 0.5*3) / 10 = 0.65 
```

**Interpretação:**
- `SUPPORTED`: Afirmação validada pelas evidências (peso 1.0)
- `PARTIAL`: Afirmação parcialmente validada (peso 0.5)
- `NOT_SUPPORTED`: Afirmação não validada (peso 0.0)

**Limitação**
```
- Todas as claims têm a mesma importância
- Cada claim contribui igualmente para o score final
```
O score está tratando:
```
"erro trivial" == "erro crítico"
```

## Configurações

### Providers LLM

O projeto suporta dois providers:

| Provider | Descrição |
|----------|-----------|
| `openai` | OpenAI API, LM Studio, ou qualquer API OpenAI-compatible |
| `ollama` | Ollama local |

### Variáveis de Ambiente

| Variável | Descrição | Obrigatório |
|----------|-----------|-------------|
| `OPENAI_API_KEY` | API key da OpenAI | Sim (exceto para Ollama) |
| `PERPLEXITY_TOKEN` | API key da Perplexity | Não |
| `LLM_PROVIDER` | Provider LLM (`openai` ou `ollama`) | Não (padrão: `openai`) |
| `LLM_BASE_URL` | URL base do LLM | Não |
| `LLM_MODEL` | Modelo LLM a usar | Não |
| `EMBEDDING_MODEL` | Modelo de embedding (padrão: `sentence-transformers/all-MiniLM-L6-v2`) | Não |
| `EMBEDDING_CACHE_FOLDER` | Pasta de cache dos modelos (padrão: `./model_cache`) | Não |

### Arquivo .env

```env
OPENAI_API_KEY=sua-chave-openai
PERPLEXITY_TOKEN=sua-chave-perplexity

# Para LM Studio ou Ollama
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:1234
LLM_MODEL=google/gemma-4-e4b

# Para Ollama
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=gpt-oss:20b
```

### Usar LM Studio

1. Baixe e instale o LM Studio
2. Baixe o modelo desejado (ex: `google/gemma-4-e4b`)
3. Inicie o servidor na interface (clicando em "Start Server")
4. Configure o `.env`:

```env
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:1234
LLM_MODEL=google/gemma-4-e4b
```

### Usar Ollama

1. Instale o Ollama
2. Baixe o modelo desejado: `ollama pull gpt-oss:20b`
3. Configure o `.env`:

```env
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=gpt-oss:20b
```

## API Utilizada

### Audit Pipeline
- **LLM**: Configurável via `LLM_MODEL` (padrão: `google/gemma-4-e4b`)
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`

### Deep Research
- **OpenAI**: `o4-mini-deep-research` (padrão) ou `o3-deep-research`
- **Perplexity**: `sonar-deep-research`

## Exemplos de Uso Avançado

### Usar diferente provider

```python
from src.auditor.services import LLMService

# Com OpenAI/LM Studio
llm = LLMService(
    api_key="sk-...",
    model="google/gemma-4-e4b",
    provider="openai",
    base_url="http://localhost:1234"
)

# Com Ollama
llm = LLMService(
    api_key="",
    model="gpt-oss:20b",
    provider="ollama",
    base_url="http://localhost:11434"
)
```

### Controlar tool calls no deep research

```python
client = OpenAIService()
response = client.deep_research("sua query", max_tool_calls=10)
```