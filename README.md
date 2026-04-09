# AuditorFidelidade

Sistema de auditoria factual que avalia a veracidade de afirmações em um texto, utilizando LLMs e embeddings vetoriais.

## Características

- **Extração de Afirmações**: Usa LLM para extrair afirmações factuais atômicas de um texto
- **Busca por Evidências**: Recupera passagens relevantes usando FAISS + similaridade de cosseno
- **Verificação**: Classifica afirmações como `SUPPORTED`, `PARTIAL` ou `NOT_SUPPORTED`
- **Pontuação**: Calcula score de 0 a 1 baseado nas verificações
- **Embeddings Locais**: Suporte a modelo local (`all-MiniLM-L6-v2`) via HuggingFace
- **Deep Research**: Integração com Perplexity e OpenAI para pesquisas aprofundadas
- **Barra de Progresso**: Visualização do andamento da auditoria
- **Resultados em JSON**: Salva resultados em arquivos JSON estruturados
- **API NVIDIA**: Integração com modelos da NVIDIA (DeepSeek LLM + NV-Embed)
- **POO**: Código estruturado com classes e serviços separados
- **Type Hints**: Sistema fortemente tipado
- **FAISS**: Banco vetorial para persistência e busca de embeddings

## Requisitos

- Python 3.8+
- API Key da NVIDIA (para modelos NVIDIA)
- API Key da OpenAI (para deep research)
- API Key da Perplexity (para deep research)

### Dependências

```
langchain
langchain-nvidia-ai-endpoints
langchain-community
langchain-core
langchain-huggingface
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
├── requirements.txt
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
        │   ├── llm_service.py       # Serviço LLM (LangChain)
        │   ├── embedding_service.py  # Serviço de Embeddings (NVIDIA ou HuggingFace)
        │   ├── vector_store.py      # VectorStore (FAISS)
        │   ├── perplexity_service.py # Deep Research via Perplexity
        │   └── openai_service.py    # Deep Research via OpenAI
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
# Configure as variáveis de ambiente no arquivo .env
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

### Exemplo de Resultado (INICIAL)

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
      "source_answer": "\n    Brazil was colonized by Portugal in the 16th century.\n    It became independent in 1822.\n    The capital of Brazil is Rio de Janeiro.\n    "
    },
    {
      "index": 1,
      "text": "The colonization of Brazil occurred in the 16th century.",
      "source_answer": "\n    Brazil was colonized by Portugal in the 16th century.\n    It became independent in 1822.\n    The capital of Brazil is Rio de Janeiro.\n    "
    },
    {
      "index": 2,
      "text": "Brazil became independent.",
      "source_answer": "\n    Brazil was colonized by Portugal in the 16th century.\n    It became independent in 1822.\n    The capital of Brazil is Rio de Janeiro.\n    "
    },
    {
      "index": 3,
      "text": "Brazil became independent in 1822.",
      "source_answer": "\n    Brazil was colonized by Portugal in the 16th century.\n    It became independent in 1822.\n    The capital of Brazil is Rio de Janeiro.\n    "
    },
    {
      "index": 4,
      "text": "The capital of Brazil is Rio de Janeiro.",
      "source_answer": "\n    Brazil was colonized by Portugal in the 16th century.\n    It became independent in 1822.\n    The capital of Brazil is Rio de Janeiro.\n    "
    }
  ],
  "results": [
    {
      "claim_index": 0,
      "claim_text": "Brazil was colonized by Portugal.",
      "label": "SUPPORTED",
      "justification": "The evidence directly confirms Brazil was colonized by Portugal during the 1500s and gained independence in 1822.",
      "passages": [
        "Brazil was colonized by Portugal during the 1500s. It gained independence in 1822.",
        "The capital of Brazil is Brasília, not Rio de Janeiro."
      ]
    },
    {
      "claim_index": 1,
      "claim_text": "The colonization of Brazil occurred in the 16th century.",
      "label": "SUPPORTED",
      "justification": "The evidence explicitly states Brazil was colonized by Portugal during the 1500s, which corresponds to the 16th century.",
      "passages": [
        "Brazil was colonized by Portugal during the 1500s. It gained independence in 1822.",
        "The capital of Brazil is Brasília, not Rio de Janeiro."
      ]
    },
    {
      "claim_index": 2,
      "claim_text": "Brazil became independent.",
      "label": "SUPPORTED",
      "justification": "The evidence explicitly states that Brazil gained independence in 1822, which directly supports the claim.",
      "passages": [
        "Brazil was colonized by Portugal during the 1500s. It gained independence in 1822.",
        "The capital of Brazil is Brasília, not Rio de Janeiro."
      ]
    },
    {
      "claim_index": 3,
      "claim_text": "Brazil became independent in 1822.",
      "label": "SUPPORTED",
      "justification": "The evidence explicitly states 'Brazil gained independence in 1822', which directly supports the claim.",
      "passages": [
        "Brazil was colonized by Portugal during the 1500s. It gained independence in 1822.",
        "The capital of Brazil is Brasília, not Rio de Janeiro."
      ]
    },
    {
      "claim_index": 4,
      "claim_text": "The capital of Brazil is Rio de Janeiro.",
      "label": "NOT_SUPPORTED",
      "justification": "The evidence explicitly states that Brasília is the capital of Brazil, not Rio de Janeiro.",
      "passages": [
        "The capital of Brazil is Brasília, not Rio de Janeiro.",
        "Brazil was colonized by Portugal during the 1500s. It gained independence in 1822."
      ]
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

### Variáveis de Ambiente

| Variável | Descrição | Obrigatório |
|----------|-----------|-------------|
| `NVIDIA_API_KEY` | API key da NVIDIA | Sim (se não usar embedding local) |
| `PERPLEXITY_TOKEN` | API key da Perplexity | Não |
| `OPEN_AI_TOKEN` | API key da OpenAI | Não |
| `USE_LOCAL_EMBEDDING` | Usar embedding local (`true`/`false`) | Não |
| `LOCAL_EMBEDDING_MODEL` | Modelo local (padrão: `sentence-transformers/all-MiniLM-L6-v2`) | Não |

### Arquivo .env

```env
NVIDIA_API_KEY=sua-chave-nvidia
USE_LOCAL_EMBEDDING=true
PERPLEXITY_TOKEN=sua-chave-perplexity
OPEN_AI_TOKEN=sua-chave-openai
```

## API Utilizada

### Audit Pipeline
- **LLM**: `deepseek-ai/deepseek-v3.1`
- **Embedding**: `nvidia/nv-embed-v1` (ou `sentence-transformers/all-MiniLM-L6-v2`)

### Deep Research
- **OpenAI**: `o4-mini-deep-research` (padrão) ou `o3-deep-research`
- **Perplexity**: `sonar-deep-research`

## Exemplos de Uso Avançado

### Usar embedding local

```python
# Via variável de ambiente
USE_LOCAL_EMBEDDING=true

# Ou via código
from src.auditor.services import EmbeddingService

embedding = EmbeddingService(use_local=True)
```

### Controlar tool calls no deep research

```python
client = OpenAIService()
response = client.deep_research("sua query", max_tool_calls=10)
```
