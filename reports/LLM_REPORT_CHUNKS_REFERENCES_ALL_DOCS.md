# Relatório de Auditoria: Análise de Chunks vs Resultados

## Visão Geral

Este relatório analisa a correspondência entre os **chunks originais** (presentes em `./chunks/`) e os **resultados da auditoria** (presentes em `results/audit_with_chunks/`). O objetivo é identificar, para cada chunk original, se possui referência válida (SUPPORTED) ou não (UNSUPPORTED).

---

## Resumo dos Dados

### Arquivos Analisados

| Fonte | Arquivo de Chunks | Chunks Originais |
|-------|-------------------|------------------|
| ChatGPT | `chatGPT_Desafios_LLMs_Multilingues_e_Portugues_references.json` | 85 |
| Gemini | `gemini_Desafios_LLMs_Multilingues_e_Portugues_references.json` | 46 |

### Resultados da Auditoria

| Modelo | Arquivo de Resultado | Total Avaliações |
|--------|---------------------|------------------|
| ChatGPT | `answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues.json` | 2.040 |
| Gemini | `answers_gemini_Desafios_LLMs_Multilingues_e_Portugues.json` | 1.242 |

---

## Análise por Chunk Original

### Metodologia

Cada chunk foi avaliado contra **múltiplos documentos de referência** (24 para ChatGPT, 27 para Gemini). Um chunk é considerado:
- **SUPPORTED**: Se pelo menos um documento de referência confirmou a claim
- **UNSUPPORTED**: Se nenhum documento confirmou a claim

---

### ChatGPT - Resultados Consolidados

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais** | 85 |
| **Chunks com SUPPORTED** | 47 (55,3%) |
| **Chunks com UNSUPPORTED** | 38 (44,7%) |
| **Média de documentos por chunk** | 24 |

### ChatGPT - Distribuição por Chunk

```
SUPPORTED   ████████████████████████████████████████████████ 47 (55.3%)
UNSUPPORTED ████████████████████████████████               38 (44.7%)
```

### ChatGPT - Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 0 | "Resumo Executivo - Nos últimos cinco anos (2021–2026), avançaram-se significativamente os LLMs multilíngues..." |
| 4 | "mas esses dados de ajuste são, em grande parte, em inglês. Em termos de benchmarks, constatou-se que o português europeu é subrepresentado..." |
| 7 | "Entre desafios técnicos destacam-se: escassez e qualidade de dados em pt..." |
| 9 | "e custos computacionais de treinar LLMs na escala tradicional. Os desafios éticos incluem: viéses..." |
| 10 | "segurança (possível geração de conteúdo nocivo ou enviesado), privacidade..." |

### ChatGPT - Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 1 | "Em 2024–2026 surgiram modelos especializados em português: Sabia (7B/65B, 2024) e Curió (1.1B, 2023)..." | Texto não fornece tamanhos específicos dos modelos, anos exatos ou contagem de tokens |
| 2 | "recentemente o Tucano 2 (0.5–3.7B, 2026) usou 320B tokens em pt" | Texto confirma parâmetros e tokens, mas não menciona o ano 2026 |
| 8 | "tokenização ineficiente para português (fertilidade de subpalavras elevada..." | Termos específicos não encontrados nas referências |
| 11 | "desinformação (hallucinações em pt podem propagar fake news) e direitos autorais..." | Afirmações específicas não presentes nos documentos |

---

### Gemini - Resultados Consolidados

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais** | 46 |
| **Chunks com SUPPORTED** | 27 (58,7%) |
| **Chunks com UNSUPPORTED** | 19 (41,3%) |
| **Média de documentos por chunk** | 27 |

### Gemini - Distribuição por Chunk

```
SUPPORTED   ████████████████████████████████████████████████ 27 (58.7%)
UNSUPPORTED ████████████████████████                      19 (41.3%)
```

### Gemini - Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 4 | "O alinhamento semântico cross-lingual é, portanto, tanto um objetivo desejado quanto uma fonte de tensão técnica." |
| 27 | "A 'maldição da nocividade' é um risco real: modelos são mais propensos a gerar respostas prejudiciais..." |
| 29 | "A variação linguística do português (BR vs PT) introduz desafios adicionais..." |
| 19 | "A abordagem de fine-tuning em línguas de baixo recurso apresenta trade-offs entre performance e generalização." |
| 23 | "A avaliação de modelos multilíngues requer métricas específicas que considerem as particularidades linguísticas." |

### Gemini - Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 0 | "Desafios Técnicos e Éticos no Desenvolvimento de Modelos de Linguagem Multilíngues..." | Texto não menciona especificamente a língua portuguesa |
| 1 | "Esta análise técnica eética explora as complexidades... examinando os avanços entre 2021 e 2026..." | Período específico não definido nas referências |
| 2 | "a literatura define como a maldição da multilinguidade..." | Termo específico não presente nos documentos |
| 3 | "Para um modelo de tamanho fixo, cada novo idioma adicionado exige uma parcela do orçamento de parâmetros..." | Descrição técnica não encontrada |

---

## Análise por Documento de Referência

### ChatGPT - Detalhamento por Documento

| Documento | Chunks | SUPPORTED | UNSUPPORTED | % SUPPORTED |
|-----------|--------|-----------|-------------|-------------|
| doc_52 | 85 | 23 | 61 | 27,1% |
| doc_64 | 85 | 23 | 61 | 27,1% |
| doc_65 | 85 | 23 | 61 | 27,1% |
| doc_66 | 85 | 23 | 61 | 27,1% |
| doc_67 | 85 | 23 | 61 | 27,1% |
| doc_53 | 85 | 6 | 79 | 7,1% |
| doc_68 | 85 | 6 | 79 | 7,1% |
| doc_69 | 85 | 6 | 79 | 7,1% |
| doc_70 | 85 | 9 | 76 | 10,6% |
| doc_71 | 85 | 9 | 76 | 10,6% |
| ... | ... | ... | ... | ... |
| doc_59 | 85 | 0 | 85 | 0,0% |
| doc_60 | 85 | 1 | 84 | 1,2% |

**Top 5 Melhores Documentos** (maior suporte):
1. doc_52, doc_64-67: 27,1% de SUPPORTED

**Top 5 Piores Documentos** (menor suporte):
1. doc_59: 0,0% de SUPPORTED
2. doc_60, doc_62, doc_63, doc_77: 1,2% de SUPPORTED

### Gemini - Detalhamento por Documento

| Documento | Chunks | SUPPORTED | UNSUPPORTED | % SUPPORTED |
|-----------|--------|-----------|-------------|-------------|
| doc_29 | 46 | 10 | 36 | 21,7% |
| doc_19 | 46 | 8 | 38 | 17,4% |
| doc_18 | 46 | 6 | 40 | 13,0% |
| doc_23 | 46 | 5 | 41 | 10,9% |
| doc_17 | 46 | 4 | 42 | 8,7% |
| doc_34 | 46 | 4 | 42 | 8,7% |
| ... | ... | ... | ... | ... |
| doc_9 | 46 | 0 | 46 | 0,0% |
| doc_24 | 46 | 0 | 46 | 0,0% |
| doc_42 | 46 | 0 | 46 | 0,0% |
| doc_47 | 46 | 0 | 46 | 0,0% |

**Top 5 Melhores Documentos**:
1. doc_29: 21,7% de SUPPORTED
2. doc_19: 17,4% de SUPPORTED
3. doc_18: 13,0% de SUPPORTED

**Top 5 Piores Documentos**:
1. doc_9, doc_24, doc_42, doc_47: 0,0% de SUPPORTED

---

## Comparação entre Modelos

| Métrica | ChatGPT | Gemini |
|---------|---------|--------|
| Total de chunks originais | 85 | 46 |
| Chunks SUPPORTED | 47 (55,3%) | 27 (58,7%) |
| Chunks UNSUPPORTED | 38 (44,7%) | 19 (41,3%) |
| Média de docs por chunk | 24 | 27 |

### Observações

1. **Gemini apresenta taxa ligeiramente maior de SUPPORTED** (58,7% vs 55,3%)
2. **Ambos os modelos têm ~40-45% de chunks não suportados** pelas referências
3. **Alta variabilidade entre documentos**: Documentos de referência têm taxa de suporte de 0% a 27%

---

## Análise de Justificativas de UNSUPPORTED

### Padrões Comuns - ChatGPT

1. **Especificações técnicas ausentes**: O texto não fornece tamanhos específicos de modelos, anos exatos ou contagem de tokens
2. **Termos não mencionados**: O texto não contém menção a termos específicos como "PORTULAN ExtraGLUE" ou "ALBA"
3. **Afirmações generalizadas**: O texto discute técnicas mas não menciona exemplos específicos

### Padrões Comuns - Gemini

1. **Foco no português ausente**: O texto não menciona especificamente a língua portuguesa
2. **Período não definido**: O texto não define o período específico (2021-2026)
3. **Terminologia não presente**: Termos técnicos específicos não encontrados nas referências

---

## Conclusões

### Achados Principais

1. **Taxa de SUPPORTED por chunk**: ~55-59% dos chunks possuem pelo menos uma referência válida
2. **Qualidade dos documentos de referência**: Alta variabilidade (0% a 27% de suporte) - alguns documentos são muito mais úteis que outros
3. **Modelo Gemini** apresenta resultado ligeiramente superior em termos de taxa de SUPPORTED por chunk

### Implicações

1. **Documentos de referência são desiguais**: Alguns documentos (doc_52 para ChatGPT, doc_29 para Gemini) oferecem muito mais suporte que outros
2. **Afirmações específicas são problemáticas**: Claims que incluem anos específicos, parâmetros de modelos, ou nomes próprios têm maior chance de serem UNSUPPORTED
3. **Generalizações são melhor suportadas**: Afirmações mais genéricas sobre desafios técnicos tendem a ter mais suporte

### Recomendações

1. **Revisar documentos de referência**: Documentos com 0% de suporte podem não ser relevantes para o tema
2. **Padronizar citações**: Para claims com informações específicas (anos, números), incluir referências mais precisas
3. **Aumentar diversidade de fontes**: Adicionar mais documentos de referência pode melhorar a taxa de suporte

---

## Arquivos Analisados

### Chunks Originais
- `chunks/chatGPT_Desafios_LLMs_Multilingues_e_Portugues_references.json`
- `chunks/gemini_Desafios_LLMs_Multilingues_e_Portugues_references.json`

### Resultados da Auditoria
- `results/audit_with_chunks/answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues.json`
- `results/audit_with_chunks/answers_gemini_Desafios_LLMs_Multilingues_e_Portugues.json`

---

*Relatório gerado em: 2026-05-16*