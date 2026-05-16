# Relatório de Auditoria de Faithfulness - Chunk-based

## Visão Geral

Este relatório apresenta os resultados da auditoria de faithfulness para as respostas dos modelos **ChatGPT** e **Gemini** sobre o tema "Desafios de LLMs Multilíngues e Português".

### Metodologia

- **Script**: `audit_answers_with_chunks.py`
- **Abordagem**: Avaliação por chunks de texto extraídos dos documentos de referência
- **Métrica**: Comparação de claims extraídas das respostas com os documentos de referência
- **Labels**: `SUPPORTED` (afirmação validada) e `UNSUPPORTED` (afirmação não validada)

---

## Métricas Consolidadas

| Indicador | Valor |
|-----------|-------|
| Total de arquivos processados | 2 |
| Total de arquivos gerados | 4 |
| **Total de claims SUPPORTED** | **272** |
| **Total de claims UNSUPPORTED** | **3001** |
| **Total geral de claims** | **3273** |

### Distribuição Percentual

| Label | Percentual |
|-------|-------------|
| SUPPORTED | 8.31% |
| UNSUPPORTED | 91.69% |

---

## Análise por Modelo

### ChatGPT

| Métrica | Valor |
|---------|-------|
| Claims SUPPORTED | 199 |
| Claims UNSUPPORTED | 1833 |
| Total de claims | 2032 |
| Percentual SUPPORTED | 9.79% |
| Percentual UNSUPPORTED | 90.21% |
| Chunks processados | 2032 |

### Gemini

| Métrica | Valor |
|---------|-------|
| Claims SUPPORTED | 73 |
| Claims UNSUPPORTED | 1168 |
| Total de claims | 1241 |
| Percentual SUPPORTED | 5.88% |
| Percentual UNSUPPORTED | 94.12% |
| Chunks processados | 1241 |

---

## Comparação entre Modelos

| Modelo | SUPPORTED | UNSUPPORTED | % SUPPORTED | Score Médio |
|--------|-----------|-------------|-------------|-------------|
| ChatGPT | 199 | 1833 | 9.79% | 0.598 |
| Gemini | 73 | 1168 | 5.88% | 0.575 |

### Observações

1. **ChatGPT apresenta melhor performance**: Maior proporção de claims SUPPORTED (9.79% vs 5.88%)
2. **Alta taxa de UNSUPPORTED em ambos**: Superior a 90% em ambos os modelos
3. **Score médio**: ChatGPT (0.598) ligeiramente superior ao Gemini (0.575)

---

## Exemplos de Claims

### ChatGPT - Claims SUPPORTED (exemplos)

| Chunk | Claim (trecho) |
|-------|----------------|
| 0 | "Resumo Executivo - Nos últimos cinco anos (2021–2026), avançaram-se significativamente os LLMs multilíngues..." |
| 11 | "A tokenização excessiva aumenta custos computacionais e latência..." |
| 22 | "Os indicadores de qualidade de dados linguísticos incluem: completeness, correctness, consistency..." |

### ChatGPT - Claims UNSUPPORTED (exemplos)

| Chunk | Claim (trecho) | Justificativa |
|-------|----------------|----------------|
| 1 | "Em 2024–2026 surgiram modelos especializados em português: Sabia (7B/65B, 2024) e Curió (1.1B, 2023)..." | O texto não fornece os tamanhos específicos dos modelos, anos exatos ou contagem de tokens |
| 2 | "recentemente o Tucano 2 (0.5–3.7B, 2026) usou 320B tokens em pt" | O texto confirma os parâmetros e tokens, mas não menciona o ano de 2026 |
| 3 | "Instrução e RLHF tornaram-se padrão de alinhamento (ex.: InstructGPT 2022, ChatGPT 2022)" | O texto discute técnicas de alinhamento (SFT e PO), mas não menciona os exemplos específicos |

### Gemini - Claims SUPPORTED (exemplos)

| Chunk | Claim (trecho) |
|-------|----------------|
| 4 | "O alinhamento semântico cross-lingual é, portanto, tanto um objetivo desejado quanto uma fonte de tensão técnica." |
| 27 | "A 'maldição da nocividade' é um risco real: modelos são mais propensos a gerar respostas prejudiciais..." |
| 42 | "A variação linguística do português (BR vs PT) introduz desafios adicionais..." |

### Gemini - Claims UNSUPPORTED (exemplos)

| Chunk | Claim (trecho) | Justificativa |
|-------|----------------|----------------|
| 0 | "Desafios Técnicos e Éticos no Desenvolvimento de Modelos de Linguagem Multilíngues..." | O texto não menciona especificamente a língua portuguesa |
| 1 | "Esta análise técnica e ética explora as complexidades... examinando os avanços entre 2021 e 2026..." | O texto não define o período específico de 2021-2026 |
| 2 | "a literatura define como a maldição da multilinguidade..." | O texto não menciona o termo específico "maldição da multilinguidade" |

---

## Arquivos Gerados

| Arquivo | Descrição |
|---------|------------|
| `answers_gemini_Desafios_LLMs_Multilingues_e_Portugues.json` | Resultado completo Gemini |
| `answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues.json` | Resultado completo ChatGPT |
| `answers_gemini_Desafios_LLMs_Multilingues_e_Portugues_SUPPORTED.json` | Apenas claims SUPPORTED (73 chunks) |
| `answers_gemini_Desafios_LLMs_Multilingues_e_Portugues_UNSUPPORTED.json` | Apenas claims UNSUPPORTED (1168 chunks) |
| `answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues_SUPPORTED.json` | Apenas claims SUPPORTED (199 chunks) |
| `answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues_UNSUPPORTED.json` | Apenas claims UNSUPPORTED (1833 chunks) |

---

## Conclusões

1. **Alta taxa de UNSUPPORTED (91.69%)**: A maioria das claims não possui suporte completo nos documentos de referência. Isso pode indicar:
   - Generalização excessiva por parte dos modelos
   - Informações não presentes nos documentos de referência
   - Necessidade de incluir mais fontes de referência

2. **ChatGPT teve melhor desempenho**: Com 9.79% de claims SUPPORTED vs 5.88% do Gemini, o ChatGPT demonstrou maior alinhamento com as fontes de referência.

3. **Padrões de falha comuns**:
   - Afirmações sobre anos específicos não mencionados nos documentos
   - Especificações de parâmetros de modelos não confirmadas
   - Termos técnicos não presentes nas referências literal

4. **Variação linguística**: O português (BR vs PT) representa um desafio adicional identificado nas análises.

---

*Relatório gerado em: 2026-05-16*
*Script de análise: `split_results_by_label.py`*