# Análise de Resultados - Auditoria de Faithfulness

## Resumo dos Resultados

| Métrica | Valor |
|---------|-------|
| Arquivos processados | 2 |
| Arquivos gerados | 4 |
| Total SUPPORTED | 25 (5.48%) |
| Total UNSUPPORTED | 431 (94.52%) |
| **Total geral** | **456** |

---

## Detalhamento por Ferramenta de DeepResearch

Os arquivos avaliados foram gerados por ferramentas de DeepResearch (Gemini e ChatGPT). A auditoria de faithfulness foi realizada utilizando o modelo **google/gemma-4-e4b**.

### Gemini (DeepResearch) - `answers_gemini_Desafios_LLMs_Multilingues_e_Portugues.json`

| Label | Quantidade | Chunks |
|-------|------------|--------|
| SUPPORTED | 17 | 17 |
| UNSUPPORTED | 415 | 415 |
| **Total** | **432** | **432** |

### ChatGPT (DeepResearch) - `answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues.json`

| Label | Quantidade | Chunks |
|-------|------------|--------|
| SUPPORTED | 8 | 8 |
| UNSUPPORTED | 16 | 16 |
| **Total** | **24** | **24** |

> **Nota**: O modelo utilizado na auditoria foi `google/gemma-4-e4b`, não o Gemini ou ChatGPT.

---

## Arquivos Gerados

```
results/
├── answers_gemini_Desafios_LLMs_Multilingues_e_Portugues_SUPPORTED.json
├── answers_gemini_Desafios_LLMs_Multilingues_e_Portugues_UNSUPPORTED.json
├── answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues_SUPPORTED.json
└── answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues_UNSUPPORTED.json
```

---

## Explicação dos Resultados

### O que significa cada label?

- **SUPPORTED**: A afirmação (claim) do modelo está respaldada pelos documentos de referência passados na auditoria
- **UNSUPPORTED**: A afirmação não possui suporte suficiente nos documentos de referência
- **CONTRADICTED**: A afirmação contradiz o conteúdo dos documentos (não presente nos dados processados)

### Análise dos Resultados

#### 1. Alta taxa de UNSUPPORTED (94.52%)

A maioria esmagadora das afirmações foi classificada como **UNSUPPORTED**. Isso pode indicar:

- **Alucinações**: O modelo generó informações que não estão presentes nos documentos de referência
- **Claims muito específicas**: As afirmações contêm detalhes muito granulares que não são encontrados nos documentos originais
- **Limitações na recuperação (RAG)**: O sistema pode não ter recuperado os trechos relevantes para suportar as afirmações

#### 2. Diferença entre ferramentas de DeepResearch

O **Gemini (DeepResearch)** processou **432 chunks** enquanto o **ChatGPT (DeepResearch)** processou apenas **24 chunks**. Essa diferença significativa pode ser explicada por:

- Diferentes documentos de referência utilizados em cada pesquisa
- Escopo diferente de busca de cada ferramenta
- Limitação de tempo ou tokens durante a execução

#### 3. Proporção SUPPORTED por ferramenta

- **Gemini (DeepResearch)**: 17/432 = **3.94%** de afirmações suportadas
- **ChatGPT (DeepResearch)**: 8/24 = **33.33%** de afirmações suportadas

O ChatGPT apresenta uma taxa de suporte significativamente maior (33.3%) comparado ao Gemini (3.9%). No entanto, é importante notar que a amostra do ChatGPT é muito menor (24 vs 432).

### Possíveis Causas da Alta Taxa de UNSUPPORTED

1. **Domínio técnico-especializado**: O tema "Desafios LLMs Multilingues e Portugues" envolve conceitos técnicos específicos que podem não estar presentes nos documentos de referência utilizados

2. **Fragmentação dos chunks**: Cada chunk tem ~500 tokens, o que pode limitar o contexto disponível para a auditoria

3. **Dados de treinamento**: Os modelos podem ter utilizado conhecimento de seus dados de treinamento ao invés de basear-se estritamente nos documentos fornecidos

4. **Qualidade dos documentos de referência**: Os documentos podem não conter informações específicas o suficiente para suportar as claims geradas

---

## Conclusão

Os resultados indicam uma **preocupação significativa de faithfulness** nas respostas geradas pelas ferramentas de DeepResearch (Gemini e ChatGPT), avaliadas pelo modelo `google/gemma-4-e4b`. A alta taxa de UNSUPPORTED (94.52%) sugere que as respostas geradas contêm afirmações que não são corroboradas pelos documentos de referência, o que pode ser um indicador de:

- Alucinações das ferramentas de DeepResearch
- Uso de conhecimento interno em vez de informações dos documentos fornecidos
- Limitations no processo de recuperação de informações relevantes

A diferença entre as taxas de SUPPORTED das duas ferramentas (3.94% vs 33.33%) também sugere que o ChatGPT pode ter um desempenho melhor em termos de groundedness nos documentos, embora a amostra seja muito menor.

---

*Relatório gerado automaticamente em: 2026-05-05*