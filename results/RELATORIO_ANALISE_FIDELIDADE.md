# Relatório de Análise: Suporte Documental das Respostas

## Resumo Executivo

Este relatório apresenta os resultados da análise de fidelidade dos modelos **Gemini** e **ChatGPT** sobre o tema "Desafios de LLMs Multilíngues e Português". Os arquivos foram processados pelo script `split_results_by_label.py`, que separou as afirmações em duas categorias: **SUPPORTED** (com suporte documental) e **UNSUPPORTED** (sem suporte documental).

---

## Métricas Gerais

| Métrica | Valor |
|---------|-------|
| Total de arquivos processados | 2 |
| Total de arquivos gerados | 4 |
| Total de afirmações SUPPORTED | **22** |
| Total de afirmações UNSUPPORTED | **788** |
| Total geral de afirmações | **810** |

### Percentuais

| Categoria | Percentual |
|-----------|------------|
| SUPPORTED | 2.72% |
| UNSUPPORTED | **97.28%** |

---

## Comparativo por Modelo

### Gemini (Google)

| Categoria | Afirmações | Chunks |
|-----------|-----------|--------|
| SUPPORTED | 15 | 15 |
| UNSUPPORTED | 390 | 390 |
| **Total** | **405** | **405** |

### ChatGPT (OpenAI)

| Categoria | Afirmações | Chunks |
|-----------|-----------|--------|
| SUPPORTED | 7 | 7 |
| UNSUPPORTED | 398 | 398 |
| **Total** | **405** | **405** |

---

## Arquivos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `answers_gemini_Desafios_LLMs_Multilingues_e_Portugues_SUPPORTED.json` | Afirmações respaldadas - Gemini |
| `answers_gemini_Desafios_LLMs_Multilingues_e_Portugues_UNSUPPORTED.json` | Afirmações sem respaldo - Gemini |
| `answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues_SUPPORTED.json` | Afirmações respaldadas - ChatGPT |
| `answers_chatGPT_Desafios_LLMs_Multilingues_e_Portugues_UNSUPPORTED.json` | Afirmações sem respaldo - ChatGPT |

---

## Insights

### 1. Taxa de Unsupported Extremamente Alta

Ambos os modelos apresentam uma taxa de **97.28%** de afirmações sem suporte documental. Isso indica que:

- A metodologia de auditoria pode estar sendo muito rigorosa
- Os modelos tendem a gerar conteúdo que não pode ser verificado diretamente nos documentos-fonte
- Pode haver uma lacuna entre o conhecimento interno dos modelos e as evidências disponíveis nos documentos

### 2. Gemini Apresenta Maior Consistência Factual

O modelo **Gemini** conseguiu **15 afirmações suportadas**, mais que o dobro do **ChatGPT (7)**. Isso sugere:

- Gemini pode ter uma melhor capacidade de ancorar suas respostas em fontes documentais
- A estratégia de geração de conteúdo do Gemini pode estar mais alinhada com a verificação factual
- Ou há diferenças na forma como cada modelo cita informações dos documentos

### 3. Proporção Relativa de Qualidade

| Modelo | Ratio SUPPORTED/Total |
|--------|----------------------|
| Gemini | 15/405 = **3.70%** |
| ChatGPT | 7/405 = **1.73%** |

O **Gemini apresenta aproximadamente 2.1x mais afirmações verificáveis** que o ChatGPT em termos proporcionais.

### 4. Distribuição Uniforme de Chunks

Cada afirmação unsupported está associada a exatamente 1 chunk, indicando:

- Cada seção não verificada foi claramente demarcada
- A granularidade da análise permite identificar exatamente onde estão as lacunas de suporte

### 5. Implicações para Aplicações Práticas

- **Alta taxa de unsupported** pode ser problemática em contextos que exigem alta precisão factual (ex: jurídico, médico, acadêmico)
- A disparidade entre modelos sugere que **a escolha do modelo impacta diretamente na veracidade das respostas**
- Recomenda-se sempre implementar uma camada de verificação documental antes de utilizar as respostas em produção

### 6. Possíveis Causas do Alto Volume de Unsupported

1. **Geração de conteúdo inferencial**: Modelos geram conclusões que não estão literalmente nos documentos
2. **Questões de formatação**: Afirmações podem estar corretas mas não exatamente no formato esperado pelo verificador
3. **Diferenças de terminologia**: O vocabulário do documento pode diferir do usado na resposta
4. **Conhecimento paramétrico**: Modelos podem estar utilizando conhecimento interno não presente nos documentos

---

## Conclusão

A análise revela uma **discrepância significativa entre a geração de conteúdo e a verificabilidade documental** nos dois modelos avaliados. O Gemini demonstra vantagem quantitativa na geração de afirmações suportadas, mas ambos os modelos apresentam taxas de unsupported acima de 95%, o que reforça a necessidade de:

1. Implementar sistemas de verificação em cascata
2. Configurar limiares de confiança mais conservadores
3. Considerar abordagens híbridas (RAG + verificação pós-geração)

---

*Relatório gerado em: 06 de maio de 2026*  
*Scripts utilizados: `split_results_by_label.py`*  
*Diretório de saída: `results/`*