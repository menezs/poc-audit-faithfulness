# Relatório de Auditoria: Análise de Chunks - Direito Similar

## Visão Geral

Este relatório analisa a correspondência entre os **chunks originais** gerados pelos modelos ChatGPT e Gemini e os **resultados da auditoria de faithfulness**. O objetivo é identificar, para cada chunk original, se possui referência válida em documentos (SUPPORTED), se não possui suporte (UNSUPPORTED), ou se contradiz as evidências (CONTRADICTED).

---

## Resumo dos Dados

### Arquivos Analisados

| Fonte | Arquivo de Chunks | Total de Chunks Originais |
|-------|-------------------|--------------------------|
| ChatGPT | `ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_references.json` | 57 |
| Gemini | `Gemini_Direito_e_Tecnologia_Regulação_e_IA_references.json` | 65 |

### Resultados da Auditoria

| Modelo | Arquivo SUPPORTED | Avaliações | Arquivo UNSUPPORTED | Avaliações |
|--------|-------------------|------------|---------------------|------------|
| ChatGPT | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | 17 | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 40 |
| Gemini | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | 25 | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 40 |

---

## Análise por Modelo

### Metodologia

Cada chunk foi avaliado contra um documento de referência específico (relação 1:1 por índice). A avaliação segue três categorias:

- **SUPPORTED**: O documento de referência confirma as informações apresentadas no chunk
- **UNSUPPORTED**: O documento de referência não confirma as informações ou apresenta apenas evidências parciais
- **CONTRADICTED**: O documento de referência contradiz diretamente as informações do chunk

A auditoria utiliza similaridade semântica para encontrar os documentos de referência mais relevantes para cada chunk, através do pipeline de `AuditPipeline`.

---

## ChatGPT - Resultados Consolidados

### Quantidade de Chunks Originais Usados

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais** | 57 |
| **Chunks Avaliados** | 57 (17 SUPPORTED + 40 UNSUPPORTED) |
| **Chunks Não Avaliados** | 0 |

### Métricas de Verificação

| Indicador | Quantidade | Percentual |
|-----------|------------|-------------|
| **SUPPORTED** | 17 | 29,8% |
| **UNSUPPORTED** | 40 | 70,2% |
| **CONTRADICTED** | 0 | 0,0% |
| **Total Avaliado** | 57 | 100% |

### Distribuição Visual

```
SUPPORTED   █████████████████                                        17 (29.8%)
UNSUPPORTED ████████████████████████████████████████████████████████  40 (70.2%)
```

### Análise de Metodologia

O modelo ChatGPT apresentou uma taxa significativa de chunks UNSUPPORTED (70,2%). Os chunks classificados como SUPPORTED geralmente continham informações factuais verificáveis diretamente nos documentos de referência, como:

- Referências a artigos específicos da LGPD (art. 20, art. 6, art. 37)
- Informações sobre o PL 2.338/2023 e seu status legislativo
- Detalhes sobre a ANPD e seu papel de coordenação
- Especificações do ECA Digital (Lei 15.211/2025)

Os chunks UNSUPPORTED frequentemente apresentavam claims com:
- Múltiplas afirmações综合 onde apenas algumas eram verificáveis
- Estatísticas ou porcentagens específicas não presentes nas referências
- Referências a múltiplas legislações internacionais sem detalhes específicos

---

## Gemini - Resultados Consolidados

### Quantidade de Chunks Originais Usados

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais** | 65 |
| **Chunks Avaliados** | 65 (25 SUPPORTED + 40 UNSUPPORTED) |
| **Chunks Não Avaliados** | 0 |

### Métricas de Verificação

| Indicador | Quantidade | Percentual |
|-----------|------------|-------------|
| **SUPPORTED** | 25 | 38,5% |
| **UNSUPPORTED** | 40 | 61,5% |
| **CONTRADICTED** | 0 | 0,0% |
| **Total Avaliado** | 65 | 100% |

### Distribuição Visual

```
SUPPORTED   ████████████████████████████                                25 (38.5%)
UNSUPPORTED ████████████████████████████████████████████████████████████ 40 (61.5%)
```

### Análise de Metodologia

O modelo Gemini também apresentou maioria de chunks UNSUPPORTED (61,5%), porém com taxa superior de SUPPORTED em comparação ao ChatGPT (38,5% vs 29,8%). Os chunks SUPPORTED do Gemini tenderam a ser mais objetivos e factuais, contendo:

- Definições técnicas de conceitos de IA
- Informações sobre classificação de risco no PL 2338/2023
- Descrições da estrutura do ECA Digital
- Details sobre princípios regulatórios (segurança, transparência, não discriminação)

Os chunks UNSUPPORTED do Gemini apresentaram padrões similares ao ChatGPT:
- Tabelas com dados estatísticos sem fonte clara
- Previsões temporais específicas ("em 2026 consolidou-se")
- Afirmações quantitativas não presentes nas referências

---

## Comparação entre Modelos

| Métrica | ChatGPT | Gemini |
|---------|---------|--------|
| Total de chunks originais | 57 | 65 |
| Chunks avaliados | 57 | 65 |
| Chunks SUPPORTED | 17 (29,8%) | 25 (38,5%) |
| Chunks UNSUPPORTED | 40 (70,2%) | 40 (61,5%) |
| Chunks CONTRADICTED | 0 (0,0%) | 0 (0,0%) |

### Observações Principais

1. **Gemini apresenta taxa maior de SUPPORTED** (38,5% vs 29,8%), indicando melhor alinhamento entre seus chunks e os documentos de referência utilizados

2. **Ambos os modelos não apresentam casos de CONTRADICTED** - nenhum documento contradiz diretamente as claims analisadas, o que sugere que os modelos não geram informações claramente falsas em relação às fontes

3. **ChatGPT tem taxa mais elevada de UNSUPPORTED** (70,2%), possivelmente devido a chunks mais extensos com múltiplas afirmações que dificultam a verificação completa

4. **Chunks UNSUPPORTED concentram-se em**: estatísticas específicas, detalhamento legislativo excessivo, e afirmações sobre tendências futuras

---

## Análise de Justificativas de UNSUPPORTED

### Padrões Comuns - ChatGPT

1. **Afirmações múltiplas não verificáveis**: Chunks que combinam várias informações onde apenas algumas são confirmadas pelas referências
2. **Detalhamento regulatório específico**: Referências a artigos de lei específicos sem menção completa nas fontes
3. **Estatísticas ausentes**: Porcentagens e dados numéricos não presentes nos documentos de referência

### Padrões Comuns - Gemini

1. **Tabelas com dados não confirmados**: Informações tabulares com estatísticas específicas sem fontes verificáveis
2. **Temporalidades indefinidas**: Afirmações sobre marcos temporais ("em 2026", "consolidou-se") sem sustentação
3. **Terminologia técnica não detalhada**: Uso de termos específicos sem explicação presente nas referências

---

## Conclusões

### Achados Principais

1. **Taxa de SUPPORTED**: ~30% para ChatGPT e ~38% para Gemini
2. **Qualidade de referência**: Gemini apresenta desempenho superior na correlação entre chunks e documentos de referência
3. **Sem contradições**: Nenhum chunk foi classificado como CONTRADICTED em ambos os modelos

### Implicações

1. **Relevância dos documentos de referência**: A metodologia de similaridade semântica permite verificar a correspondência entre chunks e fontes
2. **Afirmações específicas são problemáticas**: Claims com estatísticas, anos específicos ou muitos detalhes têm maior chance de serem UNSUPPORTED
3. **Modelo Gemini mais preciso**: Maior taxa de SUPPORTED indica melhor alinhamento entre seus chunks e os documentos utilizados

### Recomendações

1. **Revisar documentos de referência**: Para chunks UNSUPPORTED, verificar se os documentos selecionados são os mais adequados
2. **Padronizar citações**: Para claims com informações específicas (anos, números), incluir referências mais precisas
3. **Considerar confiança das fontes**: Avaliar a qualidade e atualidade dos documentos de referência utilizados

---

## Arquivos Analisados

### Chunks Originais
- `chunks/direito/ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_references.json`
- `chunks/direito/Gemini_Direito_e_Tecnologia_Regulação_e_IA_references.json`

### Resultados da Auditoria
- `results/audit_with_chunks_direito_similar/answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json`
- `results/audit_with_chunks_direito_similar/answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json`
- `results/audit_with_chunks_direito_similar/answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json`
- `results/audit_with_chunks_direito_similar/answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json`

---

*Relatório gerado em: 2026-05-19*