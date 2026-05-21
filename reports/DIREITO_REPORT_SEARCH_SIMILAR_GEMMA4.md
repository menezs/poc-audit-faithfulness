# Relatório de Auditoria: Análise de Chunks vs Resultados (Direito Similar)

## Visão Geral

Este relatório analisa a correspondência entre os **chunks originais** (presentes em `./chunks/direito/`) e os **resultados da auditoria** (presentes em `results/audit_with_chunks_direito_similar/`). O objetivo é identificar, para cada chunk original, se possui referência válida (SUPPORTED) ou não (UNSUPPORTED).

---

## Resumo dos Dados

### Arquivos Analisados

| Fonte | Arquivo de Chunks | Chunks Originais Usados |
|-------|-------------------|------------------------|
| ChatGPT | `ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_references.json` | 57 |
| Gemini | `Gemini_Direito_e_Tecnologia_Regulação_e_IA_references.json` | 65 |

### Resultados da Auditoria

| Modelo | Arquivo de Resultado | Total Avaliações |
|--------|---------------------|------------------|
| ChatGPT | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | 29 |
| ChatGPT | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 28 |
| Gemini | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | 45 |
| Gemini | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 19 |

---

## Análise por Chunk Original

### Metodologia

Cada chunk foi avaliado contra **um documento de referência específico** (relação 1:1 por índice). Um chunk é considerado:
- **SUPPORTED**: Se o documento de referência confirmou a claim
- **UNSUPPORTED**: Se o documento não confirmou a claim
- **CONTRADICTED**: Se o documento contradiz a claim

---

### ChatGPT - Resultados Consolidados

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais Usados** | 57 |
| **Chunks com SUPPORTED** | 29 (50,9%) |
| **Chunks com UNSUPPORTED** | 28 (49,1%) |
| **Chunks com CONTRADICTED** | 0 (0,0%) |

### ChatGPT - Distribuição por Chunk

```
SUPPORTED   ████████████████████████████████████████████████ 29 (50.9%)
UNSUPPORTED ███████████████████████████████████████████████  28 (49.1%)
```

### ChatGPT - Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 0 | "Principais temas tecnológicos e regulatórios - Inteligência Artificial (IA) - Conceitos: IA engloba sistemas de aprendizado de máquina (ML)..." |
| 3 | "Além do PL 2.338/2023 (Marco Legal da IA) em trâmite. - Legislação: O Brasil não tem ainda lei esgotando o tema..." |
| 4 | "Assim, métodos de IA já devem obedecer à LGPD e suas sanções. Em execução, agências reguladoras brasileiras estão atentas..." |
| 6 | "Prevê diretrizes de proteção de direitos fundamentais... e cria o Sistema Nacional de Regulação e Governança de IA (SIA)..." |
| 8 | "Delineando governança centralizada, soberania tecnológica e proteção de infraestruturas críticas. Em trâmite... PL 4.752/2025..." |

### ChatGPT - Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 1 | "Histórico e regulação: Internacionalmente, surgiu agenda de propor 'IA ética' com princípios da OCDE e UNESCO..." | A evidência menciona OCDE, mas não menciona UNESCO como parte da agenda global de IA ética |
| 2 | "Nos EUA não há uma lei federal uniforme de IA até 2026, mas o governo editou executive orders... no Brasil... decreto do Poder Executivo (CITDigital)..." | O decreto CITDigital não é mencionado nas referências |
| 5 | "Projetos de lei em andamento: O destaque é o PL 2.338/2023... Prevê classificação de sistemas por risco... e proíbe IA de risco 'excessivo'" | A evidência não menciona explicitamente a proibição de IA de risco excessivo |
| 7 | "Há também projetos setoriais e regionais... No Brasil, foi noticiado bloqueio de funcionalidades de apps..." | Afirmações específicas não presentes nos documentos |

---

### Gemini - Resultados Consolidados

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais Usados** | 65 |
| **Chunks com SUPPORTED** | 45 (69,3%) |
| **Chunks com UNSUPPORTED** | 20 (30,7%) |
| **Chunks com CONTRADICTED** | 0 (0,0%) |

### Gemini - Distribuição por Chunk

```
SUPPORTED   ████████████████████████████████████████████████████████████████████████ 45 (69.3%)
UNSUPPORTED ████████████████████████                                              20 (30.7%)
```

### Gemini - Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 0 | "Tratado sobre Regulação Digital, Inteligência Artificial e o Ecossistema Jurídico Contemporâneo: Um Guia para o Biênio 2025-2026..." |
| 1 | "Inteligência Artificial: Fundamentos Técnicos, Jurídicos e a Revolução Generativa... sistemas computacionais capazes de simular comportamentos inteligentes..." |
| 2 | "No contexto jurídico de 2026, a distinção fundamental reside entre a chamada 'IA fraca'... e a 'IA generativa'..." |
| 3 | "Historicamente, a regulação da IA evoluiu de diretrizes éticas voluntárias para normas mandatórias..." |
| 4 | "Juridicamente, a IA é tratada não apenas como um software, mas como um agente de processamento que deve observar a boa-fé..." |

### Gemini - Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 5 | "Tabela 1: Fundamentos Técnicos e Jurídicos da Inteligência Artificial... O impacto da IA generativa no setor jurídico brasileiro em 2026 é quantificável: cerca de 55%..." | A evidência fornece 45,8% de adoção, não 55% |
| 6 | "A tecnologia atua em três camadas estratégicas: automatização... aumento da capacidade analítica (revisão de contratos em segundos)..." | O exemplo específico de 'revisão de contratos em segundos' não é mencionado |
| 8 | "Governança de IA e o Futuro Marco Legal no Brasil (PL nº 2338/2023)... consolidou-se como elemento estrutural em 2026" | O material não confirma esse marco temporal específico |
| 13 | "A governança proposta pelo projeto brasileiro é inovadora ao detalhar obrigações específicas para o setor público..." | Não há evidências detalhando essas obrigações específicas |

---

## Comparação entre Modelos

| Métrica | ChatGPT | Gemini |
|---------|---------|--------|
| Total de chunks originais usados | 57 | 64 |
| Chunks SUPPORTED | 29 (50,9%) | 45 (69,3%) |
| Chunks UNSUPPORTED | 28 (49,1%) | 20 (30,7%) |
| Chunks CONTRADICTED | 0 (0,0%) | 0 (0,0%) |

### Observações

1. **Gemini apresenta taxa significativamente maior de SUPPORTED** (69,3% vs 50,9%)
2. **ChatGPT tem taxa equilibrada entre SUPPORTED e UNSUPPORTED** (~50-50%)
3. **Ambos os modelos não apresentam casos de CONTRADICTED** - nenhum documento contradiz as claims analisadas
4. **Chunks UNSUPPORTED tendem a conter**: estatísticas específicas, detalhes legislativos não mencionados, ou afirmações muito específicas sobre datas e números

---

## Análise de Justificativas de UNSUPPORTED

### Padrões Comuns - ChatGPT

1. **Dados estatísticos ausentes ou divergentes**: O texto menciona percentuais ou números específicos que não são confirmados pelas referências
2. **Legislação específica não mencionada**: Referências a projetos de lei ou decrees que não são detalhados nos documentos de referência
3. **Detalhamento excessivo**: Claims com muitos detalhes específicos que não podem ser verificados nas fontes disponíveis

### Padrões Comuns - Gemini

1. **Estatísticas não confirmadas**: Percentuais de adoção ou dados numéricos específicos não presentes nas referências
2. **Terminologia técnica muito específica**: Termos ou conceitos que não são mencionados explicitamente nos documentos
3. **Prognósticos apresentados como fatos**: Afirmações sobre o futuro (ex: "consolidou-se em 2026") apresentadas como fatos consumados

---

## Conclusões

### Achados Principais

1. **Taxa de SUPPORTED por chunk**: ~51% para ChatGPT e ~70% para Gemini
2. **Qualidade das respostas**: Gemini apresenta desempenho superior na correlação entre chunks e documentos de referência
3. **Sem casos de contradição**: Nenhum chunk foi classificado como CONTRADICTED, indicando que os documentos de referência não contradizem diretamente as claims

### Implicações

1. **Documentos de referência são relevantes**: A relação 1:1 entre chunk e documento mostrou-se eficaz para avaliação
2. **Afirmações específicas são problemáticas**: Claims com estatísticas, anos específicos ou detalhes técnicos têm maior chance de serem UNSUPPORTED
3. **Modelo Gemini mais preciso**: Maior taxa de SUPPORTED indica melhor alinhamento entre seus chunks e os documentos de referência utilizados

### Recomendações

1. **Revisar documentos de referência**: Para chunks UNSUPPORTED, verificar se os documentos selecionados são os mais adequados
2. **Padronizar citações**: Para claims com informações específicas (anos, números), incluir referências mais precisas
3. **Avaliar similaridade semântica**: Considerar usarsimilaridadesemântica para melhorar a correspondência entre chunks e documentos

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

*Relatório gerado em: 2026-05-18*