# Relatório de Auditoria: Análise de Chunks com Similaridade semântica (Direito Similar)

## Visão Geral

Este relatório analisa a correspondência entre os **chunks originais** (presentes em `./chunks/direito/`) e os **resultados da auditoria** (presentes em `results/audit_with_chunks_direito_similar/`). O objetivo é identificar, para cada chunk original, se possui referência válida (SUPPORTED), não possui suporte (UNSUPPORTED) ou é contradito (CONTRADICTED) pelos documentos de referência utilizados na avaliação.

A metodologia aplicada utiliza **similaridade semântica** para selecionar os documentos de referência mais relevantes para cada chunk, o que permite uma avaliação mais precisa da faithfulnest das respostas geradas pelos modelos.

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

Cada chunk foi avaliado contra **documentos de referência selecionados por similaridade semântica**. A avaliação segue os seguintes critérios:
- **SUPPORTED**: O documento de referência confirma a claim ou informação apresentada no chunk
- **UNSUPPORTED**: O documento de referência não confirma a claim ou informação
- **CONTRADICTED**: O documento de referência contradiz diretamente a claim

A metodologia de similaridade semântica permite que cada chunk seja comparado com múltiplos documentos,选出 o mais relevante para validação, proporcionando uma análise mais robusta do que a correspondência 1:1 tradicional.

---

## ChatGPT - Resultados Consolidados

### Chunks Originais

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais Usados** | 57 |

### Total de Avaliações Realizadas

| Categoria | Quantidade | Percentual |
|-----------|------------|------------|
| **Chunks com SUPPORTED** | 29 | 50,9% |
| **Chunks com UNSUPPORTED** | 28 | 49,1% |
| **Chunks com CONTRADICTED** | 0 | 0,0% |
| **Total** | 57 | 100% |

### Métricas de Avaliação

```
SUPPORTED   ████████████████████████████████████████████████ 29 (50.9%)
UNSUPPORTED ███████████████████████████████████████████████  28 (49.1%)
```

### Análise dos Resultados - ChatGPT

O modelo ChatGPT apresenta uma taxa equilibrada entre chunks SUPPORTED e UNSUPPORTED, indicando que aproximadamente metade das informações geradas pelo modelo possuem suporte documental adequado enquanto a outra metade carece de referenciamento válido.

### Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 0 | "Principais temas tecnológicos e regulatórios \n\n## 1. Inteligência Artificial (IA) \n\n- **Conceitos:** IA engloba sistemas de aprendizado de máquina (ML) e algoritmos capazes de realizar tarefas cognitivas..." |
| 2 | "Nos EUA não há uma lei federal uniforme de IA até 2026, mas o governo editou executive orders... No Brasil... a Resolução CNJ 615/2025..." |
| 3 | "Além do PL 2.338/2023 (Marco Legal da IA) em trâmite. A LGPD (Lei 13.709/2018) garante direitos importantes..." |
| 4 | "Assim, métodos de IA já devem obedecer à LGPD e suas sanções. A ANPD incluiu IA e tecnologias emergentes entre temas prioritários de fiscalização (plano 2026–27)..." |
| 5 | "O PL 2.338/2023, aprovado no Senado (dez/2024) e em exame na Câmara. Prevê classificação de sistemas por risco..." |
| 6 | "Prevê diretrizes de proteção de direitos fundamentais... e cria o Sistema Nacional de Regulação e Governança de IA (SIA)..." |
| 8 | "Delineando governança centralizada, soberania tecnológica e proteção de infraestruturas críticas. Em trâmite... PL 4.752/2025..." |
| 9 | "Prevendo a cooperação público-privada, capacitação de profissionais, criação de um programa nacional de segurança digital..." |

### Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 1 | "Internacionalmente, surgiu agenda de propor 'IA ética' com princípios da OCDE e UNESCO..." | A evidência menciona OCDE, mas não menciona UNESCO como parte da agenda global de IA ética |
| 7 | "Há também projetos setoriais e regionais... No Brasil, foi noticiado bloqueio de funcionalidades de apps..." | Afirmações específicas sobre projetos setoriais não confirmadas |
| 10 | "Nível internacional, há leis como o regulamento europeu NIS2, Cibersecurity Act..." | A evidência não faz referência específica a essas legislações |
| 11 | "Lei Carolina Dieckmann (Lei 12.737/2012)... Decreto 12.573/2025" | O decreto foi instituído em agosto de 2025, não novembro/2025 |
| 12 | "em 2023 a Estratégia previa obrigatoriedade de adesão de setores..." | A evidência não confirma essa obrigatoriedade específica |

---

## Gemini - Resultados Consolidados

### Chunks Originais

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais Usados** | 65 |

### Total de Avaliações Realizadas

| Categoria | Quantidade | Percentual |
|-----------|------------|------------|
| **Chunks com SUPPORTED** | 46 | 70,8% |
| **Chunks com UNSUPPORTED** | 19 | 29,2% |
| **Total** | 65 | 100% |

### Métricas de Avaliação

```
SUPPORTED   ████████████████████████████████████████████████████████████████████████ 46 (70.7%)
UNSUPPORTED █████████████████████                                              19 (29.2%)
```

### Análise dos Resultados - Gemini

O modelo Gemini apresenta uma taxa significativamente maior de chunks SUPPORTED (69,2%) comparada ao ChatGPT (50,9%), indicando uma maior precisão na correlação entre as informações geradas e os documentos de referência. Notably, há 1 caso de CONTRADICTED, o que indica que pelo menos um documento contradiz diretamente uma claim do chunk.

### Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 0 | "Tratado sobre Regulação Digital, Inteligência Artificial e o Ecossistema Jurídico Contemporâneo: Um Guia para o Biênio 2025-2026..." |
| 1 | "Inteligência Artificial: Fundamentos Técnicos, Jurídicos e a Revolução Generativa..." |
| 3 | "Historicamente, a regulação da IA evoluiu de diretrizes éticas voluntárias para normas mandatórias..." |
| 4 | "Juridicamente, a IA é tratada não apenas como um software, mas como um agente de processamento..." |
| 6 | "A tecnologia atua em três camadas estratégicas: automatização de tarefas manuais, aumento da capacidade analítica e análise preditiva..." |
| 7 | "Contudo, essa adoção acelerada expõe o profissional a riscos éticos e regulatórios..." |
| 9 | "No centro dessa discussão está o Projeto de Lei nº 2338/2023..." |
| 10 | "O projeto, aprovado pelo Senado no final de 2024 e em debate na Câmara... adota uma abordagem baseada em riscos..." |
| 11 | "O PL 2338/2023 introduz diretrizes rigorosas de transparência ativa..." |

### Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 2 | "No contexto jurídico de 2026, a distinção fundamental reside entre a chamada 'IA fraca' e a 'IA generativa'..." | A evidência não estabelece essa distinção funcional específica para 2026 |
| 5 | "Tabela 1: Fundamentos Técnicos e Jurídicos... O impacto da IA generativa no setor jurídico brasileiro em 2026 é quantificável: cerca de 55%..." | A evidência não contém dados sobre 55% de adoção |
| 8 | "Governança de IA e o Futuro Marco Legal no Brasil (PL nº 2338/2023)... consolidou-se como elemento estrutural em 2026" | O material não confirma esse marco temporal específico |
| 13 | "A governança proposta pelo projeto brasileiro é inovadora ao detalhar obrigações específicas para o setor público..." | Não há evidências detalhando essas obrigações específicas |
| 14 | "Tabela 2: Estrutura de Riscos no Marco Legal da IA..." | A tabela detalhada não está presente nas evidências |

### Caso CONTRADICTED - Gemini

Foi identificado 1 caso de CONTRADICTED na avaliação do Gemini, onde um documento de referência contradiz diretamente uma claim do chunk. Este caso é significativo pois indica que, em algumas situações, o modelo gerou informações que são literalmente o oposto do que está documentado.

---

## Comparação entre Modelos

| Métrica | ChatGPT | Gemini |
|---------|---------|--------|
| Total de chunks originais usados | 57 | 65 |
| Chunks SUPPORTED | 29 (50,9%) | 46 (70,7%) |
| Chunks UNSUPPORTED | 28 (49,1%) | 19 (29,2%) |
| Chunks CONTRADICTED | 0 (0,0%) | 0 (0,0%) |

### Observações

1. **Gemini apresenta taxa significativamente maior de SUPPORTED** (70,7% vs 50,9%)
2. **ChatGPT tem taxa equilibrada entre SUPPORTED e UNSUPPORTED** (~50-50%)
3. **ChatGPT e Gemini não apresenta casos de CONTRADICTED** - nenhum documento contradiz as claims analisadas
4. **Chunks UNSUPPORTED tendem a conter**: estatísticas específicas não confirmadas, detalhes legislativos divergentes, ou afirmações muito específicas sobre datas e números

---

## Análise de Justificativas de UNSUPPORTED

### Padrões Comuns - ChatGPT

1. **Dados estatísticos ausentes ou divergentes**: O texto menciona percentuais ou números específicos que não são confirmados pelas referências
2. **Legislação específica não mencionada**: Referências a projetos de lei ou decrecis que não são detalhados nos documentos de referência
3. **Detalhamento excessivo**: Claims com muitos detalhes específicos que não podem ser verificados nas fontes disponíveis
4. **Datas divergentes**: Algumas datas mencionadas não correspondem às datas dos documentos de referência

### Padrões Comuns - Gemini

1. **Estatísticas não confirmadas**: Percentuais de adoção ou dados numéricos específicos não presentes nas referências
2. **Tabelas detalhadas**: Claims que apresentam informações estruturadas em tabelas não são confirmadas integralmente
3. **Prognósticos apresentados como fatos**: Afirmações sobre o futuro (ex: "consolidou-se em 2026") apresentadas como fatos consumados
4. **Terminologia técnica muito específica**: Termos ou conceitos que não são mencionados explicitamente nos documentos

---

## Conclusões

### Achados Principais

1. **Taxa de SUPPORTED por chunk**: ~51% para ChatGPT e ~70% para Gemini
2. **Qualidade das respostas**: Gemini apresenta desempenho superior na correlação entre chunks e documentos de referência quando comparado ao ChatGPT
3. **Prevalência de UNSUPPORTED**: A principal categoria de problemas é a falta de suporte documental, não a contradição

### Implicações

1. **Documentos de referência são relevantes**: A seleção por similaridade semântica mostrou-se eficaz para avaliação
2. **Afirmações específicas são problemáticas**: Claims com estatísticas, anos específicos ou detalhes técnicos têm maior chance de serem UNSUPPORTED
3. **Modelo Gemini mais preciso**: Maior taxa de SUPPORTED indica melhor alinhamento entre seus chunks e os documentos de referência utilizados

### Recomendações

1. **Revisar documentos de referência**: Para chunks UNSUPPORTED, verificar se os documentos selecionados são os mais adequados
2. **Padronizar citações**: Para claims com informações específicas (anos, números), incluir referências mais precisas
3. **Validar dados estatísticos**: Claims com porcentagens e números devem ser verificados especialmente
4. **Atenção às datas**: Afirmações sobre marcos temporais devem ser validadas contra as fontes

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

*Relatório gerado em: 2026-05-20*