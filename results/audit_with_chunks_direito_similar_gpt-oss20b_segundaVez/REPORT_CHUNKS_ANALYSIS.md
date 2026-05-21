# Relatório de Auditoria: Análise de Chunks com Referências (Direito Similar - GPT-OSS 20B Segunda Rodada)

## Visão Geral

Este relatório apresenta a análise dos chunks que possuem referências corretas em documentos (SUPPORTED) e chunks sem suporte (UNSUPPORTED) para o domínio de Direito e Tecnologia. O objetivo é identificar, para cada chunk original, se possui referência válida (SUPPORTED), não possui suporte (UNSUPPORTED) ou é contradito (CONTRADICTED) pelos documentos de referência.

---

## Resumo dos Dados

### Arquivos de Chunks Originais Analisados

| Fonte | Arquivo de Chunks | Total de Chunks |
|-------|-------------------|-----------------|
| ChatGPT | `ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_references.json` | 57 |
| Gemini | `Gemini_Direito_e_Tecnologia_Regulação_e_IA_references.json` | 65 |

### Resultados da Auditoria

| Modelo | Arquivo de Resultado | Total de Avaliações |
|--------|---------------------|---------------------|
| ChatGPT | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA.json` | 57 |
| Gemini | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA.json` | 65 |

---

## Análise por Fonte de Chunks

### Metodologia

A avaliação foi realizada através de:
1. **Associação de chunks**: Cada chunk original foi associado a um documento de referência específico
2. **Avaliação de claims**: Cada claim do chunk foi verificada contra o documento de referência
3. **Classificação**:
   - **SUPPORTED**: O documento de referência confirma a claim
   - **UNSUPPORTED**: O documento não confirma a claim
   - **CONTRADICTED**: O documento contradiz a claim

---

## ChatGPT - Análise Detalhada

### Quantidade de Chunks Originais

**Total de chunks originais no arquivo**: 57 chunks

### Total de Avaliações Realizadas

| Categoria | Quantidade |
|-----------|------------|
| SUPPORTED | 22 |
| UNSUPPORTED | 35 |
| CONTRADICTED | 0 |
| **Total** | **57** |

### Métricas de Avaliação

| Indicador | Valor |
|-----------|-------|
| **Chunks com SUPPORTED** | 22 (38,6%) |
| **Chunks com UNSUPPORTED** | 35 (61,4%) |
| **Chunks com CONTRADICTED** | 0 (0,0%) |

### Distribuição Visual

```
SUPPORTED   ████████████████████████████████████████████████████████              22 (38.6%)
UNSUPPORTED ██████████████████████████████████████████████████████████████████████ 35 (61.4%)
```

### Exemplos de Chunks SUPPORTED (ChatGPT)

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 3 | "além do **PL 2.338/2023** (Marco Legal da IA) em trâmite. - **Legislação:** O Brasil não tem ainda lei esgotando o tema..." |
| 4 | "Assim, métodos de IA já devem obedecer à LGPD e suas sanções. Em execução, agências reguladoras brasileiras estão atentas: a ANPD incluiu IA e tecnologias emergentes..." |
| 6 | "Prevê diretrizes de proteção de direitos fundamentais... e cria o **Sistema Nacional de Regulação e Governança de IA (SIA)**..." |
| 8 | "delineando governança centralizada, soberania tecnológica e proteção de infraestruturas críticas. Em trâmite... **PL 4.752/2025**..." |
| 9 | "prevendo a cooperação público-privada, capacitação de profissionais, criação de um programa nacional de segurança digital..." |

### Exemplos de Chunks UNSUPPORTED (ChatGPT)

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 0 | "Inteligência Artificial (IA) - Conceitos: IA engloba sistemas de aprendizado de máquina..." | Evidência não menciona explicitamente IA generativa, classificação por risco, nem os princípios de transparência e auditabilidade dos algoritmos |
| 1 | "Internacionalmente, surgiu agenda de propor 'IA ética' com princípios da OCDE e UNESCO..." | Evidência confirma AI Act e classificação por risco, mas não fornece informação sobre agenda de IA ética baseada em OCDE e UNESCO |
| 2 | "Nos EUA não há uma lei federal uniforme de IA até 2026..." | Evidência confirma ausência de lei federal, mas não menciona executive orders, diretrizes FTC/NIST, nem regulamentações de New York |
| 5 | "O destaque é o **PL 2.338/2023**... Prevê classificação de sistemas por risco... e proíbe IA de risco 'excessivo'" | Evidência confirma aprovação e classificação de risco, mas não afirma que o PL proíbe sistemas de risco excessivo |
| 7 | "Há também projetos setoriais e regionais... No Brasil, foi noticiado bloqueio de funcionalidades de apps..." | Evidência não contém referência à Estratégia Nacional de Cibersegurança (E-Ciber) ou Decreto 12.573/2025 |

### Padrões de Falha - ChatGPT

1. **Dados estatísticos ausentes ou divergentes**: O texto menciona percentuais ou números específicos que não são confirmados pelas referências
2. **Legislação específica não mencionada**: Referências a projetos de lei ou decretos que não são detalhados nos documentos de referência
3. **Detalhamento excessivo**: Claims com muitos detalhes específicos que não podem ser verificados nas fontes disponíveis

---

## Gemini - Análise Detalhada

### Quantidade de Chunks Originais

**Total de chunks originais no arquivo**: 65 chunks

### Total de Avaliações Realizadas

| Categoria | Quantidade |
|-----------|------------|
| SUPPORTED | 39 |
| UNSUPPORTED | 26 |
| CONTRADICTED | 0 |
| **Total** | **65** |

### Métricas de Avaliação

| Indicador | Valor |
|-----------|-------|
| **Chunks com SUPPORTED** | 39 (60,0%) |
| **Chunks com UNSUPPORTED** | 26 (40,0%) |
| **Chunks com CONTRADICTED** | 0 (0,0%) |

### Distribuição Visual

```
SUPPORTED   ████████████████████████████████████████████████████████████████████████████████████ 39 (60.0%)
UNSUPPORTED ████████████████████████████████████████████████████████████████                    26 (40.0%)
```

### Exemplos de Chunks SUPPORTED (Gemini)

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 1 | "Inteligência Artificial... refere-se a sistemas computacionais capazes de simular comportamentos inteligentes humanos..." |
| 2 | "No contexto jurídico de 2026, a distinção fundamental reside entre a chamada 'IA fraca'... e a 'IA generativa'..." |
| 7 | "Contudo, essa adoção acelerada expõe o profissional a riscos éticos e regulatórios, como a violação do sigilo profissional..." |
| 9 | "No centro dessa discussão está o Projeto de Lei nº 2338/2023, que busca criar normas gerais para o desenvolvimento e uso ético da IA no Brasil" |
| 16 | "Análise Comparativa Internacional: Brasil, União Europeia e Estados Unidos... O _AI Act_ da União Europeia... estabeleceu o padrão global de conformidade..." |

### Exemplos de Chunks UNSUPPORTED (Gemini)

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 4 | "Juridicamente, a IA é tratada não apenas como um software, mas como um agente de processamento que deve observar a boa-fé..." | Evidência discute supervisão, transparência e auditabilidade, mas não afirma explicitamente que a IA é tratada como agente de processamento que deve observar boa-fé |
| 5 | "Tabela 1: Fundamentos Técnicos e Jurídicos... O impacto da IA generativa... cerca de 55%..." | Evidência discute uso de IA generativa no Judiciário com percentuais de tribunais, não de advogados (55%) |
| 6 | "A tecnologia atua em três camadas estratégicas: automatização... aumento da capacidade analítica (revisão de contratos em segundos)..." | Evidência confirma automação e análise preditiva, mas não valida a divisão em três camadas nem o exemplo específico de 'revisão de contratos em segundos' |
| 8 | "A governança da IA consolidou-se como o elemento estrutural das decisões corporativas e públicas em 2026" | Evidência não confirma que a governança de IA se consolidou como elemento estrutural em 2026, nem que deixou de ser tratada como formalidade |
| 13 | "A governança proposta pelo projeto brasileiro é inovadora ao detalhar obrigações específicas para o setor público, como a promoção da cultura e língua portuguesa..." | Evidência não contém afirmação explícita ou inferível sobre obrigações específicas para o setor público envolvendo promoção cultural ou linguística |

### Padrões de Falha - Gemini

1. **Estatísticas não confirmadas**: Percentuais de adoção ou dados numéricos específicos não presentes nas referências
2. **Terminologia técnica muito específica**: Termos ou conceitos que não são mencionados explicitamente nos documentos
3. **Prognósticos apresentados como fatos**: Afirmações sobre o futuro (ex.: "consolidou-se em 2026") apresentadas como fatos consumados
4. **Atribuição de obrigações não verificada**: Afirmações sobre obrigações específicas de leis ou projetos não confirmadas nas referências

---

## Comparação entre Modelos

| Métrica | ChatGPT | Gemini |
|---------|---------|--------|
| Total de chunks originais | 57 | 65 |
| Chunks Avaliados | 57 | 65 |
| Chunks SUPPORTED | 22 (38,6%) | 39 (60,0%) |
| Chunks UNSUPPORTED | 35 (61,4%) | 26 (40,0%) |
| Chunks CONTRADICTED | 0 (0,0%) | 0 (0,0%) |

### Observações

1. **Gemini apresenta taxa significativamente maior de SUPPORTED** (60,0% vs 38,6% do ChatGPT)
2. **ChatGPT tem maioria de UNSUPPORTED** (61,4%), indicando que a maioria de seus chunks não foi confirmada pelas referências
3. **Nenhum chunk CONTRADICTED** nesta rodada, diferentemente da rodada anterior (gpt-oss20b) onde Gemini teve 2 chunks contraditos
4. **Em comparação com a rodada anterior (gpt-oss20b)**: ChatGPT caiu de 45,6% SUPPORTED para 38,6%; Gemini subiu de 58,5% para 60,0%, indicando leve melhora na correlação dos chunks do Gemini

---

## Análise de Qualidade das Referências

### Tipos de Claims com Maior Taxa de Sucesso

1. **Legislação existente**: Referências a leis e regulamentações confirmadas (LGPD, PL 2338, ECA Digital)
2. **Conceitos técnicos**: Definições de IA, machine learning, IA generativa
3. **Instituições**: ANPD, CNJ, STF, STJ, OAB

### Tipos de Claims com Maior Taxa de Falha

1. **Estatísticas específicas**: Percentuais de adoção, números de ataques, dados de mercado
2. **Prognósticos temporais**: Afirmações sobre consolidação em anos específicos
3. **Detalhamento excessivo**: Múltiplas afirmações específicas em um único chunk
4. **Atribuição de características não documentadas**: Funcionalidades ou classificações não presentes nas referências

---

## Conclusões

### Achados Principais

1. **Taxa de SUPPORTED por chunk**: ~39% para ChatGPT e ~60% para Gemini
2. **Qualidade das respostas**: Gemini apresenta desempenho superior na correlação entre chunks e documentos de referência
3. **Sem casos de contradição**: Nenhum chunk foi classificado como CONTRADICTED, indicando que os documentos de referência não contradizem diretamente as claims nesta rodada
4. **ChatGPT mais afetado**: Apresenta maior taxa de UNSUPPORTED, sugerindo maior propensão a fazer afirmações não verificáveis pelos documentos de referência

### Implicações

1. **Documentos de referência são relevantes**: A associação entre chunk e documento mostrou-se eficaz para avaliação
2. **Afirmações específicas são problemáticas**: Claims com estatísticas, anos específicos ou detalhes técnicos têm maior chance de serem UNSUPPORTED
3. **Modelo Gemini mais preciso**: Maior taxa de SUPPORTED indica melhor alinhamento entre seus chunks e os documentos de referência utilizados

### Recomendações

1. **Revisar documentos de referência**: Para chunks UNSUPPORTED, verificar se os documentos selecionados são os mais adequados
2. **Padronizar citações**: Para claims com informações específicas (anos, números), incluir referências mais precisas
3. **Avaliar similaridade semântica**: Considerar usar similaridade semântica para melhorar a correspondência entre chunks e documentos

---

## Arquivos Analisados

### Chunks Originais
- `chunks/direito/ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_references.json` (57 chunks)
- `chunks/direito/Gemini_Direito_e_Tecnologia_Regulação_e_IA_references.json` (65 chunks)

### Resultados da Auditoria
- `results/audit_with_chunks_direito_similar_gpt-oss20b_segundaVez/answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA.json` (57 total: 22 SUPPORTED + 35 UNSUPPORTED)
- `results/audit_with_chunks_direito_similar_gpt-oss20b_segundaVez/answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA.json` (65 total: 39 SUPPORTED + 26 UNSUPPORTED)

---

*Relatório gerado em: 2026-05-20*
