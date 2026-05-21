# Relatório de Auditoria: Análise de Chunks com Referências (Direito Similar)

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
| ChatGPT | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | 26 |
| ChatGPT | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 31 |
| Gemini | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | 37 |
| Gemini | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 26 |

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
| SUPPORTED | 26 |
| UNSUPPORTED | 31 |
| **Total** | **57** |

### Métricas de Avaliação

| Indicador | Valor |
|-----------|-------|
| **Chunks com SUPPORTED** | 26 (45,6%) |
| **Chunks com UNSUPPORTED** | 31 (54,4%) |
| **Chunks com CONTRADICTED** | 0 (0,0%) |

### Distribuição Visual

```
SUPPORTED   ██████████████████████████████████████████████████████ 26 (45.6%)
UNSUPPORTED ███████████████████████████████████████████████████████████ 31 (54.4%)
```

### Exemplos de Chunks SUPPORTED (ChatGPT)

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 3 | "Além do **PL 2.338/2023** (Marco Legal da IA) em trâmite. - **Legislação:** O Brasil não tem ainda lei esgotando o tema..." |
| 4 | "Assim, métodos de IA já devem obedecer à LGPD e suas sanções. Em execução, agências reguladoras brasileiras estão atentas: a ANPD incluiu IA e tecnologias emergentes..." |
| 6 | "Prevê diretrizes de proteção de direitos fundamentais... e cria o **Sistema Nacional de Regulação e Governança de IA (SIA)**..." |
| 8 | "Delineando governança centralizada, soberania tecnológica e proteção de infraestruturas críticas. Em trâmite... **PL 4.752/2025**..." |
| 9 | "Prevendo a cooperação público-privada, capacitação de profissionais, criação de um programa nacional de segurança digital..." |

### Exemplos de Chunks UNSUPPORTED (ChatGPT)

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 0 | "Inteligência Artificial (IA) - Conceitos: IA engloba sistemas de aprendizado de máquina..." | A evidência não menciona os princípios de transparência e auditabilidade dos algoritmos |
| 1 | "Internacionalmente, surgiu agenda de propor 'IA ética' com princípios da OCDE e UNESCO..." | A evidência menciona OCDE, mas não UNESCO como parte da agenda global de IA ética |
| 2 | "Nos EUA não há uma lei federal uniforme de IA até 2026..." | O decreto CITDigital não é mencionado nas referências |
| 5 | "O destaque é o **PL 2.338/2023**... Prevê classificação de sistemas por risco... e proíbe IA de risco 'excessivo'" | A evidência não menciona explicitamente a proibição de IA de risco excessivo |
| 10 | "Nível internacional, há leis como o regulamento europeu NIS2..." | A evidência não menciona NIS2, Cibersecurity Act, HIPAA ou GLBA |

### Padrões de Falha - ChatGPT

1. **Dados estatísticos ausentes ou divergentes**: O texto menciona percentuais ou números específicos que não são confirmados pelas referências
2. **Legislação específica não mencionada**: Referências a projetos de lei ou decreys que não são detalhados nos documentos de referência
3. **Detalhamento excessivo**: Claims com muitos detalhes específicos que não podem ser verificados nas fontes disponíveis

---

## Gemini - Análise Detalhada

### Quantidade de Chunks Originais

**Total de chunks originais no arquivo**: 65 chunks

### Total de Avaliações Realizadas

| Categoria | Quantidade |
|-----------|------------|
| SUPPORTED | 38 |
| UNSUPPORTED | 27 |
| **Total** | **65** |

### Métricas de Avaliação

| Indicador | Valor |
|-----------|-------|
| **Chunks com SUPPORTED** | 38 (58,5%) |
| **Chunks com UNSUPPORTED** | 27 (41,5%) |

### Distribuição Visual

```
SUPPORTED   ██████████████████████████████████████████████████████████████████████████ 38 (58.5%)
UNSUPPORTED ███████████████████████████████████████████████████████████                27 (41.5%)
```

### Exemplos de Chunks SUPPORTED (Gemini)

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 1 | "Inteligência Artificial... refere-se a sistemas computacionais capazes de simular comportamentos inteligentes humanos..." |
| 2 | "No contexto jurídico de 2026, a distinção fundamental reside entre a chamada 'IA fraca'... e a 'IA generativa'..." |
| 4 | "Juridicamente, a IA é tratada não apenas como um software, mas como um agente de processamento..." |
| 7 | "Contudo, essa adoção acelerada expõe o profissional a riscos éticos e regulatórios..." |
| 9 | "No centro dessa discussão está o Projeto de Lei nº 2338/2023, que busca criar normas gerais..." |

### Exemplos de Chunks UNSUPPORTED (Gemini)

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 0 | "Tratado sobre Regulação Digital... O cenário jurídico brasileiro e internacional no biênio 2025-2026..." | A evidência confirma LGPD, ECA Digital e PL 2338, mas não afirma que tecnologia se tornou infraestrutura central |
| 3 | "Historicamente, a regulação da IA evoluiu de diretrizes éticas voluntárias para normas mandatórias..." | A evidência não detalha a evolução histórica específica mencionada |
| 5 | "Tabela 1: Fundamentos Técnicos e Jurídicos... O impacto da IA generativa... cerca de 55%..." | A evidência fornece dados sobre uso em tribunais (45,8%), não sobre advogados |
| 6 | "A tecnologia atua em três camadas estratégicas..." | O exemplo específico de revisão de contratos em segundos não é mencionado |
| 8 | "A governança da Inteligência Artificial consolidou-se como elemento estrutural em 2026" | O material não confirma esse marco temporal específico |

### Exemplos de Chunks CONTRADICTED (Gemini)

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 18 | "Em contraste, os Estados Unidos, sob a administração federal iniciada em 2025, adotaram uma postura de desregulamentação... Através de Ordens Executivas (como a de dezembro de 2025), o governo americano buscou centralizar a regulação a nível federal para eliminar o 'mosaico' de leis estaduais (como as de Colorado, Califórnia e Texas)..." | A evidência states que the December 2025 executive order preempts state AI laws but does not eliminate them; state statutes (Colorado, California, Texas) remain operative and DOJ is challenging them, contradicting the claim that the mosaic of state laws was eliminated. |
| 48 | "Estima-se que 40% da população ocupada no Brasil possua alta exposição à IA, mas com baixa complementaridade, tornando esses trabalhadores vulneráveis à perda de emprego por substituição" | A evidência reports that only approximately 20% of the employed population in Brazil has high exposure and low complementarity to AI, whereas the claim asserts 40%. The evidence directly conflicts with the claim. |

### Padrões de Falha - Gemini

1. **Estatísticas não confirmadas**: Percentuais de adoção ou dados numéricos específicos não presentes nas referências
2. **Terminologia técnica muito específica**: Termos ou conceitos que não são mencionados explicitamente nos documentos
3. **Prognósticos apresentados como fatos**: Afirmações sobre o futuro apresentadas como fatos consumados

---

## Comparação entre Modelos

| Métrica | ChatGPT | Gemini |
|---------|---------|--------|
| Total de chunks originais | 57 | 65 |
| Chunks Avaliados | 57 | 65 |
| Chunks SUPPORTED | 26 (45,6%) | 38 (58,5%) |
| Chunks UNSUPPORTED | 31 (54,4%) | 27 (41,5%) |
| Chunks CONTRADICTED | 0 (0,0%) | 0 (0,0%) |

### Observações Importantes

1. **Gemini apresenta taxa maior de SUPPORTED** (58,5% vs 45,6%)
2. **ChatGPT tem taxa maior de UNSUPPORTED** (54,4% vs 41,5%)
3. **Chunks UNSUPPORTED tendem a conter**: estatísticas específicas, detalhes legislativos não mencionados, ou afirmações muito específicas sobre datas e números

---

## Análise de Qualidade das Referências

### Tipos de Claims com Maior Taxa de Sucesso

1. **Legislação existente**: Referências a leis e regulamentações confirmadas (LGPD, ECA Digital, PL 2338)
2. **Conceitos técnicos**: Definições de IA, machine learning, IA generativa
3. **Instituições**: ANPD, CNJ, STF, STJ

### Tipos de Claims com Maior Taxa de Falha

1. **Estatísticas específicas**: Percentuais de adoção, números de ataques, dados de mercado
2. **Prognósticos temporais**: Afirmações sobre consolidação em anos específicos
3. **Comparações internacionais detalhadas**: Detalhes específicos de legislações estrangeiras

### Tipos de Claims Contraditas

1. **Dados não existentes nos documentos**: Afirmações sobre funcionalidades ou ferramentas que não existem nos documentos de referência
2. **Incompatibilidade com fontes**: Informações que contradizem diretamente o conteúdo dos documentos

---

## Conclusões

### Achados Principais

1. **Taxa de SUPPORTED**: ~46% para ChatGPT e ~57% para Gemini
2. **Qualidade das respostas**: Gemini apresenta desempenho superior na correlação entre chunks e documentos de referência
3. **Casos de contradição**: Apenas o Gemini apresenta chunks CONTRADICTED (2 casos, 3,1%), indicando que alguns documentos de referência contradizem diretamente as claims
4. **ChatGPT mais conservador**: Apresenta maior taxa de UNSUPPORTED, sugerindo maior cautela ou maior propensão a fazer afirmações não verificáveis

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
- `results/audit_with_chunks_direito_similar/answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA.json` (57 total: 26 SUPPORTED + 31 UNSUPPORTED)
- `results/audit_with_chunks_direito_similar/answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA.json` (65 total: 37 SUPPORTED + 26 UNSUPPORTED + 2 CONTRADICTED)

---

*Relatório gerado em: 2026-05-20*