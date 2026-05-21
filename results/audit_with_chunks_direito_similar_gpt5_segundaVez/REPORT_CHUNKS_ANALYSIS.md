# Relatório de Auditoria: Análise de Chunks vs Resultados (Direito Similar - GPT-5 Segunda Rodada)

## Visão Geral

Este relatório analisa a correspondência entre os **chunks originais** (presentes em `./chunks/direito/`) e os **resultados da auditoria** (presentes em `results/audit_with_chunks_direito_similar_gpt5_segundaVez/`). O objetivo é identificar, para cada chunk original, se possui referência válida (SUPPORTED) ou não (UNSUPPORTED).

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
| ChatGPT | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | 17 |
| ChatGPT | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 40 |
| Gemini | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | 26 |
| Gemini | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 39 |

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
| **Chunks com SUPPORTED** | 17 (29,8%) |
| **Chunks com UNSUPPORTED** | 40 (70,2%) |
| **Chunks com CONTRADICTED** | 0 (0,0%) |

### ChatGPT - Distribuição por Chunk

```
SUPPORTED   ████████████████████████████                              17 (29.8%)
UNSUPPORTED ████████████████████████████████████████████████████████  40 (70.2%)
```

### ChatGPT - Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 3 | "além do **PL 2.338/2023** (Marco Legal da IA) em trâmite. - **Legislação:** O Brasil não tem ainda lei esgotando o tema..." |
| 4 | "Assim, métodos de IA já devem obedecer à LGPD e suas sanções. Em execução, agências reguladoras brasileiras estão atentas..." |
| 6 | "Prevê diretrizes de proteção de direitos fundamentais... e cria o **Sistema Nacional de Regulação e Governança de IA (SIA)**..." |
| 8 | "delineando governança centralizada, soberania tecnológica e proteção de infraestruturas críticas. Em trâmite... **PL 4.752/2025**..." |
| 16 | "ii) **Controle parental** : ferramentas de supervisionamento (limite de tempo, controle de contatos, geolocalização)..." |

### ChatGPT - Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 0 | "Principais temas tecnológicos e regulatórios - Inteligência Artificial (IA) - Conceitos: IA engloba sistemas de aprendizado de máquina (ML)..." | Evidência não menciona explicitamente 'aprendizado profundo', 'visão computacional' ou 'auditabilidade' dos algoritmos, nem a afirmação normativa de que no Direito é importante entender esses princípios |
| 1 | "Histórico e regulação: Internacionalmente, surgiu agenda de propor 'IA ética' com princípios da OCDE e UNESCO..." | Evidência confirma AI Act e classificação por risco, mas não há menção à UNESCO |
| 2 | "Nos EUA não há uma lei federal uniforme de IA até 2026, mas o governo editou executive orders... no Brasil... decreto do Poder Executivo (CITDigital)..." | Evidência confirma ausência de lei federal nos EUA e regulação estadual, mas não menciona o decreto CITDigital |
| 5 | "Projetos de lei em andamento: O destaque é o **PL 2.338/2023**... Prevê classificação de sistemas por risco... e proíbe IA de risco 'excessivo'" | Evidências confirmam aprovação no Senado e classificação de risco, mas não confirmam que o PL proíbe explicitamente IA de risco excessivo |
| 7 | "Há também projetos setoriais e regionais... No Brasil, foi noticiado bloqueio de funcionalidades de apps..." | Evidências confirmam regulação geral e uso de IA no Judiciário, mas o bloqueio específico de funcionalidades de apps não é mencionado |

---

### Gemini - Resultados Consolidados

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais Usados** | 65 |
| **Chunks com SUPPORTED** | 26 (40,0%) |
| **Chunks com UNSUPPORTED** | 39 (60,0%) |
| **Chunks com CONTRADICTED** | 0 (0,0%) |

### Gemini - Distribuição por Chunk

```
SUPPORTED   ██████████████████████████████████████████████              26 (40.0%)
UNSUPPORTED ██████████████████████████████████████████████████████████  39 (60.0%)
```

### Gemini - Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 0 | "Tratado sobre Regulação Digital, Inteligência Artificial e o Ecossistema Jurídico Contemporâneo: Um Guia para o Biênio 2025-2026..." |
| 1 | "Inteligência Artificial: Fundamentos Técnicos, Jurídicos e a Revolução Generativa... sistemas computacionais capazes de simular comportamentos inteligentes..." |
| 9 | "No centro dessa discussão está o Projeto de Lei nº 2338/2023, que busca criar normas gerais para o desenvolvimento e uso ético da IA no Brasil" |
| 10 | "O projeto, aprovado pelo Senado no final de 2024 e em debate na Câmara dos Deputados em 2025-2026, adota uma abordagem baseada em riscos..." |
| 16 | "Análise Comparativa Internacional: Brasil, União Europeia e Estados Unidos... O AI Act da União Europeia... estabeleceu o padrão global de conformidade..." |

### Gemini - Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 2 | "No contexto jurídico de 2026, a distinção fundamental reside entre a chamada 'IA fraca'... voltada para tarefas específicas... e a 'IA generativa'..." | Evidência não menciona triagem processual como exemplo de IA fraca, nem associa "identificar padrões decisórios" à IA generativa de forma explícita |
| 5 | "Tabela 1: Fundamentos Técnicos e Jurídicos da Inteligência Artificial... O impacto da IA generativa no setor jurídico brasileiro em 2026 é quantificável: cerca de 55%..." | Evidência cobre LLMs e RAG, mas fornece 45,8% de adoção por tribunais (CNJ), não 55% dos advogados |
| 6 | "A tecnologia atua em três camadas estratégicas: automatização... aumento da capacidade analítica (revisão de contratos em segundos)..." | Evidência confirma automação e análise preditiva, mas não há menção à 'revisão de contratos em segundos' |
| 7 | "Contudo, essa adoção acelerada expõe o profissional a riscos éticos e regulatórios, como a violação do sigilo profissional..." | Evidência confirma riscos éticos e violação de sigilo, mas não menciona sanções pela OAB em casos de jurisprudências alucinadas |
| 8 | "Governança de IA e o Futuro Marco Legal no Brasil (PL nº 2338/2023)... A governança da IA consolidou-se como elemento estrutural... em 2026" | Evidências mostram tramitação com impasses e incertezas em 2026, e descrevem deveres de governança como propostas, não como realidade consolidada |

---

## Comparação entre Modelos

| Métrica | ChatGPT | Gemini |
|---------|---------|--------|
| Total de chunks originais usados | 57 | 65 |
| Chunks SUPPORTED | 17 (29,8%) | 26 (40,0%) |
| Chunks UNSUPPORTED | 40 (70,2%) | 39 (60,0%) |
| Chunks CONTRADICTED | 0 (0,0%) | 0 (0,0%) |

### Observações

1. **Gemini apresenta taxa maior de SUPPORTED** (40,0% vs 29,8% do ChatGPT)
2. **ChatGPT tem maioria de UNSUPPORTED** (~70%), indicando que a maioria de seus chunks não foi confirmada pelas referências nesta rodada
3. **Ambos os modelos não apresentam casos de CONTRADICTED** - nenhum documento contradiz as claims analisadas
4. **Em comparação com a rodada anterior (gpt5)**: ChatGPT caiu de 50,9% SUPPORTED para 29,8%; Gemini caiu de 70,3% para 40,0%, indicando critério mais rigoroso nesta segunda rodada de avaliação

---

## Análise de Justificativas de UNSUPPORTED

### Padrões Comuns - ChatGPT

1. **Ausência de menção explícita**: A referência cobre o tema geral, mas não menciona detalhes ou exemplos específicos presentes no chunk
2. **Detalhamento excessivo**: Claims que contêm múltiplas afirmações específicas (conceitos técnicos, nomes de decretos) não são totalmente sustentadas
3. **Afirmações normativas não respaldadas**: O texto do chunk apresenta posições ou interpretações que vão além do que as referências documentam

### Padrões Comuns - Gemini

1. **Estatísticas não confirmadas**: Percentuais de adoção ou dados numéricos específicos não presentes nas referências
2. **Terminologia não verificada**: Termos ou conceitos que não são mencionados explicitamente nos documentos de referência
3. **Afirmações temporais não sustentadas**: Declarações sobre consolidação de fatos em 2026 que não são corroboradas pelas referências
4. **Atribuição de função incorreta**: Funcionalidades ou características associadas a tipos errados de IA (ex.: associar triagem processual à IA fraca)

---

## Conclusões

### Achados Principais

1. **Taxa de SUPPORTED por chunk**: ~30% para ChatGPT e ~40% para Gemini
2. **Queda em relação à rodada anterior**: Ambos os modelos apresentam redução significativa na taxa de SUPPORTED, sugerindo que a segunda rodada de avaliação (gpt5_segundaVez) aplicou critérios mais rigorosos
3. **Gemini mantém vantagem relativa**: Apesar da queda, Gemini ainda apresenta taxa de SUPPORTED superior ao ChatGPT (40% vs 30%)
4. **Sem casos de contradição**: Nenhum chunk foi classificado como CONTRADICTED, indicando que os documentos de referência não contradizem diretamente as claims

### Implicações

1. **Critério mais rigoroso**: A segunda rodada resultou em mais chunks classificados como UNSUPPORTED, indicando possível ajuste no limiar de aceitação do modelo avaliador
2. **Afirmações específicas continuam problemáticas**: Claims com estatísticas, terminologia técnica ou marcos temporais têm maior chance de serem UNSUPPORTED
3. **Modelo Gemini mais alinhado**: Embora ambos tenham caído, Gemini mantém correlação mais forte entre chunks e documentos de referência

### Recomendações

1. **Revisar critérios de avaliação**: Investigar se o rigor adicional desta rodada reflete melhoria na detecção ou alteração no comportamento do modelo julgador
2. **Padronizar citações**: Para claims com informações específicas (anos, números, decretos), incluir referências mais precisas
3. **Avaliar similaridade semântica**: Considerar usar similaridade semântica para melhorar a correspondência entre chunks e documentos

---

## Arquivos Analisados

### Chunks Originais
- `chunks/direito/ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_references.json`
- `chunks/direito/Gemini_Direito_e_Tecnologia_Regulação_e_IA_references.json`

### Resultados da Auditoria
- `results/audit_with_chunks_direito_similar_gpt5_segundaVez/answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA.json`
- `results/audit_with_chunks_direito_similar_gpt5_segundaVez/answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA.json`
- `results/audit_with_chunks_direito_similar_gpt5_segundaVez/answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json`
- `results/audit_with_chunks_direito_similar_gpt5_segundaVez/answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json`
- `results/audit_with_chunks_direito_similar_gpt5_segundaVez/answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json`
- `results/audit_with_chunks_direito_similar_gpt5_segundaVez/answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json`
- `reports/DIREITO_REPORT_CHUNKS_REFERENCES_SEARCH_SIMILAR.md`

---

*Relatório gerado em: 2026-05-20*
