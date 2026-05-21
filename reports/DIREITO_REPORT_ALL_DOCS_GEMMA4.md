# Relatório de Auditoria: Análise de Chunks vs Resultados - Direito e Tecnologia

## Visão Geral

Este relatório analisa a correspondência entre os **chunks originais** (presentes em `./chunks/`) e os **resultados da auditoria** (presentes em `results/audit_with_chunks_direito/`). O objetivo é identificar, para cada chunk original, se possui referência válida (SUPPORTED) ou não (UNSUPPORTED) nos documentos de referência.

---

## Resumo dos Dados

### Arquivos Analisados

| Fonte | Arquivo de Chunks | Chunks Originais |
|-------|-------------------|------------------|
| ChatGPT | `ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_references.json` | 57 |
| Gemini | `Gemini_Direito_e_Tecnologia_Regulação_e_IA_references.json` | 65 |

### Resultados da Auditoria

| Modelo | Arquivo SUPPORTED | Arquivo UNSUPPORTED | Total Avaliações |
|--------|-------------------|---------------------|------------------|
| ChatGPT | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | `answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 456 |
| Gemini | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json` | `answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json` | 3.442 |

**Nota**: As métricas finais são calculadas com base na quantidade de chunks originais (57 para ChatGPT, 65 para Gemini), não na quantidade de avaliações realizadas.

---

## Análise por Chunk Original

### Metodologia

Cada chunk foi avaliado contra **múltiplos documentos de referência**. Um chunk é considerado:
- **SUPPORTED**: Se pelo menos um documento de referência confirmou a claim
- **UNSUPPORTED**: Se nenhum documento confirmou a claim (pelo menos uma avaliação sem suporte)

Esta análise permite identificar quais chunks possuem referências válidas nos documentos e quais carecem de suporte documental.

---

### ChatGPT - Resultados Consolidados

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais** | 57 |
| **Chunks com SUPPORTED** (≥1 referência válida) | 24 (42,1%) |
| **Chunks com UNSUPPORTED** (sem referência válida) | 33 (57,9%) |

#### Distribuição por Categoria

```
SUPPORTED   ████████████████████████████                 24 (42.1%)
UNSUPPORTED ███████████████████████████████            33 (57.9%)
```

#### Análise Detalhada

O modelo ChatGPT apresenta **42,1%** dos chunks com pelo menos uma referência válida (SUPPORTED), enquanto **57,9%** não possuem nenhum suporte documental.

**Observações:**
- 24 chunks (42,1%) possuem pelo menos uma referência válida confirmando a claim
- 33 chunks (57,9%) não possuem nenhuma referência válida nos documentos de referência
- A taxa de 42,1% indica que menos da metade dos chunks possuem suporte documental adequado

#### Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 0 | "Principais temas tecnológicos e regulatórios - Inteligência Artificial (IA)..." |
| 3 | "Além do PL 2.338/2023 (Marco Legal da IA) em trâmite..." |
| 5 | "Projetos de lei em andamento: O destaque é o PL 2.338/2023..." |
| 6 | "Prevê diretrizes de proteção de direitos fundamentais... e cria o SIA..." |
| 8 | "Delineando governança centralizada, soberania tecnológica..." |
| 9 | "Prevendo a cooperação público-privada, capacitação de profissionais..." |
| 19 | "Bloqueio de conteúdo nocivo: sistemas que impeçam acesso de menores..." |
| 21 | "Vedação de perfilamento publicitário e análise emocional..." |
| 28 | "ix) Remoção de conteúdo impróprio mediante notificação..." |

#### Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 1 | "Histórico e regulação: Internacionalmente, surgiu agenda de propor IA ética..." | Princípios da OCDE/UNESCO não mencionados nos documentos |
| 2 | "Nos EUA não há uma lei federal uniforme de IA até 2026..." | Não menciona legislação federal dos EUA, FTC, NIST |
| 7 | "Há também projetos setoriais e regionais..." | Casos reais e precedentes específicos não encontrados |
| 10 | "Nível internacional, há leis como o regulamento europeu NIS2..." | NIS2, Cibersecurity Act não mencionados nos documentos |
| 11 | "Lei Carolina Dieckmann (Lei 12.737/2012) – crimes informáticos..." | Lei específica não encontrada nos documentos |
| 12 | "Em 2023 a Estratégia previa obrigatoriedade de adesão..." | PL 4752/25 e estratégias específicas não mencionadas |

---

### Gemini - Resultados Consolidados

| Indicador | Valor |
|-----------|-------|
| **Total de Chunks Originais** | 65 |
| **Chunks com SUPPORTED** (≥1 referência válida) | 46 (70,8%) |
| **Chunks com UNSUPPORTED** (sem referência válida) | 19 (29,2%) |

#### Distribuição por Categoria

```
SUPPORTED   ████████████████████████████████████████████████ 46 (70.8%)
UNSUPPORTED ██████████████████████                        19 (29.2%)
```

#### Análise Detalhada

O modelo Gemini apresenta uma taxa significativamente maior de chunks com suporte válido: **70,8%** (46 de 65 chunks), enquanto **29,2%** (19 chunks) não possuem nenhum suporte. Isso representa uma taxa aproximadamente **28 pontos percentuais superior** ao ChatGPT.

**Observações:**
- 46 chunks (70,8%) possuem pelo menos uma referência válida confirmando a claim
- 19 chunks (29,2%) não possuem nenhuma referência válida nos documentos de referência
- A taxa de SUPPORTED (70,8%) sugere que os chunks do Gemini estão mais alinhados com as fontes documentais disponíveis

#### Exemplos de Chunks SUPPORTED

| Chunk # | Trecho do Texto |
|---------|-----------------|
| 0 | "Tratado sobre Regulação Digital, Inteligência Artificial..." |
| 1 | "Inteligência Artificial: Fundamentos Técnicos, Jurídicos e a Revolução Generativa..." |
| 2 | "No contexto jurídico de 2026, a distinção reside entre IA fraca e IA generativa..." |
| 3 | "Historicamente, a regulação da IA evoluiu de diretrizes éticas voluntárias..." |
| 4 | "Juridicamente, a IA é tratada não apenas como um software..." |
| 7 | "Contudo, essa adoção acelerada expõe o profissional a riscos éticos e regulatórios..." |
| 8 | "Governança de IA e o Futuro Marco Legal no Brasil (PL nº 2338/2023)..." |
| 9 | "No centro dessa discussão está o Projeto de Lei nº 2338/2023..." |
| 17 | "O marco europeu foca na segurança do produto e na conformidade técnica..." |
| 24 | "A Autoridade Nacional de Proteção de Dados (ANPD) amadureceu sua atuação..." |
| 35 | "A Jurisprudência do STJ sobre IA Generativa como Prova..." |

#### Exemplos de Chunks UNSUPPORTED

| Chunk # | Trecho do Texto | Justificativa |
|---------|-----------------|----------------|
| 5 | "Tabela 1: Fundamentos Técnicos e Jurídicos da Inteligência Artificial..." | Dado estatístico de 55% de advogados não fundamentado |
| 10 | "O projeto, aprovado pelo Senado no final de 2024..." | Datas específicas e classificação de risco não encontradas |
| 13 | "A governança proposta pelo projeto brasileiro é inovadora..." | Obrigações específicas para setor público não detalhadas |
| 14 | "Análise Comparativa Internacional: Brasil, União Europeia e Estados Unidos..." | Detalhes comparativos específicos não encontrados |
| 20 | "Em 2025, o lançamento do Painel de Fiscalização..." | Dado específico não mencionado nos documentos |
| 21 | "A Lei nº 15.211/2025 obriga as empresas de tecnologia..." | Detalhes específicos da lei não encontrados |

---

## Comparação entre Modelos

| Métrica | ChatGPT | Gemini |
|---------|---------|--------|
| Total de chunks originais | 57 | 65 |
| Chunks SUPPORTED | 24 (42,1%) | 46 (70,8%) |
| Chunks UNSUPPORTED | 33 (57,9%) | 19 (29,2%) |

### Observações

1. **Gemini apresenta taxa significativamente maior de SUPPORTED** (70,8% vs 42,1%)
2. **A diferença é de aproximadamente 28 pontos percentuais**, indicando que o Gemini possui chunks mais alinhados com as fontes documentais
3. **ChatGPT tem mais da metade dos chunks sem suporte** (57,9%), enquanto Gemini tem menos de um terço (29,2%)

---

## Análise de Justificativas de UNSUPPORTED

### Padrões Comuns - ChatGPT

1. **Especificações regulatórias ausentes**: Referências a legislações específicas (FTC, NIST, NIS2, HIPAA) não encontradas nos documentos
2. **Dados estatísticos não mencionados**: Números específicos sobre uso de IA não corroborados
3. **Projetos de lei não referenciados**: PLs específicos (PL 4.752/2025, PL 4752/25) não mencionados
4. **Casos concretos não documentados**: Precedentes jurídicos específicos não encontrados

### Padrões Comuns - Gemini

1. **Dados quantitativos não sustentados**: Estatísticas específicas (ex: 55% dos advogados) sem fundamento documental
2. **Estruturas conceituais não descritas**: Modelos de três camadas ou分类ções específicas não encontradas
3. **Legislação específica não mencionada**: Detalhes de projetos de lei (datas, classificações) ausentes
4. **Obrigações setoriais não detalhadas**: Regras específicas para setores não encontradas

---

## Análise por Documento de Referência

### ChatGPT - Documentos de Referência

Os documentos de referência utilizados para avaliação do ChatGPT incluem o PL 2338/2023 e documentos da ANPD. Cada chunk foi avaliado contra múltiplos documentos para determinar se possui suporte válido.

### Gemini - Documentos de Referência

Os documentos de referência utilizados para avaliação do Gemini incluem:
- Guia Completo da LGPD para Advogados em 2026
- Comparação entre EU AI Act e legislação americana
- Documentos sobre Inteligência Artificial (Sinapses)
- Guia para Advogado em 2026

Cada chunk foi avaliado contra múltiplos documentos para determinar se possui suporte válido.

---

## Conclusões

### Achados Principais

1. **Taxa de SUPPORTED por chunk**: ChatGPT (42,1%) vs Gemini (70,8%)
2. **Taxa de UNSUPPORTED**: ChatGPT (57,9%) vs Gemini (29,2%)
3. **Qualidade dos documentos de referência**: Documentos específicos sobre IA jurídica oferecem maior suporte
4. **Modelo Gemini** apresenta resultado significativamente superior em termos de taxa de SUPPORTED por chunk

### Implicações

1. **Chunks do Gemini estão mais alinhados** com as fontes documentais disponíveis no domínio de Direito e Tecnologia
2. **Claims com informações específicas** (datas, números, referências legislativas específicas) têm maior chance de serem UNSUPPORTED
3. **Afirmações generalizadas sobre conceitos** (definições de IA, governança, princípios) tendem a ter mais suporte

### Recomendações

1. **Revisar documentos de referência**: Adicionar documentos que abordem especificamente os temas que estão gerando UNSUPPORTED
2. **Padronizar citações**: Para claims com informações específicas (anos, números de projetos de lei), incluir referências mais precisas
3. **Aumentar diversidade de fontes**: Adicionar mais documentos de referência pode melhorar a taxa de suporte geral

---

## Arquivos Analisados

### Chunks Originais
- `chunks/ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_references.json`
- `chunks/Gemini_Direito_e_Tecnologia_Regulação_e_IA_references.json`

### Resultados da Auditoria
- `results/audit_with_chunks_direito/answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json`
- `results/audit_with_chunks_direito/answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json`
- `results/audit_with_chunks_direito/answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_SUPPORTED.json`
- `results/audit_with_chunks_direito/answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA_UNSUPPORTED.json`

---

*Relatório gerado em: 2026-05-17*