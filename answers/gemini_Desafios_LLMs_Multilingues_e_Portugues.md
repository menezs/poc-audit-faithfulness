## **Desafios Técnicos e Éticos no Desenvolvimento de Modelos de Linguagem Multilíngues: Uma Análise do Ecossistema da Língua Portuguesa e Avanços Recentes** 

O desenvolvimento de modelos de linguagem de grande escala (LLMs) representa um dos marcos mais significativos da computação contemporânea, alterando fundamentalmente a forma como a informação é processada, gerada e consumida globalmente. No entanto, o progresso desta tecnologia não tem sido distribuído de forma equânime entre as diversas línguas e culturas do planeta. Embora o português seja uma das línguas mais faladas no mundo, com centenas de milhões de falantes em múltiplos continentes, ele é frequentemente categorizado em contextos de pesquisa como uma língua de recursos médios ou mesmo baixos, dependendo da especificidade do domínio técnico ou da variante regional considerada.[1] Esta análise técnica e ética explora as complexidades inerentes ao desenvolvimento de LLMs multilíngues, com foco no idioma português, examinando os avanços ocorridos entre 2021 e 2026, as metodologias propostas na literatura acadêmica, os benchmarks de avaliação e as lacunas que ainda impedem a paridade plena com os modelos anglocêntricos. 

## **A Arquitetura da Multilinguidade e os Desafios de Representação** 

A base técnica da maioria dos LLMs modernos reside na arquitetura Transformer, que revolucionou o processamento de linguagem natural ao permitir o processamento paralelo e a captura de dependências de longo alcance em textos.[3] No entanto, a aplicação desta arquitetura em contextos multilíngues introduz o que a literatura define como a maldição da multilinguidade. Este fenômeno descreve a relação inversa onde a adição de novos idiomas ao regime de treinamento melhora o desempenho em línguas de baixos recursos até um ponto crítico, após o qual a performance geral em benchmarks monolíngues e cross-linguais começa a declinar devido à competição por capacidade nos parâmetros do modelo.[4 ] 

Para um modelo de tamanho fixo, cada novo idioma adicionado exige uma parcela do "orçamento" de parâmetros, forçando uma compressão das representações semânticas. Em idiomas como o português, esta compressão pode levar à perda de nuances sintáticas e morfológicas, especialmente quando o modelo tenta alinhar o espaço vetorial da língua-alvo com o do inglês, que frequentemente atua como uma língua pivô implícita.[5] O alinhamento semântico cross-lingual é, portanto, tanto um objetivo desejado quanto uma fonte de tensão 

técnica. 

## **Tokenização e Ineficiência Estrutural** 

Um dos desafios técnicos mais persistentes e menos discutidos na superfície é a tokenização. O processo de converter texto em unidades discretas (tokens) é frequentemente otimizado para o inglês nos modelos globais. Em idiomas com morfologia rica, como o português, os tokenizers treinados predominantemente em dados ingleses tendem a fragmentar excessivamente as palavras.[7 ] 

Esta fragmentação excessiva gera uma cascata de ineficiências: 

1. **Redução da Janela de Contexto:** Como o português requer mais tokens para expressar a mesma ideia que o inglês, a capacidade efetiva de processamento de documentos longos é significativamente menor.[8 ] 

2. **Aumento de Custos:** Em modelos disponibilizados via API, o custo operacional para usuários lusófonos é proporcionalmente superior ao de usuários anglófonos para a execução da mesma tarefa.[7 ] 

3. **Degradação Semântica:** Fragmentos muito pequenos (sub-tokens) podem não carregar significado semântico suficiente, dificultando o aprendizado de representações robustas pelo modelo.[8 ] 

A literatura recente propôs o conceito de transferência de tokenizer como uma solução paliativa. Métodos como o _Orthogonal Mapping Pursuit_ (OMP) e o _Fast Vocabulary Transfer_ (FVT) permitem que um modelo pré-treinado seja adaptado a um novo tokenizer otimizado para a língua-alvo, reduzindo a fragmentação sem a necessidade de um retreinamento completo, o que seria proibitivo em termos de custo e impacto ambiental.[7 ] 

|**Métrica de Tokenização**|**Modelo Anglocêntrico**<br>**(Genérico)**|**Modelo Especializado**<br>**(Sabiá/GlórIA)**|
|---|---|---|
|Razão Token/Palavra (PT)|~1.5 - 1.8|~1.1 - 1.3|
|Eficiência de Contexto|Menor (Fragmentação alta)|Maior (Preservação de<br>morfemas)|
|Custo de Inferência (API)|Superior por unidade de<br>sentido|Otimizado para o idioma|



A especialização de domínio, como demonstrada pela família de modelos Sabiá, exemplifica como o treinamento centrado em corpora brasileiros pode mitigar essas perdas, oferecendo uma performance competitiva a um custo por token significativamente inferior ao de modelos de fronteira como o GPT-4o ou Claude 3.5 Sonnet.[9 ] 

## **O Paradigma dos Dados: Escassez, Qualidade e Sintetização** 

O desenvolvimento de LLMs é alimentado por dados, e a disponibilidade de corpora de alta qualidade em português é um gargalo técnico central. Embora o volume bruto de dados em português na web seja vasto, a qualidade necessária para o treinamento de modelos de instrução e raciocínio é escassa. Auditorias recentes em conjuntos de dados globais revelam que a representação da América Latina nos dados de treinamento é inferior a 0.5% do total, contrastando fortemente com a dominância da América do Norte e Europa Ocidental.[10 ] 

## **Curadoria de Corpora Nativos: O Caso ClassiCC-PT e GigaVerbo** 

Para enfrentar a baixa representação, pesquisadores brasileiros e portugueses têm desenvolvido corpora curados. O ClassiCC-PT, por exemplo, é um corpus de aproximadamente 126 bilhões de tokens, derivado de crawls do Common Crawl, mas submetido a rigorosos processos de filtragem e classificação de qualidade.[11] A metodologia envolve a aplicação de classificadores para identificar conteúdos educacionais e científicos (STEM), garantindo que o modelo aprenda padrões de linguagem sofisticados em vez de apenas reproduzir o ruído inerente à web.[11 ] 

O projeto GigaVerbo seguiu uma trajetória semelhante, consolidando diversos datasets abertos e aplicando pipelines de limpeza que utilizam bibliotecas como Trafilatura para extração e FastText para identificação de idioma, com limiares de confiança ajustados para evitar a contaminação por línguas similares, como o espanhol ou o galego.[12 ] 

## **Dados Sintéticos como Multiplicadores de Qualidade** 

Dada a dificuldade de obter dados humanos em escala para domínios específicos, a geração de dados sintéticos tornou-se uma ferramenta indispensável. Entre 2024 e 2026, a técnica de reescrita de documentos emergiu como uma abordagem promissora. Ao utilizar um modelo "professor" (geralmente um LLM de grande porte) para transformar textos ruidosos da web em formatos estruturados, como diálogos, resumos ou explicações acadêmicas, pesquisadores conseguiram criar versões sintéticas de alta qualidade que servem para o pré-treinamento e ajuste fino de modelos menores.[13 ] 

**Fonte de Dados Volume (Tokens) Natureza Uso Principal** 

|ClassiCC-PT|126B|Web Curada|Pré-treinamento base|
|---|---|---|---|
|GigaVerbo|~6B<br>(Sintético/Curado)|Multidomínio|Continuidade de treino|
|AnonyMed-BR|Especializado|Médico<br>(Real/Sintético)|Ajuste fino de domínio|
|MATH-PT|1.7K questões|Acadêmico (Nativo)|Avaliação de raciocínio|



Entretanto, o uso excessivo de dados sintéticos introduz riscos de colapso de distribuição, onde o modelo perde a capacidade de representar a variabilidade natural da linguagem humana, tornando-se um eco de suas próprias produções ou das de outros modelos.[14] O desafio técnico atual reside em equilibrar a diversidade de fontes para mitigar este efeito, garantindo que o modelo mantenha a robustez contra exemplos adversariais e evite a "auto-preferência".[14 ] 

## **Avaliação de Desempenho e Benchmarks Lusófonos** 

A eficácia de um LLM para o português não pode ser medida apenas por traduções de benchmarks ingleses como o MMLU. Traduções automáticas frequentemente introduzem vieses e obscurecem fenômenos linguísticos específicos, como jogos de palavras, rimas ou expressões idiomáticas.[16] Por isso, o desenvolvimento de benchmarks nativos tem sido uma prioridade nos últimos cinco anos. 

## **Benchmarks Profissionais e Acadêmicos no Brasil** 

O cenário brasileiro é rico em avaliações que testam não apenas a fluência linguística, mas o conhecimento institucional e legal. O OAB-Bench, que utiliza questões reais do Exame de Ordem da OAB, exige que o modelo redija peças processuais e responda a questões discursivas complexas.[17] Os resultados deste benchmark mostram que modelos de especialização nacional, como o Sabiá-4, conseguem atingir taxas de aprovação de 100% em exames estruturados, superando modelos globais em termos de eficiência de custo por ponto de acerto.[17 ] 

Outro marco é o Revalida, exame para médicos estrangeiros, onde LLMs de ponta têm demonstrado performance superior à média humana. Modelos como o GPT-4o e Claude Opus alcançaram acurácias superiores a 83%, enquanto modelos de tamanho médio como o Mixtral 

8x7B atingiram cerca de 63.7%.[18] Estes dados indicam que a capacidade de raciocínio clínico em português já está em um estágio avançado, embora a aplicação prática ainda esbarre em barreiras éticas de transparência e justificativa de decisões.[19 ] 

## **A Questão da Variedade: PT-BR vs. PT-PT** 

A disparidade entre as variedades brasileira e europeia do português representa um desafio técnico e cultural único. Como a maioria dos dados de treinamento provém do Brasil, modelos multilíngues tendem a produzir saídas com léxico e sintaxe brasileiros mesmo quando solicitados a escrever em português de Portugal.[16 ] 

O benchmark ALBA foi especificamente desenhado para avaliar a proficiência em português europeu (pt-PT) através de oito dimensões linguísticas, incluindo semântica cultural e análise de discurso.[16] Experimentos revelam que modelos genéricos frequentemente falham em distinguir regionalismos e provérbios intrínsecos à cultura portuguesa, evidenciando a necessidade de modelos como o GlórIA ou AMALIA, que utilizam corpora especificamente curados de Portugal para garantir a autenticidade linguística.[16 ] 

## **Desafios Éticos e Sociais: Vieses e Alinhamento de Valores** 

Os desafios éticos no desenvolvimento de LLMs são tão profundos quanto os técnicos, estendendo-se às dimensões sociais, culturais e legais. Um dos pontos mais críticos é a representação de valores humanos diversos em modelos que são, em sua essência, produtos de uma cultura específica. 

## **O Viés WEIRD e a Hegemonia Cultural** 

A literatura aponta consistentemente que os LLMs atuais refletem a psicologia de sociedades WEIRD ( _Western, Educated, Industrialized, Rich, and Democratic_ ). Ao comparar estimativas de valores morais geradas por modelos com dados de pesquisas reais de dezenas de países, observa-se que os LLMs tendem a superestimar as preocupações morais ocidentais e subestimar os valores de nações não-ocidentais, como a Nigéria ou a Indonésia.[2 ] 

No Brasil, este viés manifesta-se em julgamentos morais que podem não estar alinhados com o contexto socioeconômico local. Por exemplo, em tarefas de moderação de conteúdo, um modelo treinado com valores anglo-americanos pode falhar em identificar sutilezas de discursos de ódio específicos do contexto brasileiro ou, inversamente, censurar expressões culturais legítimas por não compreender seu uso in-group.[5] A "maldição da nocividade" é um risco real: modelos são mais propensos a gerar respostas prejudiciais quando provocados em idiomas com menos recursos de segurança alinhados, como variantes dialetais ou gírias regionais.[1 ] 

## **Transparência, Explicabilidade e o Efeito "Black-Box"** 

Especialmente em setores críticos como a medicina, a natureza de "caixa-preta" dos LLMs é uma barreira fundamental. Médicos e profissionais de saúde exigem justificativas para decisões diagnósticas, algo que modelos que operam puramente por probabilidade estatística têm dificuldade em fornecer de forma confiável.[3] O risco de "alucinações" — declarações médicas plausíveis, mas incorretas — exige que sistemas implementados em hospitais tenham camadas de supervisão humana rigorosa e mecanismos de _chain-of-thought_ para extrair gráficos de conhecimento que possam ser auditados.[3 ] 

Além disso, a responsabilidade legal por decisões assistidas por IA que levam a desfechos adversos permanece um vácuo regulatório no Brasil. A fragmentação da governança e a despragmatização do julgamento clínico são riscos apontados pela literatura, onde o uso excessivo de IA pode levar a uma atrofia das competências humanas críticas.[26 ] 

## **Privacidade e Segurança: A Proteção de Dados Sensíveis** 

A conformidade com a Lei Geral de Proteção de Dados (LGPD) é um requisito inegociável para a implantação de LLMs em território nacional. No entanto, LLMs são vulneráveis a ataques de extração de dados, onde adversários podem recuperar informações pessoalmente identificáveis (PII) presentes no corpus de treinamento através de prompts criativos.[19 ] 

## **Anonimização e Aprendizado Federado** 

Para mitigar esses riscos, abordagens como o aprendizado federado e a privacidade diferencial têm sido propostas. O aprendizado federado permite o treinamento de modelos em dados distribuídos (por exemplo, em diferentes hospitais) sem a necessidade de centralizar informações sensíveis, preservando a soberania dos dados locais.[27 ] 

No campo da anonimização de registros médicos brasileiros, o desenvolvimento de sistemas como o AnonyMed-BR mostrou que é possível utilizar modelos de linguagem para identificar e mascarar PII com alta precisão (F1 > 0.90), combinando estratégias extrativas (baseadas em NER) e gerativas (reescrita de texto).[28] A capacidade de anonimizar dados em larga escala abre portas para a criação de novos datasets de pesquisa sem violar a privacidade dos pacientes, embora o risco de reidentificação por padrões ocultos em grandes massas de dados de saúde ainda exija monitoramento constante.[19 ] 

## **Avanços Recentes e Metodologias de Fronteira (2021-2026)** 

Nos últimos cinco anos, o campo testemunhou uma transição de modelos genéricos para sistemas altamente especializados e eficientes. A técnica de Fine-Tuning Parametricamente Eficiente (PEFT) democratizou o acesso à tecnologia para pesquisadores com recursos limitados. 

## **Eficiência com LoRA e DoRA em Português** 

Estudos sistemáticos realizados com o modelo BERTimbau (referência para o português brasileiro) demonstraram que a técnica LoRA pode atingir cerca de 95.8% do desempenho de um ajuste completo, consumindo uma fração do tempo e da memória de GPU.[29] Um achado significativo é a sensibilidade à taxa de aprendizado: para o português, taxas de aprendizado mais elevadas do que as convencionalmente usadas para o inglês produziram ganhos de performance de até 19 pontos percentuais em tarefas de compreensão de texto.[30 ] 

|**Configuração**<br>**de Treino**|**Modelo Base**|**Tempo de**<br>**Treino**<br>**(Relativo)**|**Memória GPU**<br>**(Pico)**|**F1 Score (QA)**|
|---|---|---|---|---|
|Full Fine-Tuning|BERTimbau-Lar<br>ge|100%|>18 GB|84.86|
|LoRA|BERTimbau-Lar<br>ge|26.5%|<10 GB|81.32|
|QLoRA|BERTimbau-Lar<br>ge|18%|<6 GB|76.50|
|Sabiá-4<br>(Proprietário)|Sabiá-4|N/A|N/A|Alta<br>(Especializado)|



Além da eficiência, a resiliência à quantização foi identificada como uma propriedade de modelos maiores. Modelos com centenas de milhões ou bilhões de parâmetros sofrem menos degradação ao serem comprimidos para precisão de 4 bits em comparação com modelos pequenos, o que é crucial para a implantação em dispositivos de borda ou servidores de custo reduzido.[31 ] 

## **O Surgimento de Agentes e Long Context** 

Modelos recentes como o Sabiá-4 e Sabiazinho-4 introduziram capacidades agentic, permitindo que a IA navegue na web, utilize ferramentas externas e siga instruções em múltiplos turnos de conversa.[32] A expansão da janela de contexto para 128k ou 256k tokens resolve parcialmente a 

ineficiência da tokenização mencionada anteriormente, permitindo que o modelo "leia" processos judiciais completos ou livros inteiros antes de gerar uma resposta.[32 ] 

## **Lacunas Persistentes e Direções Futuras** 

Apesar do otimismo tecnológico, várias lacunas permanecem sem solução definitiva no ecossistema lusófono. 

## **1. Multimodalidade e Representação Visual** 

A maioria dos benchmarks e datasets de alta qualidade em português ainda é exclusivamente textual. Existe uma carência crítica de modelos e conjuntos de dados que integrem imagem, áudio e vídeo no contexto cultural brasileiro.[10] Por exemplo, exames radiológicos acompanhados de laudos em português ou debates políticos transcritos e anotados são recursos escassos que limitam o desenvolvimento de IAs multimodais soberanas.[36 ] 

## **2. Soberania Digital e Dependência de Infraestrutura** 

O Brasil e outros países de língua portuguesa ainda dependem fortemente de infraestruturas de processamento e modelos de base desenvolvidos por um pequeno número de empresas globais. O debate sobre soberania digital, promovido em fóruns como o MWC 2026 e a SBPC, destaca que sem o controle sobre os dados, os algoritmos e a capacidade de processamento, as nações correm o risco de se tornarem meras consumidoras de uma inteligência que não reflete seus interesses estratégicos.[38 ] 

## **3. Ética em Tempo Real e Moderação Contextual** 

A velocidade com que a desinformação multimodal se espalha exige ferramentas de detecção que operem em tempo real e compreendam o contexto cultural volátil do Brasil. O paradoxo da IA generativa sugere que, à medida que os modelos melhoram na criação de conteúdo, a detecção técnica torna-se cada vez mais difícil, exigindo abordagens que combinem tecnologia com educação midiática e "teoria da inoculação".[35 ] 

## **4. Generalização vs. Especialização** 

Há uma tensão contínua entre criar modelos multilíngues massivos que "sabem tudo" e modelos monolíngues especializados que são mais precisos em domínios locais. A tendência para 2026 parece ser o uso de modelos de base globais adaptados via _continued pre-training_ e PEFT, mas a busca por um modelo puramente nativo que rivalize com os gigantes globais em todas as tarefas ainda enfrenta o desafio do custo computacional e da disponibilidade de dados de raciocínio de alta densidade.[32 ] 

## **Considerações Finais** 

O desenvolvimento de LLMs para a língua portuguesa é um campo em ebulição, marcado por uma transição necessária de uma postura reativa para uma liderança proativa na criação de 

recursos linguísticos. Os desafios técnicos de tokenização e escassez de dados estão sendo enfrentados com inovações em transferência de aprendizado e dados sintéticos, enquanto o desenvolvimento de benchmarks nativos como o OAB-Bench e o ALBA garante uma avaliação mais justa e representativa da competência dos modelos. 

No entanto, o progresso técnico deve ser acompanhado por uma vigilância ética rigorosa. O combate ao viés WEIRD, a proteção da privacidade em setores sensíveis e a busca pela soberania digital são imperativos que determinarão se os LLMs servirão como ferramentas de emancipação tecnológica ou como novos vetores de desigualdade cultural. A colaboração entre academia, indústria e governo é fundamental para garantir que o português, em todas as suas variedades, ocupe um lugar de destaque na inteligência artificial do futuro, preservando sua riqueza, sua história e seus valores fundamentais. 

## **Referências citadas** 

1. Multilingual LLMs: Progress, Challenges, and Future Directions - Prem AI, acessado em maio 4, 2026, https://blog.premai.io/multilingual-llms-progress-challenges-and-future-directions/ 

2. Exploring Cultural Variations in Moral Judgments with Large Language Models - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2506.12433v1 

3. A systematic review of ethical considerations of large language models in healthcare and medicine - Frontiers, acessado em maio 4, 2026, - 

https://www.frontiersin.org/journals/digital health/articles/10.3389/fdgth.2025.1653 631/full 

4. Multilingual Large Language Models and Curse of Multilinguality - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2406.10602v1 

5. A Survey on Multilingual Large Language Models: Corpora, Alignment, and Bias - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2404.00929v1 

6. Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Mo - ACL Anthology, - 

acessado em maio 4, 2026, https://aclanthology.org/2025.acl long.118.pdf 

7. Efficient Low-Resource Language Models Using ... - ACL Anthology, acessado em - 

maio 4, 2026, https://aclanthology.org/2026.eacl srw.49.pdf 

8. Trans-Tokenization and Cross-lingual Vocabulary Transfers: Language Adaptation of LLMs for Low-Resource NLP - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2408.04303v1 

9. Sabiá-3 Technical Report - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2410.12049v3 

10. BRoverbs - Measuring how much LLMs understand Portuguese proverbs - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2509.08960v1 

11. Building High-Quality Datasets for Portuguese LLMs: From Common Crawl Snapshots to Industrial-Grade Corpora - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2509.08824v1 

12. Better Open Source LLMs for Portuguese - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2603.03543v1 

13. Synthetic Rewriting as a Quality Multiplier: Evidence from Portuguese Continued Pretraining, acessado em maio 4, 2026, https://arxiv.org/html/2603.24826v1 

14. The Impact of Synthetic Data Diversity on LLM Fine-TuningAccepted to Findings of the Association for Computational Linguistics (ACL 2026). - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2511.01490v3 

15. Synthetic Data Generation Methods for LLMs: A Comprehensive Guide - Towards AI, acessado em maio 4, 2026, https://towardsai.net/p/machine-learning/synthetic-data-generation-methods-for-ll ms-a-comprehensive-guide 

16. ALBA: A European Portuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative LLMs - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2603.26516v1 

17. maritaca-ai/oab-bench - GitHub, acessado em maio 4, 2026, https://github.com/maritaca-ai/oab-bench 

18. Benchmarking open-source large language models on Portuguese Revalida multiple-choice questions - PMC, acessado em maio 4, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12082654/ 

19. Large Language Models: A Structured Taxonomy and Review of Challenges, Limitations, Solutions, and Future Directions - MDPI, acessado em maio 4, 2026, https://www.mdpi.com/2076-3417/15/14/8103 

20. GlórIA: A Generative and Open Large Language Model for Portuguese Pre-print - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2402.12969v1 

21. ALBA: A European Portuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative LLMs - ResearchGate, acessado em maio 4, 2026, 

https://www.researchgate.net/publication/403262919_ALBA_A_European_Portug uese_Benchmark_for_Evaluating_Language_and_Linguistic_Dimensions_in_Gen erative_LLMs 

22. AMALIA Technical Report: A Fully Open Source Large Language Model for European Portuguese - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2603.26511v1 

23. Moral stereotyping in large language models - PNAS, acessado em maio 4, 2026, https://www.pnas.org/doi/10.1073/pnas.2519941123 

24. Understanding and Addressing Bias in Content Moderation - Musubi Labs, acessado em maio 4, 2026, https://www.musubilabs.ai/post/understanding-and-addressing-bias-in-content-mo deration 

25. Content Moderation Ethics: Navigating Bias, Censorship & Fairness - GetStream.io, acessado em maio 4, 2026, 

https://getstream.io/blog/content-moderation-ethics/ 

26. Editorial: Ethical considerations of large language models: challenges and best practices, acessado em maio 4, 2026, - 

https://www.frontiersin.org/journals/digital health/articles/10.3389/fdgth.2026.1807 664/full 

27. LLM-Assisted Scoping Review of Artificial Intelligence in Brazilian Public Health: 

Lessons from Transfer and Federated Learning for Resource-Constrained Settings - PMC, acessado em maio 4, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC12840889/ 

28. Guardians of the data: NER and LLMs for effective medical record anonymization in Brazilian Portuguese - Frontiers, acessado em maio 4, 2026, - 

https://www.frontiersin.org/journals/public health/articles/10.3389/fpubh.2025.1717 303/full 

29. [2603.21418] Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs - arXiv, acessado em maio 4, 2026, https://arxiv.org/abs/2603.21418 

30. Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs - ACL Anthology, acessado em maio 4, 2026, 

   - 

   - https://aclanthology.org/2026.propor-1.91.pdf

31. Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs - ACL Anthology, acessado em maio 4, 2026, 

https://aclanthology.org/2026.propor-1.91/ 

32. Sabiá-4 Technical Report - arXiv, acessado em maio 4, 2026, https://arxiv.org/pdf/2603.10213.pdf

33. Sabiá-4 Technical Report - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2603.10213v1 

34. [2603.10213] Sabiá-4 Technical Report - arXiv, acessado em maio 4, 2026, https://arxiv.org/abs/2603.10213 

35. Industrialized Deception: The Collateral Effects of LLM-Generated Misinformation on Digital Ecosystems - arXiv, acessado em maio 4, 2026, https://arxiv.org/html/2601.21963v1 

36. Pirá: A Bilingual Portuguese-English Dataset for Question-Answering about the Ocean - USP, acessado em maio 4, 2026, http://sites.poli.usp.br/p/fabio.cozman/Publications/Article/paschoal-pirozelli-freiredelgado-peres-jose-nakasato-oliveira-brandao-costa-cozman-cikm2021.pdf 

37. Response accuracy of GPT-4 across languages: insights from an expert-level diagnostic radiology examination in Japan - PMC, acessado em maio 4, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC11790683/ 

38. Consolidação da IA impulsiona debate sobre soberania digital - Meio e Mensagem, acessado em maio 4, 2026, https://www.meioemensagem.com.br/mwc/consolidacao-da-ia-impulsiona-debatesobre-soberania-digital 

39. Em evento do MGI, especialistas apontam soberania digital como desafio da transição digital no Brasil - Portal Gov.br, acessado em maio 4, 2026, https://www.gov.br/gestao/pt-br/assuntos/noticias/2026/abril/em-evento-do-mgi-es pecialistas-apontam-soberania-digital-como-desafio-da-transicao-digital-no-brasil 

40. Vozes da Ciência – O Brasil que queremos: Inteligência Artificial e Soberania Digital, acessado em maio 4, 2026, 

https://www.youtube.com/watch?v=H1DNDa9HVa8 

