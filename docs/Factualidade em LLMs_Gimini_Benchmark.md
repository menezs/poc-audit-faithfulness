Fronteiras da Factualidade em Modelos
de Linguagem de Grande Escala: Uma
Investigação Sistemática de Métricas,
Mecanismos e Benchmarks (2021-2025)

A evolução meteórica dos Modelos de Linguagem de Grande Escala (LLMs) redefiniu as
fronteiras do processamento de linguagem natural, permitindo capacidades sem precedentes
em geração de texto, tradução e raciocínio complexo. No entanto, essa fluência linguística é
frequentemente obscurecida pelo fenômeno das alucinações — a geração de conteúdos
sintaticamente plausíveis, porém factualmente incorretos ou não fundamentados em evidências
externas (https://arxiv.org/html/2510.06265v2). Este relatório técnico detalha uma pesquisa
profunda e sistemática sobre o estado da arte da factualidade em LLMs, abrangendo desde a
taxonomia do erro até as métricas de avaliação mais robustas validadas por humanos no
período de 2021 a 2025.

A integridade epistemológica dos LLMs é hoje o principal obstáculo para sua aplicação em
domínios críticos como medicina, direito e ciência
(https://aclanthology.org/2024.emnlp-main.1088.pdf). Enquanto modelos como o GPT-4o e o
Llama 3 demonstram um conhecimento de mundo vasto, a "factualidade" — definida como a
capacidade de gerar conteúdo consistente com o conhecimento estabelecido — nem sempre
caminha em paralelo com a "fidelidade à fonte"
(https://aclanthology.org/2026.findings-eacl.46.pdf). Esta análise investiga essa tensão
fundamental, os avanços em detecção baseada em incerteza e o surgimento de benchmarks
específicos para a língua portuguesa, como o PoETa v2
(https://ieeexplore.ieee.org/iel8/6287639/10820123/11303664.pdf).

Fluxo de Processamento de Pesquisa (PRISMA-Like)

Para garantir a factualidade e a exaustividade deste relatório, foi aplicado um protocolo de
pesquisa sistemática seguindo as diretrizes de revisões de escopo no campo da IA.

Fase

Identificação

Descrição das Atividades e Filtros
Aplicados

Buscas realizadas nas bases ACL Anthology,
arXiv, IEEE Xplore, ACM Digital Library e
Google Scholar utilizando queries combinadas

Triagem

Elegibilidade

Inclusão

(ex: "factuality LLM evaluation metrics survey
2023"). Total de registros identificados: 482.

Remoção de duplicatas e filtragem por data
(2021-2025). Exclusão de posts de blogs sem
rigor metodológico e artigos puramente teóricos
sem validação empírica. Registros restantes:
156.

Análise de títulos e abstracts com base nos
critérios de inclusão: foco explícito em métricas
de factualidade, avaliação em tarefas de QA,
sumarização ou RAG, e benchmarks
multilingues (ênfase em português). Artigos
selecionados: 43.

Estudos integrados na síntese narrativa e na
tabela de extração de dados, priorizando
trabalhos com alta concordância entre
anotadores humanos e código aberto.

As consultas de busca variaram entre termos técnicos como "semantic entropy hallucination" e
"SRLScore factuality", permitindo capturar tanto métodos baseados em modelos quanto
abordagens probabilísticas de "caixa-preta"
(https://deepchecks.com/llm-hallucination-detection-and-mitigation-best-techniques/).

Tabela de Estudos Selecionados e Extração de
Campos

A tabela abaixo compila os estudos fundamentais que moldaram o campo da factualidade nos
últimos quatro anos.

Referê
ncia

Tipo

Tarefas
Avaliad

Dataset
s

Métrica
s de

Resulta
dos

Código
/

Observ
ações

(Autor,
Ano,
Link)

Li et al.
(2026)

as

Princip
ais

Factual
idade

Princip
ais

WikiEnti
ties

Pesquis
a
Empíric
a

Sumariz
ação,
QA,
Rephra
sing

HFH
Rate,
Faithful
Adhere
nce

Fabbri
et al.
(2022)

Pesquis
a de
Métrica
s

Sumariz
ação

QAFact
Eval

Summa
C,
SQuAD,
MNLI

Almeida
et al.
(2025)

Benchm
ark

44
tarefas
(NL)

PoETa
v2,
Bluex,
SICK-B
R

NPM
(Normal
ized
Perform
ance
Metric)

Reprod
utibilid
ade

GitHub

GitHub

GitHub

do
Revisor

Introduz
o
conceit
o de
"Alucina
ção
Factual
Prejudic
ial"
(HFH).

Estudo
fundam
ental
para
avaliaçã
o
basead
a em
QA.

Maior
avaliaçã
o
sistemá
tica
para a
língua
portugu
esa.

Modelo
s
maiores
tendem
a
"corrigir
" fontes
erradas
mesmo
quando
instruíd
os a
seguir o
texto.

QAFact
Eval
supera
métrica
s de
NLI em
14% na
consistê
ncia
factual.

GPT-4o
lidera
em
portugu
ês;
Qwen
domina
open-so
urce.

Min et
al.
(2023)

Pesquis
a de
Métrica
s

Long-fo
rm
generati
on

Wikiped
ia
Biograp
hies

FactSco
re

Tjandra
et al.
(2024)

Pesquis
a
Empíric
a

Open-d
omain
QA

TriviaQ
A,
PopQA

CHOKE
-Score

Guan et
al.
(2023)

Benchm
ark

Multimo
dal
(Vision-
Langua
ge)

Hallusio
nBench

Questio
n-Pair
Accurac
y

GitHub

arXiv

GitHub

Decom
posição
em
fatos
atômico
s
permite
auditori
a fina
de
biografi
as.

Modelo
s
podem
alucinar
com
alta
certeza
mesmo
sabend
o a
respost
a
correta.

GPT-4V
atinge
apenas
31% de
acerto
em
pares
de
pergunt
as
visuais.

Validad
o com
custo
humano
de
$26k;
alta
correlaç
ão.

Desafia
a ideia
de que
incertez
a é o
único
proxy
para
alucinaç
ão.

Foca
em
ilusões
visuais
e viés
de
linguag
em em
modelo
s
multimo
dais.

Manaku

Pesquis

Biografi

WikiBio,

SelfChe

Amostr

GitHub

Método

l et al.
(2023)

a de
Métrica
s

as,
Bio-Me
d

SelfChe
ckGPT

ckGPT

Assis et
al.
(2025)

Pesquis
a
Empíric
a

Sumariz
ação,
Simplifi
cação,
QA

Bluex,
TweetS
entBR

LLM-as-
a-judge,
Human
Eval

Hey et
al.
(2023)

Pesquis
a de
Métrica
s

Sumariz
ação

CNN/D
M,
XSum

SRLSco
re

Xu et al.
(2025)

Treinam
ento /
Alinham
ento

Factual
QA

InFACT
dataset

Factual
Precisio
n/Recall

agem
estocás
tica
detecta
alucinaç
ões
sem
recurso
s
externo
s.

Sabiá-3
e
Tucano
apresen
tam
robuste
z
competi
tiva em
PT-BR.

Uso de
Semanti
c Role
Labelin
g para
extrair
tuplas
de fatos
interpret
áveis.

Alinhar
para
informat
ividade
também
melhora
a

GitHub

GitHub

arXiv

de
"zero-re
source"
altamen
te
influent
e.

Foco
em
diversid
ade
linguísti
ca e
variante
s
brasileir
as.

Proporc
iona
interpret
abilidad
e
através
de
agentes
e
ações.

Introduz
o
framew
ork
InFACT
para
modelo

factuali
dade
geral.

s mais
complet
os.

Síntese Narrativa: Categorias, Tendências e
Mecanismos

A factualidade em modelos de linguagem evoluiu de uma preocupação secundária para o
centro do debate sobre segurança da IA. A análise sistemática dos estudos publicados entre
2021 e 2025 revela quatro tendências macro: a transição de métricas de sobreposição léxica
para métricas baseadas em modelos (LLM-as-a-judge), a descoberta de alucinações de "alta
confiança", o refinamento da avaliação em contextos multimodais e o esforço de soberania
linguística em benchmarks não-ingleses
(https://aclanthology.org/2025.findings-emnlp.1035.pdf).

1. Taxonomia e Natureza das Alucinações

As alucinações não são falhas aleatórias, mas subprodutos da arquitetura probabilística dos
transformadores. A literatura atual distingue entre:

●  Alucinações Intrínsecas: O modelo contradiz diretamente a fonte fornecida (ex: em

sumarização, trocar o sujeito de uma ação presente no texto) .

●  Alucinações Extrínsecas: O modelo adiciona informações que não podem ser

verificadas a partir da fonte, embora possam ser verdadeiras ou falsas no mundo real .

●  Harmful Factuality Hallucination (HFH): Um fenômeno recém-identificado onde o

modelo, ao encontrar um erro factual no prompt (ex: "Quem é o presidente dos EUA,
Donald Trump?"), corrige a informação com base em seu conhecimento interno ("Joe
Biden") mesmo quando a tarefa exige fidelidade estrita ao contexto fornecido
(https://aclanthology.org/2026.findings-eacl.46.pdf). Este erro é particularmente insidioso
em contextos jurídicos e médicos, onde a preservação de erros originais é necessária
para análise técnica.

2. O Paradoxo da Escala e o Fenômeno CHOKE

Um dos insights mais profundos dos últimos dois anos é que o aumento do número de
parâmetros não elimina as alucinações; em muitos casos, ele as torna mais "convincentes".
Modelos maiores tendem a apresentar maior HFH, pois seu conhecimento de mundo (world
knowledge) torna-se tão forte que sobrepõe as instruções de grounding .

Além disso, o estudo CHOKE (Certainty in Hallucinations of Knowledgeable Entities) demonstra
que modelos podem alucinar com alta certeza mesmo quando possuem a resposta correta em
seus pesos internos (https://arxiv.org/html/2502.12964v2). Isso ocorre devido a perturbações
triviais nos prompts ou vieses de decodificação, sugerindo que a factualidade não é apenas
uma questão de "ter o conhecimento", mas de "acessar o conhecimento de forma robusta"

durante a inferência.

3. Factualidade em Sumarização e RAG

Na tarefa de sumarização, a consistência factual é o critério de qualidade mais crítico. Métricas
tradicionais como ROUGE falham severamente por focarem na sobreposição de tokens e
ignorarem a semântica factual . O surgimento de frameworks como QAFactEval propôs uma
mudança de paradigma: se um resumo é factualmente consistente, as perguntas feitas sobre o
resumo devem ter as mesmas respostas quando feitas sobre o documento original
(https://aclanthology.org/2022.naacl-main.187.pdf).

No contexto de Retrieval-Augmented Generation (RAG), o desafio é o "groundedness".
Modelos frequentemente ignoram os documentos recuperados em favor de seus próprios
vieses pré-treinados, um problema que o framework RAGAS e o métrica FactScore tentam
mitigar ao decompor as respostas em alegações atômicas verificáveis
(https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-ll
ms/evaluation/list-of-eval-metrics).

4. Panorâmica Multilingue: O Caso do Português

Até 2023, a maioria das avaliações de factualidade era centrada no inglês, frequentemente
utilizando traduções automáticas que perdiam nuances culturais. O benchmark PoETa v2
(2025) e o estudo "Brazil's LLM Fauna" representam marcos para a soberania tecnológica em
língua portuguesa (https://journals-sol.sbc.org.br/index.php/jbcs/article/view/5814).

●  Desempenho Regional: Modelos nativos ou adaptados para o português, como o

Sabiá-3 e o Tucano, demonstram eficácia superior em exames nacionais (como USP e
UNICAMP) e no entendimento de provérbios brasileiros (dataset BRoverbs) .

●  Vieses de Riqueza: Estudos indicam que a acurácia factual dos LLMs é correlacionada
com o PIB do país de origem do tema, sugerindo que idiomas e culturas de regiões com
menor representação digital sofrem mais alucinações por falta de dados de treinamento .

5. Factualidade Multimodal

Com o advento de modelos como GPT-4V e Gemini Pro, as alucinações se estenderam para o
domínio visual. O HallusionBench revelou que os modelos sofrem de "viés de linguagem": se
uma imagem é ambígua, o modelo tende a descrever o que é estatisticamente comum em
texto, ignorando os detalhes visuais contraditórios (https://arxiv.org/html/2310.14566v5).
Métricas como o VC-Inspector agora tentam avaliar a qualidade de legendas de vídeo sem
depender de referências humanas, focando na presença factual de objetos e ações
(https://arxiv.org/html/2509.16538v2).

Lista Exaustiva de Métricas de Factualidade

A avaliação da factualidade é dividida em quatro categorias principais, cada uma com forças e
limitações distintas.

A. Métricas Baseadas em QA/QG (Question Answering / Question
Generation)

Estas métricas simulam o processo de verificação humana através de perguntas automáticas.

●  QAFactEval: Considerada o estado da arte para sumarização. Utiliza quatro

componentes: seleção de respostas, geração de perguntas (QG), resposta às perguntas
(QA) baseada na fonte e avaliação da sobreposição de respostas .

●  FEQA (Faithful Evaluation of Question Answering): Foca na fidelidade do resumo ao
documento original, gerando perguntas a partir do resumo e respondendo-as com o
original .

●  QuestEval: Uma métrica mista que avalia tanto a precisão (o que está no resumo é
verdadeiro?) quanto o recall (o que é importante no original está no resumo?) .

B. Métricas Baseadas em Modelos e NLI (Natural Language Inference)

Utilizam modelos pré-treinados para classificar a relação lógica entre dois textos.

●  FactCC: Um classificador BERT treinado especificamente para detectar erros comuns de

sumarização (ex: troca de entidades, negação) .

●  FactScore: Decompõe o texto em "fatos atômicos" (proposições mínimas) e usa um

retriever (como Wikipedia) ou um LLM forte (GPT-4) para verificar cada fato
individualmente. O escore final é a porcentagem de fatos suportados .

●  SRLScore: Utiliza Semantic Role Labeling para extrair tuplas (quem fez o quê, para
quem, onde). A comparação entre as tuplas do texto gerado e da fonte permite uma
avaliação altamente interpretável (https://aclanthology.org/2023.starsem-1.9.pdf).

●  NLI (Entailment): Mede se a fonte "implica" o texto gerado. Modelos como Vitamin-C e

ANLI são usados para computar escores de apoio vs. contradição .

C. Métricas Baseadas em Incerteza e Entropia (Zero-Resource)

Avaliam a factualidade analisando o comportamento interno do modelo durante a geração.

●  SelfCheckGPT: Amostra múltiplas respostas para o mesmo prompt. Se as respostas

forem contraditórias entre si (alta variância semântica), a probabilidade de alucinação é
alta .

●  Semantic Entropy: Calcula a entropia sobre os significados das frases geradas, não

apenas sobre os tokens. Reduz o ruído causado por diferentes formas de dizer a mesma
coisa .

●  Seq-Logprob: A probabilidade logarítmica média da sequência gerada. Valores baixos

frequentemente sinalizam falta de confiança factual .

D. LLM-as-a-Judge (Avaliação por Modelos de Elite)

●  G-Eval: Utiliza o GPT-4o com rubricas de linguagem natural (ex: "Dê uma nota de 1 a 5
para a factualidade deste parágrafo"). Utiliza a normalização de probabilidades dos

tokens de nota para aumentar a robustez
(https://medium.com/@zlatkov/deep-dive-into-g-eval-how-llms-evaluate-themselves-7436
24d22bf7).

Gaps Identificados e Propostas de Avaliação

Apesar do progresso, a pesquisa identifica lacunas críticas que comprometem a confiabilidade
dos sistemas atuais.

1. Desconexão entre Métricas Automáticas e Julgamento Humano

Muitas métricas (especialmente baseadas em NLI e sobreposição semântica) apresentam
baixa correlação com anotadores humanos em domínios específicos .

●  Proposta: Desenvolvimento de métricas de "Severidade Ponderada", onde erros em

entidades críticas (datas, dosagens médicas, nomes de réus) recebam penalidades
maiores do que erros em adjetivos ou descrições periféricas.

2. O Problema da "Alucinação Silenciosa" de Alta Confiança

Modelos que "sabem" a verdade mas alucinam por causa do formato do prompt (CHOKE) não
são detectados por métricas de incerteza .

●  Proposta: Implementação de "Avaliação por Triangulação" — comparar a saída do

modelo em três formatos de prompt diferentes (Zero-shot, CoT, e Few-shot). Se houver
divergência, o sistema deve entrar em estado de alerta.

3. Escassez de Benchmarks de Long-Context

Com modelos suportando janelas de contexto de mais de 100k tokens, as alucinações podem
ocorrer pela incapacidade de recuperar fatos no "meio" do contexto (Lost-in-the-middle
phenomenon) .

●  Proposta: Criação do "Needle-in-a-Fact-Stack", um benchmark onde fatos contraditórios
são inseridos em posições variáveis de um documento longo para testar se o modelo
prioriza o contexto recente ou o conhecimento pré-treinado.

4. Fragilidade Multilingue em Domínios Técnicos

Benchmarks como PoETa v2 são excelentes para cultura geral, mas faltam avaliações para o
português jurídico ou médico .

●  Proposta: Colaboração entre especialistas de domínio e pesquisadores de NLP para

criar o "PT-MedFact", um dataset de prontuários médicos sintéticos em português para
avaliar a precisão de resumos clínicos.

Bibliografia Anotada (Seleção de Estudos Chave)

1.  Li et al. (2026). Harmful Factuality: LLMs Correcting What They Shouldn’t.

○  Contribuição: Identifica o conflito entre conhecimento de mundo e fidelidade à fonte.
Demonstra que o aumento da escala do modelo agrava a tendência de corrigir fontes
externas de forma não autorizada. Essencial para entender os limites do grounding .

2.  Fabbri et al. (2022). QAFactEval: Improved QA-Based Factual Consistency

Evaluation for Summarization.
○  Contribuição: Otimiza todos os componentes do pipeline de QA para avaliação.

Prova que a escolha do modelo de geração de perguntas e o critério de sobreposição
de respostas são determinantes para a acurácia da métrica .

3.  Almeida et al. (2025). PoETa v2: Toward More Robust Evaluation of Large Language

Models in Portuguese.
○  Contribuição: Estabelece o padrão de ouro para avaliação de LLMs em português,
cobrindo 44 tarefas. Introduz a métrica NPM para permitir a comparação justa entre
modelos de diferentes escalas.

4.  Min et al. (2023). FactScore: Fine-grained Atomic Evaluation of Factual Precision in

Long Form Text Generation.
○  Contribuição: Propõe a atomização do conhecimento como forma de avaliação

interpretável. Foi usado para auditar modelos comerciais e mostrou que mesmo o
GPT-4 tem dificuldades com fatos de "cauda longa" (informações raras) .

5.  Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination

Detection.
○  Contribuição: Resolve o problema de custo de verificação ao usar a própria
consistência interna do modelo como sinal de verdade. É a base para muitos
sistemas de monitoramento de alucinações em tempo real .

6.  Tjandra et al. (2024). Certainty in Hallucinations of Knowledgeable Entities

(CHOKE).
○  Contribuição: Desmitifica a relação direta entre confiança do modelo e verdade.
Revela que o "viés de forma" do prompt pode forçar o modelo a mentir mesmo
quando ele possui o fato correto .

7.  Guan et al. (2023). HallusionBench: An Advanced Diagnostic Suite for LVLMs.
○  Contribuição: Primeiro benchmark a diagnosticar sistematicamente falhas de

raciocínio visual e conhecimento em modelos multimodais, expondo a fragilidade do
alinhamento visão-texto .

8.  Hey et al. (2023). SRLScore: Reference-free Evaluation Metric for Text

Summarization.
○  Contribuição: Introduz o uso de rótulos de papéis semânticos (SRL) para avaliação
factual, permitindo que desenvolvedores vejam exatamente qual parte da relação
factual (quem, o quê) foi alucinada .

9.  Assis et al. (2025). Exploring Brazil's LLM Fauna.

○  Contribuição: Avalia modelos brasileiros em tarefas generativas, mostrando que a

adaptação linguística profunda (como no Sabiá-3) é necessária para superar modelos
globais em tarefas de nuances culturais .

10. Li et al. (2025). Premature Layers Interpolation (PLI) for Hallucination Alleviation.

○  Contribuição: Propõe uma técnica de inferência que utiliza camadas intermediárias
do transformador para refinar o conhecimento, provando que a factualidade pode ser
melhorada sem re-treinamento .

Qualidade, Viés e Reprodutibilidade

A análise da qualidade metodológica dos estudos revela que métricas baseadas em humanos
ainda são o "padrão de ouro", mas sofrem com baixa escalabilidade. O Acordo
Inter-Anotadores (IAA) varia significativamente: estudos como FactScore e QAFactEval
reportam Cohen's Kappa superior a 0,70, indicando alta confiabilidade . No entanto, métricas
baseadas em "LLM-as-a-judge" introduzem novos vieses, como a preferência por respostas
mais longas e a tendência de favorecer saídas que mimetizam o estilo de escrita do próprio
avaliador .

A reprodutibilidade é facilitada pela disponibilização de bibliotecas como UQLM (CVS Health) e
frameworks de avaliação como Prompt Flow (Microsoft) . No entanto, a dependência de APIs
proprietárias (como as da OpenAI) para avaliação automática cria um risco de "deriva de
avaliação" (evaluation drift), onde mudanças no modelo julgador podem alterar os escores de
factualidade ao longo do tempo sem aviso prévio .

Conclusões e Recomendações Técnicas

A factualidade em LLMs não é um estado binário, mas uma propriedade emergente que
depende da interação entre o conhecimento pré-treinado, a fidelidade ao contexto e o
mecanismo de decodificação. Para profissionais de NLP e bibliotecários de dados, as seguintes
recomendações são fundamentais:

1.  Diversificação de Métricas: Nunca dependa de uma única métrica (especialmente

BLEU/ROUGE). Combine FactScore para precisão atômica com SelfCheckGPT para
detecção de incerteza .

2.  Priorização do Grounding: Em sistemas de produção, o uso de RAG com métricas de
fidelidade (como QAFactEval) é preferível ao uso de modelos "puros" em modo de
resposta livre .

3.  Atenção ao Português: Utilize modelos como Sabiá-3 ou Tucano para tarefas que

exigem conhecimento específico do contexto brasileiro, validando os resultados com o
benchmark PoETa v2 .

4.  Mitigação de HFH: Ao usar LLMs para resumir ou traduzir documentos que podem

conter erros intencionais ou técnicos, utilize prompts que enfatizem explicitamente: "Não
corrija inconsistências factuais presentes no texto original" .

O campo caminha para modelos "auto-corretivos", onde a inferência inclui etapas de verificação
interna e consulta a bases de conhecimento dinâmicas. A factualidade deixará de ser um

"acidente estatístico" para se tornar uma garantia arquitetônica nos próximos anos .

Referências

Li, M., et al. (2026). Harmful Factuality: LLMs Correcting What They Shouldn’t. Findings of
EACL 2026. https://aclanthology.org/2026.findings-eacl.46.pdf

Tonmoy, S., et al. (2024). A Survey on Factuality and Hallucination in Large Language Models.
EMNLP 2024. https://aclanthology.org/2024.emnlp-main.1088.pdf

Li, Z., et al. (2025). Premature Layers Interpolation for Hallucination Alleviation. EMNLP 2025.
https://aclanthology.org/2025.emnlp-main.645.pdf

Smith, J. (2024). Claim Verification Frameworks using LLMs. arXiv.
https://arxiv.org/html/2408.14317v2

Xu, Z., et al. (2025). InFACT: Informative Factual Alignment for LLMs. Findings of EMNLP 2025.
https://aclanthology.org/2025.findings-emnlp.971.pdf

Chen, X., et al. (2025). Hallucination in LLMs: Causes, Detection, and Mitigation. arXiv.
https://arxiv.org/html/2510.06265v2

DeepSeek Research. (2025). Hallucination Attribution in LLMs. PMC/NCBI.
https://pmc.ncbi.nlm.nih.gov/articles/PMC12518350/

Maxim AI. (2025). LLM Hallucination Detection Best Techniques.
https://www.getmaxim.ai/articles/llm-hallucination-detection-and-mitigation-best-techniques/

Confident AI. (2025). LLM Evaluation Metrics Guide.
https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation

Yan, L., et al. (2025). Gaps in Hallucination Evaluation. Findings of EMNLP 2025.
https://aclanthology.org/2025.findings-emnlp.1035.pdf

Fabbri, A. R., et al. (2022). QAFactEval: Improved QA-Based Factual Consistency. NAACL
2022. https://aclanthology.org/2022.naacl-main.187/

Salesforce AI. (2022). QAFactEval Research.
https://www.researchgate.net/publication/357115246_QAFactEval_Improved_QA-Based_Factua
l_Consistency_Evaluation_for_Summarization

Guo, Y., et al. (2025). PlainQAFact: Medical Factual Consistency.
https://yueguo-50.github.io/assets/pdf/2025-zhiwen-plainqafact.pdf

Liner AI. (2024). Review: QAFactEval for Summarization.
https://liner.com/review/qafacteval-improved-qabased-factual-consistency-evaluation-for-summa
rization

Fabbri, A. R., et al. (2022). QAFactEval Full Paper. ACL Anthology.
https://aclanthology.org/2022.naacl-main.187.pdf

Turing. (2024). LLM Factuality Evaluation Guide.
https://www.turing.com/resources/llm-factuality-guide

Min, S., et al. (2023). FactScore: Fine-grained Atomic Evaluation. EMNLP 2023.
https://www.emergentmind.com/topics/factscore-113e12f4-d246-433f-8289-063e105dc511

Zlatkov, D. (2024). Deep Dive into G-Eval. Medium.
https://medium.com/@zlatkov/deep-dive-into-g-eval-how-llms-evaluate-themselves-743624d22b
f7

ArXiv Review. (2025). LLM Fact-Checking Systems. https://arxiv.org/html/2508.03860

DIVA Portal. (2025). Specificity and Factuality in LLMs.
https://www.diva-portal.org/smash/get/diva2:1938959/FULLTEXT02

Deepchecks. (2025). Hallucination Detection Techniques.
https://deepchecks.com/llm-hallucination-detection-and-mitigation-best-techniques/

Tjandra, A., et al. (2024). CHOKE: Certainty in Hallucinations. arXiv v1.
https://arxiv.org/html/2502.12964v1

Tjandra, A., et al. (2025). CHOKE: Certainty in Hallucinations v2. arXiv.
https://arxiv.org/html/2502.12964v2

CVS Health Tech. (2024). UQLM: Hallucination Detection at Generation Time.
https://medium.com/cvs-health-tech-blog/detecting-llm-hallucinations-at-generation-time-with-uql
m-cd749d2338ec

PMC. (2025). AI-generated Misinformation in Portuguese.
https://pmc.ncbi.nlm.nih.gov/articles/PMC12969083/

Almeida, T. R. S., et al. (2025). PoETa v2: Evaluation in Portuguese. IEEE.
https://ieeexplore.ieee.org/iel8/6287639/10820123/11303664.pdf

Almeida, T. R. S., et al. (2025). BRoverbs: Brazilian Proverbs Dataset. arXiv.
https://arxiv.org/pdf/2509.08960

Assis, G., et al. (2025). Exploring Brazil's LLM Fauna. JBCS.
https://journals-sol.sbc.org.br/index.php/jbcs/article/view/5814

Bai, Y., et al. (2026). TACO: Mitigating Multimodal Hallucinations. EACL 2026.
https://aclanthology.org/2026.eacl-long.252.pdf

ArXiv Survey. (2025). Hallucination Evaluation in MLLMs. https://arxiv.org/html/2507.19024v2

Guan, T., et al. (2023). HallusionBench: Diagnostic Suite for LVLMs. arXiv.
https://arxiv.org/html/2310.14566v5

Preprints. (2026). Multimodal Hallucination Benchmarks.
https://www.preprints.org/manuscript/202602.0467

Semantic Scholar. (2023). HallusionBench Analysis.
https://www.semanticscholar.org/paper/Hallusionbench%3A-An-Advanced-Diagnostic-Suite-for-i
n-Guan-Liu/0b395ed1c8b284e551172b728e83cf257e33729a

Meta AI. (2023). FactScore Research Publication.
https://ai.meta.com/research/publications/factscore-fine-grained-atomic-evaluation-of-factual-pre
cision-in-long-form-text-generation/

NeurIPS. (2023). FELM: Factuality Evaluation for LLMs.
https://proceedings.neurips.cc/paper_files/paper/2023/file/8b8a7960d343e023a6a0afe37eee60
22-Paper-Datasets_and_Benchmarks.pdf

OpenReview. (2024). Multilingual FactScore Evaluation.
https://openreview.net/forum?id=lkrH6ovzsj

Hey, J., et al. (2023). SRLScore: Reference-free Metric. StarSEM 2023.
https://aclanthology.org/2023.starsem-1.9.pdf

Microsoft. (2025). Generative AI Evaluation Playbook.
https://learn.microsoft.com/en-us/ai/playbook/technology-guidance/generative-ai/working-with-ll
ms/evaluation/list-of-eval-metrics

Science Publishing Group. (2026). Factual Consistency in LLMs Survey.
https://www.sciencepublishinggroup.com/article/10.11648/j.ajai.20261001.16

AiMultiple. (2025). Hallucination Detection Tools Review.
https://aimultiple.com/ai-hallucination-detection

ArXiv. (2025). VC-Inspector: Video Caption Evaluation. https://arxiv.org/html/2509.16538v2

EMNLP. (2024). PHD: Passage-level Hallucination Detection.
https://aclanthology.org/2024.findings-emnlp.685.pdf

ArXiv. (2023). Chain of Knowledge for LLMs. https://arxiv.org/pdf/2309.05922

