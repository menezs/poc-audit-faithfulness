# Computer Science > Computation and Language

[Submitted on 11 Mar 2025 (v1), last revised 19 Mar 2026 (this version, v4)]

# Title:PlainQAFact: Retrieval-augmented Factual Consistency Evaluation Metric for Biomedical Plain Language Summarization

View PDF HTML (experimental)Abstract:Hallucinated outputs from large language models (LLMs) pose risks in the medical domain, especially for lay audiences making health-related decisions. Existing automatic factual consistency evaluation methods, such as entailment- and question-answering (QA) -based, struggle with plain language summarization (PLS) due to elaborative explanation phenomenon, which introduces external content (e.g., definitions, background, examples) absent from the scientific abstract to enhance comprehension. To address this, we introduce PlainQAFact, an automatic factual consistency evaluation metric trained on a fine-grained, human-annotated dataset PlainFact, for evaluating factual consistency of both source-simplified and elaborately explained sentences. PlainQAFact first classifies sentence type, then applies a retrieval-augmented QA scoring method. Empirical results show that existing evaluation metrics fail to evaluate the factual consistency in PLS, especially for elaborative explanations, whereas PlainQAFact consistently outperforms them across all evaluation settings. We further analyze PlainQAFact's effectiveness across external knowledge sources, answer extraction strategies, answer overlap measures, and document granularity levels, refining its overall factual consistency assessment. Taken together, our work presents a sentence-aware, retrieval-augmented metric targeted at elaborative explanations in biomedical PLS tasks, providing the community with both a new benchmark and a practical evaluation tool to advance reliable and safe plain language communication in the medical domain. PlainQAFact and PlainFact are available at: this https URL

## Submission history

From: Zhiwen You [view email]**[v1]**Tue, 11 Mar 2025 20:59:53 UTC (2,017 KB)

**[v2]**Sun, 7 Sep 2025 23:40:32 UTC (3,175 KB)

**[v3]**Mon, 9 Feb 2026 04:23:43 UTC (3,116 KB)

**[v4]**Thu, 19 Mar 2026 17:16:56 UTC (3,117 KB)

### References & Citations

export BibTeX citation
Loading...

# Bibliographic and Citation Tools

Bibliographic Explorer

*(What is the Explorer?)*
Connected Papers

*(What is Connected Papers?)*
Litmaps

*(What is Litmaps?)*
scite Smart Citations

*(What are Smart Citations?)*# Code, Data and Media Associated with this Article

alphaXiv

*(What is alphaXiv?)*
CatalyzeX Code Finder for Papers

*(What is CatalyzeX?)*
DagsHub

*(What is DagsHub?)*
Gotit.pub

*(What is GotitPub?)*
Hugging Face

*(What is Huggingface?)*
ScienceCast

*(What is ScienceCast?)*# Demos

# Recommenders and Search Tools

Influence Flower

*(What are Influence Flowers?)*
CORE Recommender

*(What is CORE?)*# arXivLabs: experimental projects with community collaborators

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.

Have an idea for a project that will add value for arXiv's community? **Learn more about arXivLabs**.