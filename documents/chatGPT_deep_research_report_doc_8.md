# Computer Science > Computation and Language

[Submitted on 10 Nov 2025]

# Title:Stress Testing Factual Consistency Metrics for Long-Document Summarization

View PDF HTML (experimental)Abstract:Evaluating the factual consistency of abstractive text summarization remains a significant challenge, particularly for long documents, where conventional metrics struggle with input length limitations and long-range dependencies. In this work, we systematically evaluate the reliability of six widely used reference-free factuality metrics, originally proposed for short-form summarization, in the long-document setting. We probe metric robustness through seven factuality-preserving perturbations applied to summaries, namely paraphrasing, simplification, synonym replacement, logically equivalent negations, vocabulary reduction, compression, and source text insertion, and further analyze their sensitivity to retrieval context and claim information density. Across three long-form benchmark datasets spanning science fiction, legal, and scientific domains, our results reveal that existing short-form metrics produce inconsistent scores for semantically equivalent summaries and exhibit declining reliability for information-dense claims whose content is semantically similar to many parts of the source document. While expanding the retrieval context improves stability in some domains, no metric consistently maintains factual alignment under long-context conditions. Finally, our results highlight concrete directions for improving factuality evaluation, including multi-span reasoning, context-aware calibration, and training on meaning-preserving variations to enhance robustness in long-form summarization. We release all code, perturbed data, and scripts required to reproduce our results at this https URL.

## Submission history

From: Zain Muhammad Mujahid [view email]**[v1]**Mon, 10 Nov 2025 23:24:25 UTC (996 KB)

### Current browse context:

cs.CL

### References & Citations

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