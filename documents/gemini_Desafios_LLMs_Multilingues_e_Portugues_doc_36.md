# Computer Science > Computation and Language

[Submitted on 22 Mar 2026]

# Title:Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs

View PDF HTML (experimental)Abstract:Although large language models have transformed natural language processing, their computational costs create accessibility barriers for low-resource languages such as Brazilian Portuguese. This work presents a systematic evaluation of Parameter-Efficient Fine-Tuning (PEFT) and quantization techniques applied to BERTimbau for Question Answering on SQuAD-BR, the Brazilian Portuguese translation of SQuAD v1. We evaluate 40 configurations combining four PEFT methods (LoRA, DoRA, QLoRA, QDoRA) across two model sizes (Base: 110M, Large: 335M parameters). Our findings reveal three critical insights: (1) LoRA achieves 95.8\% of baseline performance on BERTimbau-Large while reducing training time by 73.5\% (F1=81.32 vs 84.86); (2) higher learning rates (2e-4) substantially improve PEFT performance, with F1 gains of up to +19.71 points over standard rates; and (3) larger models show twice the quantization resilience (loss of 4.83 vs 9.56 F1 points). These results demonstrate that encoder-based models can be efficiently fine-tuned for extractive Brazilian Portuguese QA with substantially lower computational cost than large generative LLMs, promoting more sustainable approaches aligned with \textit{Green AI} principles. An exploratory evaluation of Tucano and Sabiá on the same extractive QA benchmark shows that while generative models can reach competitive F1 scores with LoRA fine-tuning, they require up to 4.2$\times$ more GPU memory and 3$\times$ more training time than BERTimbau-Base, reinforcing the efficiency advantage of smaller encoder-based architectures for this task.

## Submission history

From: Didier A. Vega-Oliveros [view email]**[v1]**Sun, 22 Mar 2026 21:56:05 UTC (116 KB)

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