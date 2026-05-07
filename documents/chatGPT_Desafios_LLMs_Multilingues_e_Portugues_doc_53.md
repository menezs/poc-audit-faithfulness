# Multilingual Large Language Models: A Systematic Survey

###### Abstract

This paper provides a comprehensive survey of the latest research on multilingual large language models (MLLMs). MLLMs not only are able to understand and generate language across linguistic boundaries, but also represent an important advancement in artificial intelligence.

We first discuss the architecture and pre-training objectives of MLLMs, highlighting the key components and methodologies that contribute to their multilingual capabilities.

We then discuss the construction of multilingual pre-training and alignment datasets, underscoring the importance of data quality and diversity in enhancing MLLM performance.

An important focus of this survey is on the evaluation of MLLMs. We present a detailed taxonomy and roadmap covering the assessment of MLLMs’ cross-lingual knowledge, reasoning, alignment with human values, safety, interpretability and specialized applications. Specifically, we extensively discuss multilingual evaluation benchmarks and datasets, and explore the use of LLMs themselves as multilingual evaluators.

To enhance MLLMs from black to white boxes, we also address the interpretability of multilingual capabilities, cross-lingual transfer and language bias within these models.

Finally, we provide a comprehensive review of real-world applications of MLLMs across diverse domains, including biology, medicine, computer science, mathematics and law.
We showcase how these models have driven innovation and improvements in these specialized fields while also highlighting the challenges and opportunities in deploying MLLMs within diverse language communities and application scenarios. We listed the paper related in this survey and publicly available at a Github repository.111https://github.com/tjunlp-lab/Awesome-Multilingual-LLMs-Papers

###### Contents

- 1 Introduction
- 2 Taxonomy and Roadmap
- 3 Architectures
- 4 Multilingual Corpora
- 5 Pre-training
- 6 Multilingual Tuning
- 7 Multilingual Evaluation
- 8 Interpretability
- 9 Application
- 10 Challenges and Future Directions
- 11 Conclusion

## 1 Introduction

Scientific progress and technological innovation are a key driving force of human society, and the recent advancement of language science and technology has been part of them. While being a crucial mean for people to communicate, language has enhanced knowledge accumulation and cultural inheritance. And given the multiple languages in the world, efforts towards economic, cultural, political and other forms of communication have always require multilingual understanding. With the deepening of globalization, the development of language technology and the search for multilingual comprehension has only accelerated.

In the current information society and Artificial Intelligence (AI) age, we are at an unprecedented juncture of knowledge availability and cognitive revolution. Being a key factor underlying this transformative shift, generative Large Language Models (LLMs) are fostering a new dimension of intelligence with their extraordinary language processing capabilities — the “intelligence of language” (Chang et al., 2024). Though still in its infancy, this form of intelligence is sweeping across every domain of scientific research and technological innovation.

We are witnessing the emergence of a new level of intelligence capabilities that has the potential to transcend human cognitive limits, specially when this encompasses different languages and cultures. Therefore, endowing LLMs with multilingual capabilities has become a crucial endeavour towards realizing their full potential.

A Multilingual LLM (MLLM) is an LLM that seeks to learn such capabilities across multiple languages by means of a single model.

Efforts to equip AI models with multilingual capabilities can be traced back to machine translation systems based on statistical methods, such as IBM’s Candide system (Hutchins, 1999), and even further back, to rule-based systems, relying on manually designed correspondences between expressions of different languages.

Over a decade ago, with the rapid advancement of computational power and the arrival of the big data era, neural network-based end-to-end multilingual language models began to emerge. In 2014, Cho et al. (2014) proposed an attention-based neural machine translation model, greatly improving translation quality. In 2016, Google researchers introduced the GNMT neural machine translation system, built on a large-scale corpus, achieving state-of-the-art performance across multiple language pairs.

These early multilingual models typically required separate training and optimization for each language pair, lacking scalability and generalization capabilities. Seeking to address this bottleneck, a new generation of pre-trained multilingual language models emerged.

In 2019, Devlin et al. (2019) introduced the BERT model, which achieved stronger cross-lingual transfer abilities through pre-training on large-scale multilingual corpora.

Since then, a series of multilingual pre-trained language models, including XLM (Conneau & Lample, 2019a), mBART (Liu et al., 2020), among many others, have been proposed, broadening the potential of multilingual modeling. Researchers no longer viewed multilingual processing as an extension of machine translation but rather as a holistic ability to understand and handle meaning across languages.

To improve the training efficiency of models on massive multilingual data, innovative training objectives and architectures have been continuously proposed, such as the denoising autoencoding of XLM-R (Conneau et al., 2020a) and the multitask architecture of mT5 (Xue et al., 2021), just to mention a couple of examples, among countless others.

The rapid adoption of ChatGPT (OpenAI, 2022), attracting over 100 million users in just two months, has highlighted the transformative potential of LLMs. Their capabilities, including natural text generation, code generation and tool usage, coupled with their surprisingly strong multilingual abilities, have both triggered enthusiasm and critical questions (Zhang et al., 2024c; Tars et al., 2022).

The path forward for MLLMs is not without challenges. While many surveys have explored specific aspects of MLLMs (Xu et al., 2024e; Qin et al., 2024), such as their training data, architecture or applications, a comprehensive examination of their multilingual capabilities, limitations, and challenges is lacking.

Additionally, critical issues related to responsible AI, such as fairness and toxicity, have not been adequately addressed in that context (Philippy et al., 2023; Tang et al., 2024).

The present survey aims to fill this gap by providing a comprehensive survey of the research on MLLMs. We will analyze the specific challenges MLLMs face in handling linguistic diversity, thus including non-English and low-resource languages, and explore data construction strategies, model training and fine-tuning approaches, as well as core responsible AI issues. Furthermore, we will delve into the practical challenges of deploying MLLMs in real-world language communities and application domains.

This survey seeks thus to provide essential insights for navigating the evolving landscape of MLLMs and their impact on a global scale.

We will discuss data construction strategies, model training and fine-tuning approaches, and core Responsible AI issues such as fairness, bias and toxicity in the context of MLLMs. We will also analyze the specific challenges MLLMs encounter in handling linguistic diversity, along with relevant evaluation methodologies. Furthermore, we will delve into the practical challenges of deploying MLLMs in real-world language communities and application domains.

As the performance of MLLMs continues to improve, leveraging them for developing multilingual language technologies globally has become increasingly viable. This survey aims thus to provide researchers in linguistics and artificial intelligence with a cutting-edge perspective on the development of MLLMs. We will analyze the opportunities and challenges that MLLMs face in towards realizing the full potential of language intelligence.

Our goal is to facilitate more inclusive and responsible language technology development, ensuring that these powerful tools are accessible and beneficial to diverse language communities worldwide.

By exploring the intricacies of multilingualism, encompassing linguistic diversity, language variations, and multilingual knowledge transfer, we strive to bridge the gap between the current state of LLM research and the urgent need for equitable and effective language understanding across all languages. Through this comprehensive examination of MLLMs, we aim to contribute for language intelligence to empower global communication, to foster cultural understanding and to unlock the potential of diverse knowledge systems.

## 2 Taxonomy and Roadmap

To provide a comprehensive overview of the rapidly evolving area of Multilingual Large Language Models, the present survey adopts a structured taxonomy, organizing the research landscape into six fundamental and interconnected domains.

Beginning with the foundational element of multilingual data, this survey considers diverse sources like web crawls, books, and code repositories.

It then delves into the neural architecture choices for building effective MLLMs, analyzing the strengths and weaknesses of common architectures like decoder-only and encoder-decoder.

Building upon these foundations, this survey addresses the methodologies for pre-training and fine-tuning MLLMs, discussing various objectives like masked language modeling and translation language modeling, as well as fine-tuning techniques like instruction tuning and preference tuning.

The crucial aspect of evaluating MLLMs performance is addressed through a comprehensive review of benchmarks and datasets, emphasizing the importance of balanced evaluation across diverse languages, including low-resource languages.

This paper then takes into account the “black box” dimension of MLLMs and discusses interpretability techniques to understand how these models achieve their multilingual capabilities and how to identify and mitigate potential linguistic biases (Gurgurov et al., 2024; Blevins et al., 2024; Nezhad & Agrawal, 2024; Kojima et al., 2024b; Liu et al., 2024b).

Finally, the survey showcases diverse real-world applications of MLLMs. Our overarching objective is to address these six fundamental domains, as illustrated in Figure 1.

The next Section 3 discusses the key architectural components and pre-training objectives that contribute to the development and training of MLLMs. MLLMs typically adopt either the decoder-only or encoder-decoder architecture. The acquisition of multilingual knowledge comes from training on multilingual corpora. The main architectural difference between multilingual and monolingual LLMs lies in the vocabulary, which needs to be larger in MLLMs to address the out-of-vocabulary issue. In terms of pre-training objectives, MLLMs use similar techniques to monolingual models, such as masked language modeling, but also explore novel objectives like translation language modeling to enhance cross-lingual abilities.

Section 4 discusses the training data for MLLMs. The pre-training stage requires large amounts of unlabeled text data, while the alignment stage requires supervised fine-tuning with parallel data and reinforcement learning with feedback data. In terms of pre-training data, we summarize the characteristics of various web data, book data, code data, and academic paper data. Regarding parallel data, we detail the generation methods for supervised fine-tuning data, including human-generated and model-assisted generation, as well as the process of collecting reinforcement learning feedback data.

The pre-training of MLLMs is discussed in Section 5, where the process of data curation is examined, together with the pre-training objectives that should be set and the pre-training strategies that have been explored in the literature.

Section 6 presents strategies for multilingual tuning, with the goal of adapting the general capabilities of multilingual LLMs to specific objectives. It begins by discussing approaches that extend LLMs tuning directly into multilingual contexts, with a focus on multilingual data collection and cross-lingual transfer. Additionally, it introduces strategies to further enhance cross-lingual alignment during multilingual tuning. Finally, it addresses specialized tuning techniques for the enhancement of specific multilingual capabilities.

Next, in Section 7, we address the evaluation of MLLMs by extensively discussing the evaluation, from multilingual tokenization to the datasets and including benchmarks used to evaluate MLLMs’ capabilities across diverse tasks in a multilingual context. We further discuss the evaluation methods for determining the multilingualism of MLLMs and explore the use of MLLMs themselves in evaluating MLLMs’ performance.

In seeking to transform an MLLM from a black box to a white box, Section 8 addresses the core question of how the model represents multilingual capacities. It also tackles advanced issues such as cross-lingual transfer and the causes of language bias.

In section 9, we undertake a thorough investigation into the evolutionary pathway and the latest breakthroughs of MLLMs spanning diverse disciplinary realms, with a particular emphasis on their real-world implementations and applications.

Overall, our aim is to address and help to clarify the following fundamental questions:

-
•
What are the capabilities of MLLMs?

-
•
What is the language boundary of MLLMs?

-
•
What factors must be taken into account when constructing and tuning MLLMs?

-
•
How to evaluate the multilingual transfer capabilities of MLLMs?


## 3 Architectures

As large language models can excel in English, their performance varies though across the other languages. Accordingly, there is a growing need to advance multilingual large language models in order to ensure a balanced performance across a broader range of languages. Against this context, the goal of the present section is to provide a thorough examination of both LLMs and MLLMs. We will discuss the architecture of these models, typically based on the Transformer model (Vaswani et al., 2017), as well as other architectures that may have the potential to challenge such supremacy of Transformers. Organized according to the major families of architectures, the deployment of major MLLMs along the last few years are summarized in Figure 2.

### 3.1 Dense Transformer

MLLMs tend typically to adopt the same architecture as monolingual LLMs, even in the era of pre-trained language models (Bendale et al., 2024). However, the finer details of the Transformer architecture can vary significantly when building a top performing language model, which inherently affects the performance and stability of pre-training.

In an initial period, the preferred architecture was usually either the encoder-only model (Devlin et al., 2018; Conneau & Lample, 2019b; Conneau et al., 2019; Sun et al., 2021) or the encoder-decoder model (Lewis et al., 2019; Raffel et al., 2020b; Xue et al., 2020; Costa-jussà et al., 2022), both of which typically utilize non-causal masked attention to perform denoising tasks.

Following the subsequent success of GPT models (Brown et al., 2020a; Ouyang et al., 2022a), the preferred approach shifted to the decoder-only model, which excel in generation tasks. Useful pre-training of these models involves training with a substantial number of parameters, say over 7 billion, on a massive amount of tokens, over 1 trillion. This process requires extensive GPU memory and time, which represents a significant cost. To address these challenges and enhance the performance of pre-trained models, various modifications have been made to the vanilla Transformers, often related to the self-attention mechanism, layer normalization, activation functions and position embeddings.

LLaMA (Touvron et al., 2023b) is one of the first open foundation models that paved the way for the success of open source MLLMs. Despite using the vanilla Transformer architecture, LLaMA introduces significant modifications. It adopts pre-normalization like GPT-3 (Brown et al., 2020a), and uses RMSNorm (Zhang & Sennrich, 2019) instead of the traditional layer normalization, which eliminates the shift operation of layer normalization and improves FLOPs. Additionally, LLaMA chooses SwiGLU, following PaLM (Chowdhery et al., 2022), to enhance performance, and Rotary Position Embedding (RoPE) (Su et al., 2024), following GPTNeo (Black et al., 2021), instead of standard absolute position embeddings to support long-context extension. LLaMA also removes the linear bias of the feed-forward networks and self-attention mechanism. This design approach has been widely adopted by successors such as Yi (Young et al., 2024), Deepseek-LLM (Bi et al., 2024), InternLM (Team, 2023).

Other MLLMs employ different architectures. For example, Qwen1.0 (Bai et al., 2023b) retains the bias in self-attention, and Gemma uses GeGLU (Team et al., 2024) instead of SwiGLU. Falcon (Almazrouei et al., 2023) and BaiChuan-13B (Yang et al., 2023b) uses ALiBi (Press et al., 2021) instead of RoPE for position embeddings. To decrease memory usage during inference, Grouped Query Attention (GQA) (Ainslie et al., 2023) has been implemented in many LLMs, including LLaMA-2 (Touvron et al., 2023d), Yi (Young et al., 2024), DeepSeek-LLM (Bi et al., 2024), InternLM 2 (Cai et al., 2024), and OpenELM (Mehta et al., 2024). Compared to Multi-Head Self-Attention (MHA), GQA groups multiple queries and uses a single key and value head, reducing memory usage.

MLLMs like BLOOM (Le Scao et al., 2023) and PolyLM (Wei et al., 2023) use the GeLU activation function and vanilla layer normalization, as well as ALiBi instead of RoPE for position embeddings. These architectural differences are not specific to language, but rather general improvements. BLOOM also expanded the vocabulary size to 250K to reduce the risk of under-segmenting for multilingual purposes, whereas other monolingual LLMs from the same period used much smaller vocabulary sizes, typically around 32K. Another MLLM FuxiTranyu (Sun et al., 2024) makes a balance between performance and training efficiency, using GeLU activation function and RMSNorm. Similar to BLOOM, FuxiTranyu expands the vocabulary size to 250K to fit the multilingual corpora.

Researchers have also proposed new alternatives to MHA or GQA. For instance, the Mistral-7B model (Jiang et al., 2023) adopts Sliding Window Attention (SWA), and DeepSeek-V2 (DeepSeek-AI et al., 2024a) proposes Multi-Head Latent Attention (MLA). These variants aim to reduce the KV cache during inference.

### 3.2 MoE Transformer

The Mixture of Experts (MoE) architecture is an effective approach for scaling model parameters while keeping the FLOPs similar to their dense counterparts. Research on MoE models has shown promising results across multiple tasks (Shazeer et al., 2017).

OpenMoE (Xue et al., 2024) is one of the first attempts to build an open-sourced LLM adopting an MoE approach, with parameters ranging from 650M to 34B, pre-trained on 1T tokens.
Unlike Switch Transformer or GShard (Fedus et al., 2022; Lepikhin et al., 2020), which replace the Feed-Forward Network (FFN) module with MoE, OpenMoE interleaves the MoE module with the FFN module.
Mixtral-8x7B (Jiang et al., 2024), in turn, is an outstanding LLM that pushes forward the experimentation with MoE.
It adopts an architecture similar to the dense Mistral-7B model, replacing the FFN module with the MoE module.
Mixtral-8x7B contains 8 experts per layer and uses a top-2 routing mechanism to select experts for each token. Xai’s Grok-1 model222https://x.ai/blog/grok follows this setup, scaling the parameters to 314B.

DeepSeek MoE (Dai et al., 2024) employs a similar architecture but with a fine-grained expert design. The original experts are segmented into experts and correspondingly routed to experts instead of experts to achieve a more flexible combination of activated experts. Additionally, it leaves experts as shared experts, leading to a final number of routed experts of while experts are activated. Using this MoE architecture, performance similar to their dense counterparts is achieved, with the same total number of parameters. The Qwen1.5-MoE series (Team, 2024a) follows DeepSeek MoE’s successful approach, using 4 shared experts and 4 of 60 activated experts. DBRX (Team, 2024b), in turn, uses a similar idea to build a fine-grained MoE architecture with 16 experts, activating 4 of them. These fine-grained MoE shows promising results compared to Mixtral-8x7B and Grok-1. However, they do not use shared experts.

In summary, the MoE approach provides an effective way to scale the parameters of MLLMs, and fine-grained expert design leads to surprisingly good performance compared to dense counterparts. In the context of multilingual models, the main issue with model architecture is often its limited capabilities. It remains an open question to be examined whether a fine-grained MoE architecture would significantly boost the performance of MLLMs.

### 3.3 Competitors of Transformer

Despite the supremacy of the Transformer architecture for MLLMs, it has a significant drawback: its time complexity scales quadratically with the sequence length. Competitors to the transformer often propose architectures with linear time complexity.

RWKV (Peng et al., 2023) is a variant of the Recurrent Neural Networks (RNN) architecture that stacks residual blocks with time-mixing and channel-mixing modules. RWKV has scaled to 7 billion parameters and matches Transformer performance on multiple benchmarks. Mamba (Gu & Dao, 2023) follows previous state-space models like S4 and H3 (Gu et al., 2022; Fu et al., 2023), adopting the state-space architecture while scaling parameters to 1.3 billion. It has shown promising results on multiple validation sets using perplexity as the metric, although whether it can scale much larger remains unknown.

Jamba (Lieber et al., 2024) can be seen as a successful fusion of Mamba and Transformers, stacking Mamba layers, Mamba with MoE layers, and Transformer layers together. Jamba inherits the high-quality output of transformers and the high throughput of Mamba, demonstrating remarkable performance on several benchmarks compared to LLaMA-2 70B and Mixtral 8x7B, with a total of 52 billion parameters and 12 billion activated parameters.

## 4 Multilingual Corpora

Training MLLMs typically involves three major stages: pre-training, supervised fine-tuning (SFT), and reinforcement learning from human feedback (RLHF).
Pre-training requires vast amounts of unlabeled, raw text, while SFT and RLHF occur during the alignment phase.
The GPT-3 (Brown et al., 2020b), based on the transformer architecture, demonstrates strong contextual learning abilities with pre-training using only large amounts of unlabeled data. However, it still shows significant gaps in areas such as value alignment.
The powerful capabilities of ChatGPT arise from quality data used in supervised fine-tuning and intense RLHF (Ouyang et al., 2022b).
Subsequently, skillful chat models such as the Qwen-Chat series (Bai et al., 2023a), LLaMA-Chat series (Touvron et al., 2023a), and Baichuan 2-Chat series (Yang et al., 2023a) have emerged, all having undergone both SFT and RLHF.
Consequently, gathering, preparing, curating and handling data deserves and has gained substantial research, as witnessed by the expanding ELRA Language Resources Association’s LREC conferences,333https://www.elra.info/elra-events/lrec/ among others.

In this section, we will introduce the pre-training and alignment data for MLLMs separately. For the alignment phase, we will discuss SFT data and RLHF data in detail.

### 4.1 Multilingual Pre-training Datasets

Pre-training is the most time-consuming and resource-intensive stage during a typical MLLMs training.
Compared to alignment data, the amount of pre-training data tends to be much larger. Qwen (Bai et al., 2023a) has undergone pre-training on 2 to 3 trillion tokens, Baichuan 2 (Yang et al., 2023a) on 2.6 trillion tokens, and the LLaMA 3444https://ai.meta.com/blog/meta-llama-3 model on 15 trillion tokens.
To ensure that the model can learn and generalize across input data during the pre-training stage, the pre-training data encompasses a wide range of fields and sources, and data filtering methods should be employed to ensure its quality.
For MLLMs, a rich collection of multilingual data enables the model to learn connections between languages, and generalize across them, thereby enhancing its multilingual capabilities.

In this section, we will focus on multilingual pre-training data sets.

| name | open-source | from/type | size | language | low% | non-English% | date | used by |
Anna’s Archive555https://zh.annas-archive.org
|
✔ | Book | 862.2 TB | Multi | N/A | N/A | 2024-06 | / |
| CC100 (Conneau et al., 2020a) | ✔ | CommonCrawl 2018 | 2.5TB | 100 | 61% | 85% | 2020-07 | XLM-R |
| CulturaX (Nguyen et al., 2024) | ✔ | mC4 & OSCAR | 27TB (6.3T tokens) | 167 | 34% | 55% | 2023-09 | / |
| mC4 (Xue et al., 2021) | ✔ | CommonCrawl | 251GB | 101 | 46% | 62% | 2021-06 | mT5 |
| MultiUN (Eisele & Chen, 2010) | ✔ | Parallel Corpora | 4353MB | 7 | N/A | N/A | 2010-05 | / |
News-crawl666https://commoncrawl.org/blog/news-dataset-available
|
✔ | CommonCrawl | N/A | Multi | N/A | N/A | 2024-03 | / |
OSCAR 23.01777https://oscar-project.org (Ortiz Su’arez et al., 2020; Ortiz Su’arez et al., 2019)
|
✔ | CommonCrawl | 9.49TB | 153 | 36% | 64% | 2023-01 | / |
| ParaCrawl (Bañón et al., 2020) | ✔ | Parallel Corpora | 1527 M sentences | 48 | N/A | N/A | 2021-09 | / |
| RedPajama (Computer, 2023) | ✔ | Mixed | 1.2T tokens | Multi | N/A | N/A | 2023-04 | / |
| RedPajama v2 (Computer, 2023) | ✔ | CommonCrawl | 30T tokens | 5 | 14% | 32% | 2023-10 | / |
| ROOTS (Laurençon et al., 2023) | ✔ | OSCAR/Github etc. | 1.6TB | 59 | 15% | 70% | 2023-03 | BLOOM |
| UNCorpus (Ziemski et al., 2016) | ✔ | Parallel Corpora | 799,276 docs | 6 | N/A | N/A | 2016-05 | / |
| FineWeb | ✔ | CommonCrawl | 29.2 TB | Multi | N/A | N/A | 2024-04 | / |
Gutenberg project888https://www.gutenberg.org
|
✔ | Books | 70,000+ Books | Multi | N/A | N/A | 2024-06 | / |
| Zyda (Tokpanov et al., 2024) | ✔ | Extract from existing data | 1.3T tokens | Multi | N/A | N/A | 2024-06 | / |

#### 4.1.1 Multilingual Data from the Web

Taking into account the sources of data, typically the data can be roughly divided into web crawl, book, code, academic paper, news, and others.

Considering the distribution of data domains, they can be divided into general and domain-specific data.

For pre-training datasets, the focus is mainly on general data. We list these data sets and summarize its major aspects in Table 1.

The most well-known source of pre-training data for MLLMs is CommonCrawl999https://commoncrawl.org, a large-scale collection of web dumps.
It crawls billions of web pages every month and releases a snapshot, now containing snapshots from 2008 to the present, and it is continuously being updated.
CommonCrawl contains web data in multiple languages but includes a lot of pornographic, violent, and misaligned information that needs to be filtered before use.

Nowadays, many web-based corpora are cleaned and obtained from CommonCrawl.
CC100 (Conneau et al., 2020a), mC4 (Xue et al., 2021), FineWeb101010https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1, RedPajama v2 (Computer, 2023), RedPajama (part) (Computer, 2023), and SlimPajama (Shen et al., 2023) are extracted and filtered from CommonCrawl.
They contain multiple languages and represent the largest data volumes.

CC100 (Conneau et al., 2020a) is a dataset gathered from the web with over 100 languages, generated from 12 snapshots of CommonCrawl from the year 2018. Only one snapshot was used for English, while the other languages were extracted using all 12 snapshots. This dataset was processed using CCNET (Wenzek et al., 2020), an open-source CommonCrawl processing pipeline. The CCNET processing pipeline includes language identification, deduplication, and filtering based on language model perplexity.

mC4 (Xue et al., 2021) is the multilingual version of the C4 dataset, which is primarily aimed at English while mC4 covers 108 languages. It is extracted and deduplicated from CommonCrawl, followed by filtering based on various rules, including line length etc.

RedPajama (Computer, 2023) was initiated to reproduce the training data for LLama. Most of its data comes from CommonCrawl and C4, with a small portion from code, books, papers, and wikipedias.

RedPajama v2 (Computer, 2023) is a large-scale dataset processed using the CCNET pipeline on CommonCrawl, containing data in 5 languages.

SlimPajama (Shen et al., 2023) is a high-quality dataset derived from RedPajama after extensive filtering and deduplication.

FineWeb is primarily English with a small amount of multilingual data. It has undergone more meticulous cleaning, including common URL-based filtering and custom filters for removing list-like documents, documents with repeated lines, and documents with potentially incorrect line formats. After thorough processing, FineWeb’s data quality surpasses other high-quality datasets such as SlimPajama and C4.

Zyda (Tokpanov et al., 2024) dataset, encompassing 1.3 trillion tokens, is a high-quality pre-training dataset created by integrating multiple premium datasets such as RefinedWeb, SlimPajama, C4, and arXiv. This integration process involved meticulous de-duplication and filtering at a fine-grained level, both within individual datasets and across different datasets.

Oscar111111https://oscar-project.org (Ortiz Su’arez et al., 2020; Ortiz Su’arez et al., 2019) is extracted from CommonCrawl through filtering and deduplication, and it also serves as one of the sources for the ROOTS datasets.

CulturaX (Nguyen et al., 2024) is derived from both Oscar and mC4. Consequently, there is a relatively high overlap among these datasets.

#### 4.1.2 Multilingual Book Datasets

Books represent a high-quality segment of pre-training data once all books should have undergone a careful manual review, and the complexity and quality are much higher than those of web crawls.
Two massive e-book projects, Anna’s Archive121212https://zh.annas-archive.org and the Gutenberg Project131313https://www.gutenberg.org, exemplify this type of data. Anna’s Archive currently holds over 862.2 TB of data, while the exact volume of data in the Gutenberg Project remains unknown.

#### 4.1.3 Datasets with Code and Articles

Programming code and academic papers represent the highest quality data for pre-training. Studies have shown that code data can significantly enhance the performance of models on various tasks.
Most of the code data is sourced from GitHub141414https://github.com, which hosts a vast amount of open-source code, following a series of filtering and cleaning processes.

BigCode includes several code datasets, such as the Stack (Kocetkov et al., 2022) and Stack v2 (Lozhkov et al., 2024), featuring over 600 programming languages including C++, Java, Python etc., which are used by the large code model StarCoder2(Lozhkov et al., 2024).

Academic materials primarily include journal and conference papers, mainly sourced from open-access repositories such as arXiv151515https://arxiv.org, PubMed161616https://pubmed.ncbi.nlm.nih.gov, and PhilPapers.171717https://philpapers.org
These materials provide high-quality knowledge for the model and are typically included as part of the dataset.

| name | open-source | entries | language | low% | non-English% | date | used by |
| Aya Collection (Singh et al., 2024) | ✔ | 513M | 114 | Yes | balance | 2024-02 | N/A |
| Aya Dataset (Singh et al., 2024) | ✔ | 204K | 65 | Yes | balance | 2024-02 | N/A |
| Bactrain-X (Li et al., 2023c) | ✔ | 3M | 52 | Yes | balance | 2023-05 | N/A |
| CAMEL (Li et al., 2023a) | ✔ | 1.6M | multi | No | balance | 2023-03 | N/A |
| Flan 2021 (Wei et al., ) | ✔ | 62 datasets | multi | N/A | N/A | 2021-09 | Flan LAMDA |
| Flan 2022 (Longpre et al., 2023) | ✔ | 1836 datasets | multi | N/A | N/A | 2023-01 | Flan T5 |
GuanacoDataset181818https://guanaco-model.github.io
|
✔ | 534K | 5 | No | N/A | 2023-03 | Guanaco |
| LMSYS-Chat-1M (Zheng et al., 2023a) | ✔ | 1M | multi | Little | Little | 2023-09 | N/A |
| OASST1 (Köpf et al., 2023) | ✔ | 161K | 35 | 20% | 57% | 2023-04 | N/A |
OpenOrca191919https://huggingface.co/datasets/Open-Orca/OpenOrca
|
✔ | 4.2M | multi | Little | Little | 2023-06 | N/A |
| Phoenix-sft-data-v1 (Chen et al., 2023e; d) | ✔ | 464K | multi | < 20% | 41% | 2023-05 | Phoenix |
| SUPER-NATURAL INSTRUCTIONS (Wang et al., 2022) | ✔ | 1616 datasets | 55 | Little | Few | 2022-04 | Tk-Instruct |
| xP3 (Muennighoff et al., 2023a) | ✔ | 82 datasets | 46 | 28% | 60% | 2022-11 | BLOOMz |

### 4.2 Multilingual Alignment Datasets

In this section, we introduce the datasets for so-called alignment. Alignment is an important research topic as pre-training enables models to acquire knowledge, while alignment allows models to follow instructions and align to human preferences and ethics. The alignment process for MLLMs is usually divided into two stages: Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF).

#### 4.2.1 Datasets for Supervised Fine-Tuning

SFT involves inputting diverse instructions and producing the respective outputs that align with human preferences.

Two data-related factors influence the effectiveness of instruction fine-tuning. First, data quality: high-quality data is more effective than larger quantities of data (Touvron et al., 2023d). Second, the diversity of instructions: a richer variety of instruction types tend to better help the alignment of the models.

The development of SFT datasets can be categorized into LLM-generated and human-annotated data. According to LLaMA 2 (Touvron et al., 2023d), synthesized SFT data is competitive with human-annotated SFT data. Consequently, a significant portion of SFT data is actually derived from model-assisted generation or entirely model-generated. This approach allows for the reallocation of substantial human annotation costs to other alignment data, such as RLHF.

We classify the data into three categories based on the method of synthesis: AI-Generated, Manually-Created, and mixed, and provide an overview of the SFT datasets.

AI-generated data is produced by MLLMs like GPT-4202020https://openai.com/index/gpt-4/ or others, sometimes using seed instructions or prompts provided by humans. Manually-created data is constructed entirely by humans. Mixed datasets include portions that are manually-created, AI-generated, and data obtained by applying instruction templates to traditional NLP datasets. A summary of multilingual SFT datasets is shown in Table 2.

##### AI-Generated Data

-
•
The alpaca dataset (Taori et al., 2023) is a SFT dataset of English instructions generated by GPT-4, containing a total of 52K samples, and serves as a source for some SFT data.

-
•
The Bactrian-X (Li et al., 2023c) dataset includes 3.4 million pairs of instructions and responses across 52 languages. These instructions are derived from alpaca dataset and Dolly (Conover et al., 2023b) and translated into 52 languages, with model responses generated by GPT-3.5-turbo.

212121https://openai.com/index/gpt-3-5-turbo-fine-tuning-and-api-updates/ -
•
CAMEL (Li et al., 2023a) provides a role-playing SFT dataset that includes AI society and code instruction fine-tuning data, comprising a total of 584K entries, of which approximately 107K have been translated into multiple languages.

-
•
The GuanacoDataset

222222https://guanaco-model.github.io extends the alpaca dataset to include Japanese, German and Chinese, adding 534K entries. -
•
OpenORCA

232323https://huggingface.co/datasets/Open-Orca/OpenOrca is an open-source reproduction based on the Orca paper (Mukherjee et al., 2023), primarily in English. It expands entries from FLAN by using ChatGPT 3.5 or ChatGPT 4 to generate additional responses.

##### Manually-Created Data

-
•
The Aya (Singh et al., 2024) is an extensive multilingual instruction fine-tuning dataset. It includes instructions written by native speakers in 65 different languages.

-
•
OASST1 (Köpf et al., 2023) is a large-scale, multilingual assistant-style conversation corpus entirely generated and annotated by humans. It encompasses 35 languages and includes over 161K messages.


##### Mixed

-
•
The Aya collections (Singh et al., 2024) comprise data translated from other languages, data converted according to instruction templates, and the original Aya dataset, totaling approximately 513 million samples.

-
•
LMSYS-Chat-1M (Zheng et al., 2023a) includes 1 million real-world conversations with 25 MLLMs. These conversations were collected from 21K unique IP addresses on MLLMs dialogue website.

-
•
Phoenix-sft-data-v1 (Chen et al., 2023e; d) contains 465K instruction and conversation data entries. The instruction data includes Chinese and English instructions sourced from the Alpaca dataset, as well as instructions in other languages obtained through translation. The outputs for the other language instructions consist of two parts: one part directly translates the outputs from the Alpaca dataset, and the other part translates only the instructions, with the outputs generated by GPT-3.5. Additionally, some instruction data is synthesized manually using a self-instructed approach.

-
•
Flan2021 (Wei et al., ) consists of 62 instruction datasets built from existing datasets, primarily in English. Flan2022 (Longpre et al., 2023) expands on this by adding chain-of-thought (CoT) and dialogue data, encompassing 1,836 tasks.

-
•
Super-NaturalInstructions (Wang et al., 2022) collects over 1,600 NLP tasks and converts them into an instruction dataset format.


#### 4.2.2 RLHF data

The RLHF datasets, also known as the preference datasets, typically includes dialogue in the form of inputs and outputs, with each output containing preference information from other models or human feedback.

Compared to SFT data, the cost for RLHF data is much higher. It is estimated that LLama 2 (Touvron et al., 2023d) spent around $8 million to annotate its RLHF dataset.242424https://www.interconnects.ai/p/llama-2-from-meta?sd=pf

Consequently, publicly available preference datasets are relatively scarce, and multilingual preference datasets are even rarer. The main ones include Chatbot Arena Conversations and OASST1 pairwise RLHF reward.

OASST1 (Köpf et al., 2023) dataset comprises multi-turn conversations between humans and MLLMs, including human feedback on the dialogues. This dataset is structured in the form of dialogue trees, with a total of 161K dialogue trees containing 461K quality rates.

Chatbot Arena Conversations (Zheng et al., 2023b) includes 33K pairs of human preference data collected from 13K unique IP addresses. Each sample features two model-generated results and human preference information in different directions.

## 5 Pre-training

Thanks to the results obtained by the open-source community, the techniques underlying the pre-training of MLLMs have become increasingly more transparent. The primary techniques in the pre-training phase of MLLMs revolve around the collection and pre-processing of multilingual data, which takes a dominant role. Pre-training objectives and strategies are also crucial to their eventual performance.

### 5.1 Data Curation

Research has demonstrated that a high-quality, de-duplicated pre-training dataset is essential for the enhanced performance of pre-trained LLMs (Young et al., 2024; Lee et al., 2022). Given that a substantial portion of pre-training data is sourced from the web, ensuring the quality and safety of this data is crucial. The process of developing such dataset for a well-performing LLM can be organized into five parts: language identification, quality filtering, safety filtering, deduplication, and up-sampling of certain domains.

Language identification is crucial for subsequent filtering methods since heuristic rule filters are typically based on the statistical information of the dataset, which can vary significantly across different languages.

For quality filtering, combining filters based on heuristic rules with machine learned filters improves data quality as they permit to take into consideration both syntactic and semantic aspects (Young et al., 2024; Bi et al., 2024; Cai et al., 2024).

Heuristic rule filters often involve filtering documents based on pre-defined block lists, such as URLs, and the statistical information of documents, including character/token/digit/symbol ratios and the frequency of repeated words/n-grams/paragraphs (Young et al., 2024; Cai et al., 2024; Mehta et al., 2024). These rules are usually designed based on observations of the data. For instance, BLOOM (Le Scao et al., 2023) defines a series of indicators to filter out low-quality data for each language, with indicators selected by fluent speakers of each language.

Learned filters typically involve task-specific trained models.
Examples include using KenLM252525https://github.com/kpu/kenlm to measure document perplexity or trained quality models to score data quality.
For instance, Cai et al. (2024) organize human annotators to label document quality and fine-tune BERT models to serve as filters.
LLaMA 3 (AI@Meta, 2024) finds that using LLaMA 2 to identify high-quality data yield excellent results, and leverages it to generate high-quality text data.

To enhance the safety of pre-training data, documents containing unsafe content must be filtered out. Like quality filters, safety filters are composed of manually designed heuristic rules and machine learned filters, targeting, for instance, personally identifiable information (PII), toxicity or unwanted words or domains (Young et al., 2024; Cai et al., 2024; Bai et al., 2023b; Touvron et al., 2023d; Team et al., 2024). As another example, (DeepSeek-AI et al., 2024b) removes also contentious content from the pre-training corpus to mitigate cultural bias in the data.

Turning to text deduplication, it significantly impacts LLM performance. Current deduplication methods include exact-match deduplication and fuzzy-match deduplication, using algorithms like MinHash and LSH (Young et al., 2024; Bi et al., 2024; Bai et al., 2023b; Team, 2023; AI@Meta, 2024). Effective deduplication ensures that redundant data does not skew the training process.

### 5.2 Pre-training Objectives

Most popular MLLMs have increasingly adopted the decoder-only architecture and causal mask attention mechanism. This focus has led to a corresponding focus in terms of pre-training objectives, moving from masked language modeling to next token prediction. Under this pre-training objective, MLLMs are trained autoregressively to predict the next token given the current tokens, which can be formulated as:

| (1) |

where stands for a sequence of tokens, represents each time step, and the parameters of the model.

Besides the next token prediction objectives, masked language modeling tasks can also be used to train LLMs. UL2 (Tay et al., 2022) proposes using a combination of denoising objectives for pre-training. It includes R-Denoiser, S-Denoiser, and X-Denoiser. R-Denoiser is the standard span corruption introduced in T5 (Raffel et al., 2020b), typically masking 15% of the input tokens in spans of 2-5 tokens. S-Denoiser is akin to prefix language modeling, partitioning the input sequence into two sub-sequences corresponding to source and target. X-Denoiser is similar to R-Denoiser but with longer spans and a higher corruption rate.

By incorporating these denoisers, UL2 pre-training objectives can be applied to any model architecture. It has been successfully applied to PaLM-2, showing promising results. OpenMoE (Xue et al., 2024) also experimented with UL2 as the pre-training objective but found that the model’s improvement slowed beyond the early training stage. Consequently, its training reverted to next token prediction.

### 5.3 Pre-training Strategies

To fully utilize computing resources during the pre-training stage, packing (Raffel et al., 2020b) is used instead of padding, as padding introduces meaningless computation. Packing involves collecting different sequences into a single document and then truncating it into multiple smaller segments of a predefined sequence length. However, this method disregards sentence semantics, potentially truncating sentences in the middle or concatenating multiple irrelevant sequences into a single segment.

Going beyond a single pre-training stage, many MLLMs incorporate a multi-stage pre-training strategy aimed at extending the context length (Cai et al., 2024; DeepSeek-AI et al., 2024a). The quadratic complexity of self-attention computation related to context length results in a significant increase in both computation and memory costs. And implementing a multi-stage pre-training with an expanded long context window has proven to be an effective solution, as this stage requires fewer tokens than the initial stage.

For example, InternLM 2 (Cai et al., 2024) dedicates 90% of total training steps to a 4096 context length, while the remaining 9% of training steps use a 32K context length.

In addition to the stage that extends context length, MiniCPM (Hu et al., 2024) proposes another pre-training strategy to enhance MLLM performance on downstream tasks, called the warmup-stable-decay learning rate scheduler (WSD LRS). The warmup and stable stages are the same as in conventional pre-training. In the decay stage, the learning rate rapidly decreases from the maximum to the minimum. The loss curve demonstrates that this strategy achieves significantly better performance compared to the original pre-training strategy. MiniCPM suggests that 10% of training steps dedicated to the decay stage is sufficient.

## 6 Multilingual Tuning

There are over 7,000 languages spoken on the planet, yet most existing LLMs are primarily English-based or have limited multilingual capabilities. This hampers access to cutting-edge Artificial Intelligence for non-English speakers and poses the risk of cultural uniformisation and the loss of linguistic diversity. To democratize LLMs across languages, a number of efforts have sought to extend tuning of MLLMs to multilingual scenarios.

As summarized in Figure 3, in this section, we start by introducing common tuning strategies to adapt the general capabilities of LLMs to specific objectives. Then, we present two common multilingual tuning strategies. The first is direct multilingual tuning, which simply expands tuning data to multiple languages; the second focuses on cross-lingual alignment during multilingual tuning, transferring capabilities from the pivot language (primarily English) to other languages. We will also discuss specific tuning methods used to enhance particular multilingual capabilities.

### 6.1 Basic Tuning Strategies

Here we introduce foundational tuning techniques for adapting the general capabilities of MLLMs to specific objectives. These techniques also serve as the basis for multilingual tuning. They are: instruction tuning, preference tuning, and continual pre-training.

#### 6.1.1 Instruction Tuning

In general, MLLMs are trained using a self-supervised learning objective, which involves predicting a token on the basis of the preceding or surrounding tokens (Brown et al., 2020b; Chowdhery et al., 2023; Zeng et al., 2021; Zhang et al., 2022; Scao et al., 2022; Rae et al., 2021; Touvron et al., 2023a; Shaham et al., 2024a). When a prompt is entered, the MLLMs generate a completion for that prompt that is contextually appropriate. However, despite the appropriateness of these completions, they may not always align with human preferences, thereby limiting the practical applications of MLLMs. To address this issue, instruction tuning has been proposed, which aims to enhance the alignment of generated responses with human preferences and has become the most widely method adopted to fine-tune MLLMs.

Instruction tuning is a technique designed to bridge the gap between the next-token prediction objective of MLLMs during pre-training and the goal of providing desired responses for humans in practical scenarios. Unlike the pre-training stage, which typically involves training the LLMs on document corpora, instruction tuning involves training the models with instructional prompts and corresponding completions.

The pioneering work on instruction tuning involves fine-tuning MLLMs on instructions paraphrased from samples of various NLP tasks, which substantially enhances the performance of MLLMs on unseen tasks (Wei et al., 2022; Sanh et al., 2022a). However, fine-tuning MLLMs solely on data from NLP tasks can result in models that excel at completing these tasks rather than effectively interacting with humans.

Notably, Ouyang et al. (2022b); Bai et al. (2022); Askell et al. (2021) fine-tuned MLLMs with instruction data that more closely align with everyday human instructions. This type of instruction data can be authored by either humans or LLMs themselves (Wang et al., 2023f; Xu et al., 2023a; Luo et al., 2023b). A critical difference between pre-training and instruction tuning is that the latter employs a prompt template that specifies the roles and tasks for MLLMs, which can be populated with samples to conduct the tuning process. Additionally, while pre-training typically computes loss over all tokens in a training sample, instruction tuning can ignore the loss of prompt tokens (Li et al., 2023g). However, recent studies suggest that optimizing the loss over both prompt tokens and response tokens yields superior performance compared to optimizing the loss over response tokens alone (Shi et al., 2024d; Huerta-Enochian & Ko, 2024).

#### 6.1.2 Preference Tuning

In contrast to instruction tuning, which involves training large language models with instructional prompts and their corresponding completions (Zhang et al., 2023d), preference tuning typically employs preference data. This data is generated by annotating the more preferred completion between two options (Bai et al., 2022) or by ranking completions according to human preference (Ouyang et al., 2022b). Currently, two primary techniques are used for preference tuning: Reinforcement Learning from Human Feedback (RLHF) (Ziegler et al., 2019; Stiennon et al., 2020; Nakano et al., 2021; Ouyang et al., 2022b) and Direct Preference Optimization (DPO) (Rafailov et al., 2023).

RLHF comprises three essential steps for training LLMs: (1) training LLMs with instruction tuning, also known as supervised fine-tuning; (2) training a reward model using preference data to predict human preferences; and (3) applying reinforcement learning to further train MLLMs from step 1, thereby enhancing the alignment between MLLMs and human preferences.

In step 2, the trained reward model produces a scalar reward to quantify the degree of human preference for given completions, where a higher reward denotes greater alignment with human preferences. The parameters of the reward model are optimized based on the assumption that the preference data used for training are sampled from a preference function that adheres to the Bradley-Terry model (Bradley & Terry, 1952).

Let denote the prompt, and let and denote the “win” and “lose” responses, respectively, where “win” and “lose” represent the preferred and less preferred responses as chosen by humans. Let denote the preference function that maps a prompt and its corresponding completion to a scalar value, reflecting human preferences. The sigmoid function is denoted by . The notation indicates that is more preferred than . According to the Bradley-Terry model, the probability that is preferred over can be expressed as follows:

| (2) |

Consequently, the reward model can be trained to minimize the following objective:

| (3) |

All parameters of the reward model, with the exception of the final embedding layer, can be initialized from either pre-trained MLLMs (Cui et al., 2023a; Gooding & Mansoor, 2023; Kirk et al., 2023) or instruction-tuned LLMs (Touvron et al., 2023c). For MLLMs, the final embedding layer maps the hidden states of each token to the probability distribution over the vocabulary. However, to adapt the MLLMs into a reward model, the final embedding layer is replaced with a randomly initialized linear layer. This new layer maps the hidden states of each token to a scalar reward.

In step 3, reinforcement learning techniques, such as Proximal Policy Optimization (PPO) (Schulman et al., 2017), are employed to train the LLMs to maximize the following objective:

| (4) |

where is the corresponding completion for prompt , is the reward model trained in step 2, is the trained policy (trained LLM) and is a reference MLLM, which can be the instruction-tuned MLLMs. The hyperparameter can be tuned to control the strength of the restriction on the distribution divergence between the trained policy and the reference LLM, thereby ensuring the trained policy remains close to the reference MLLM.

Although RLHF has demonstrated impressive performance in aligning MLLMs with human preferences, achieving stable training in practice remains challenging and computationally expensive. To address these challenges, DPO has been proposed. While the objective of DPO is derived from Eq. 4, which is also the objective of RLHF, DPO circumvents the need to explicitly train a reward model by expressing human preferences in terms of the optimal policy :

| (5) |

Consequently, the training objective of DPO can be derived from Eq. 5 by minimizing the following objective:

| (6) |

Apart from RLHF and DPO, there has been a surge of preference tuning approaches proposed in recent years, including IPO (Azar et al., 2024), PRO (Song et al., 2024), RRHF (Yuan et al., 2023b), KTO (Ethayarajh et al., 2024), and SLiC-HF (Zhao et al., 2023), among others (Wu et al., 2024b; Meng et al., 2024; Xiong et al., 2024).

#### 6.1.3 Continual Pre-training

Pre-trained MLLMs can be further adapted to specific objectives through methods such as instruction tuning and preference tuning. Continual pre-training serves as an intermediary step between the initial pre-training and task-specific fine-tuning, adapting MLLMs’ general knowledge to specific domains or integrating entirely new knowledge (Shi et al., 2024a; Mu et al., 2024b; Li et al., 2024f). This process enhances the adaptability of MLLMs to specific tasks, thereby improving task performance.

The training objective during continual pre-training is next-token prediction, similar to that in the pre-training stage. However, a significant challenge in continual pre-training is catastrophic forgetting (Shi et al., 2024a), which occurs when MLLMs are optimized on the target data that are domain-specific or entirely new compared to the pre-training data. In essence, this phenomenon arises from the distributional differences between the target data and the original pre-training data. Catastrophic forgetting is a critical issue because it not only erases previously acquired knowledge but also hinders effective knowledge transfer.

To mitigate catastrophic forgetting, common approaches include replay, regularization, and architecture-based methods. Specifically, replay-based methods involve incorporating samples from the original training data alongside new target data (Shi et al., 2024b; Chaudhry et al., 2019; Riemer et al., 2019; Buzzega et al., 2020; Sarfraz et al., 2023; Bang et al., 2021). Regularization-based methods impose constraints to minimize discrepancies between updated and original model parameters (Zhou & Cao, 2021; Aljundi et al., 2018), while architecture-based methods primarily focus on parameter-efficient fine-tuning techniques (Dettmers et al., 2023; Xu et al., 2023f; Lester et al., 2021; Li & Liang, 2021; Yang et al., 2024b; Wu et al., 2024a; Wang et al., 2023e).

### 6.2 Direct Multilingual Tuning

To democratize MLLMs across languages, many studies directly extend the tuning of MLLMs to multilingual contexts. The majority of these efforts involve direct multilingual instruction tuning, enabling MLLMs to follow instructions in multiple languages (Lai et al., 2023b; Li et al., 2023d; Muennighoff et al., 2023b; Chen et al., 2023f; Shaham et al., 2024b; Wei et al., 2023; Nguyen et al., 2023b; Ji & Chen, 2024; Kew et al., 2023; Chen et al., 2024b; Shen et al., 2024b; Chirkova & Nikoulina, 2024; Üstün et al., 2024; Weber et al., 2024), while others focus on multilingual preference tuning, aligning MLLMs with human preferences across multiple languages (Lai et al., 2023b; Wu et al., 2024d; Shen et al., 2024b). These studies adopt significantly different approaches to collecting multilingual tuning data. Additionally, the research question of how to effectively stimulate cross-lingual transfer during the multilingual tuning process is widely discussed.

#### 6.2.1 Multilingual Tuning Data Collection

Expanding the tuning of MLLMs into multilingual scenarios necessitates the collection of multilingual tuning data. Here, we’ll outline this process separately for multilingual instruction tuning and multilingual preference tuning.

##### Multilingual Instruction Tuning

Combining P3 (Sanh et al., 2022b) with 30 other multilingual datasets, Muennighoff et al. (2023b) created the xP3 dataset, which encompasses 46 languages. They subsequently translated the English prompts in xP3 into non-English languages using the Google Cloud API, resulting in the xP3-mt dataset.

Additionally, Bactrian-X (Li et al., 2023d) collected 67K English instructions from Alpaca (Taori et al., 2023) and Dolly (Conover et al., 2023a), which were then translated into 51 languages using Google Translate. They further employed ChatGPT to generate multilingual responses to mitigate issues such as “translationese” and non-native answer styles.

In addition to the strategy of first translating instructions and then generating language-specific responses, Phoenix (Chen et al., 2023f) included additional data by directly translating instructions and responses using GPT-4. Their multilingual dataset was also derived from English Alpaca and expanded to over 40 languages.

Similarly, based on Alpaca, Polylm (Wei et al., 2023) leveraged 175 English task seeds from it and iteratively collected and filtered samples for 11 languages using a Self-Instruct (Wang et al., 2023f) method.

SeaLLMs (Nguyen et al., 2023b), in turn, focused on nine Southeast Asian languages. To address the scarcity of Southeast Asian data (3.3%) in their instruction dataset, they adopted a hybrid training strategy, merging the instruction data with multilingual pre-training data to achieve more balanced language ratios.

In contrast, Üstün et al. (2024) considered a broader range of languages, 101 in total. Various methods were employed to construct large-scale multilingual datasets, including aggregating and refining multilingual templates, as well as carefully selecting elusive human annotations from fluent speakers of different languages. Additionally, data augmentation strategies, such as machine translation and generating synthetic data combined with translation, were also utilized.

##### Multilingual Preference Tuning

To gather multilingual preference data, Okapi (Lai et al., 2023b) employed a series of procedures. Initially, they expanded the 52K English Alpaca dataset to 158K using Self-Instruct (Wang et al., 2023f) techniques.

Subsequently, ChatGPT was used to translate both instructions and responses into 26 languages. After translation, the 52K dataset for each language was first used to perform instruction tuning on MLLMs. Then, 42K instructions were fed into the instruction-tuned model to sample responses. Finally, ChatGPT was employed to translate the obtained responses back into English and rank preferences, resulting in the final preference data.

It’s worth noting that they conducted this process separately for each language. Afterward, they used the collected preference data to train a reward model for each language, and further trained a preference-tuned model for each language using the remaining 64K instructions per language.

#### 6.2.2 Cross-Lingual Transfer Elicitation

In the process of direct multilingual tuning, effectively stimulating cross-lingual transfer under limited resources is an important research question (Xu et al., 2023e). Here, we introduce how this challenge is addressed in terms of multilingual instruction tuning and multilingual preference tuning, respectively. Overall, these studies indicate that in direct multilingual tuning, cross-lingual transfer is influenced by factors such as the selection of source languages, linguistic relationships, the number of languages involved, and the scale of multilingual instruction data (Razumovskaia et al., 2024; Faisal & Anastasopoulos, 2024; Kim et al., 2024b).

##### Multilingual Instruction Tuning

First, several studies suggest that simply performing monolingual instruction tuning on a single language can lead to certain degrees of zero-shot cross-lingual transfer (Muennighoff et al., 2023b; Shaham et al., 2024b; Chirkova & Nikoulina, 2024). Specifically, Shaham et al. (2024b) found that various source languages yield varying levels of cross-lingual transfer, with English, Italian, and Spanish demonstrating superior results in their experimental settings, while Chirkova & Nikoulina (2024) noted that adjusting instruction tuning hyperparameters for multilinguality and using sufficient instruction data can improve zero-shot cross-lingual transfer.

Furthermore, assuming multilingual instruction data is available, several studies explore key factors to promote cross-lingual transfer, such as the number of languages and the volume of instruction data. Comparing multilingual and monolingual instruction tuning, Shaham et al. (2024b) found that replacing even a small number of monolingual (English) training samples (as few as 40) with multilingual ones significantly improves performance on these languages. This finding aligns with other research (Chen et al., 2024b; Weber et al., 2024), emphasizing the advantages of multilingual instruction tuning over monolingual instruction tuning, especially in resource-constrained settings. Additionally, considering the amount of multilingual instruction data, Weber et al. (2024) suggested that sufficient instruction data is essential for achieving better multilingual performance. It is worth noting, however, that this stands in contrast to the Superficial Alignment Hypothesis (Zhou et al., 2023a).

Regarding the number of languages, it is suggested that including just a few languages (e.g., 2-4) can enhance cross-lingual transfer, especially for languages not encountered during pre-training (Shaham et al., 2024b; Kew et al., 2023). However, Ji & Chen (2024) suggested that increasing the number of languages may further enhance multilingual performance. They also explored other factors such as language similarity and concluded that the optimal number of languages depends on both language similarity and downstream evaluation.

##### Multilingual Preference Tuning

While the aforementioned studies explore cross-lingual transfer in direct multilingual instruction tuning, other works address it in multilingual preference tuning (Chai et al., 2024a). Specifically, Wu et al. (2024d) found that reward models trained on source languages can be effectively utilized for cross-lingual preference tuning in target languages. They even observed cases where using reward models from source languages outperforms those trained on target languages, possibly due to a regularization effect. Additionally, high-resource languages (e.g., English) tend to be more effective in inducing cross-lingual preference transfer than low-resource languages (Wu et al., 2024d; Shen et al., 2024b).

### 6.3 Multilingual Tuning Augmented by Cross-Lingual Alignment

While direct multilingual tuning can enhance the multilingual capabilities of MLLMs to some extent, many approaches augment multilingual tuning with cross-lingual alignment to further improve cross-lingual transfer (Li et al., 2024c; Peng & Søgaard, 2024). They typically use English, the best resourced language, as the pivot language, aligning the understanding, reasoning, and generation capabilities of MLLMs in non-English languages with those in English.

Based on how cross-lingual signals are incorporated to achieve alignment, these works can be classified into two categories: Translation-Assisted Tuning and Cross-Lingual Tuning.

Specifically, Translation-Assisted Tuning defines auxiliary translation-related tasks and combines it with original multilingual tuning, while Cross-Lingual Tuning transforms multilingual tuning tasks into cross-lingual forms without explicitly relying on translation task.

It’s worth noting that multilingual tuning here generally refers to multilingual instruction tuning, as most approaches perform cross-lingual alignment at this stage. The only exception is the work done by She et al. (2024), which models cross-lingual alignment as a preference optimization problem.

#### 6.3.1 Translation-Assisted Tuning

Assuming the internal translation capability of MLLMs can facilitate cross-lingual alignment, Translation-Assisted Tuning methods define auxiliary translation-related tasks and combine them with original multilingual tuning. Such tasks involve machine translation (Ranaldi et al., 2023; Zhu et al., 2023b; 2024e; 2024f) or other variations (Zhang et al., 2023c; Cahyawijaya et al., 2023).

The most straightforward approach is to convert parallel data into translation instruction format and incorporate them to the original multilingual instruction data. While Ranaldi et al. (2023) focused on bilingual scenarios, Zhu et al. (2023b) initially explored the correlation between the scale of translation instructions and translation performance through bilingual training, which subsequently served as a reference for determining the allocation of translation instructions in multilingual settings.

Specifically, for multilingual reasoning tasks, Zhu et al. (2024f) proposed translation instruction tuning of parallel reasoning questions (Question Alignment) before English instruction tuning for reasoning tasks, thereby facilitating the transfer of English reasoning capabilities of MLLMs to non-English languages.

Similarly, starting with Question Alignment, Zhu et al. (2024e) further integrated general multilingual translation instructions (from English to non-English) with English instruction tuning for reasoning tasks, enhancing the ability of MLLMs to generate multilingual reasoning outputs.

Other studies explore variations of traditional machine translation tasks and transform them into instruction tuning formats.

Specifically, Zhang et al. (2023c) extended translation instruction tuning to a multi-turn interactive setting, aiming to simultaneously enhance the cross-lingual alignment and instruction-following capabilities of MLLMs.

Additionally, Cahyawijaya et al. (2023) introduced a cross-lingual semantic similarity task, where LLMs are trained to determine the semantic relationship between two parallel sentences. They also introduced a bilingual denoising task, training MLLMs to reconstruct the target side of input parallel sentences that have been noised.

#### 6.3.2 Cross-Lingual Tuning

Without relying on auxiliary translation-related tasks, the approaches in Cross-Lingual Tuning reconstruct multilingual instruction tuning into a cross-lingual form, enhancing the cross-lingual alignment of MLLMs during instruction learning in a more integrated manner (Si et al., 2024).

In (Chai et al., 2024b), cross-lingual signals are introduced by specifying instructions in non-English languages and responses in English. Certain tokens in the non-English instructions may be replaced with English to construct code-switched instructions.

In (Zhang et al., 2023f) and (Upadhayay & Behzadan, 2023), MLLMs were trained to respond to non-English instructions by first thinking in English and then responding in a non-English language. When the input is a non-English instruction, MLLMs are trained to sequentially output the English translation of the instruction, an English response, and finally a non-English response. The Random Online CoT introduced by Chai et al. (2024b) serves a similar purpose.

Other works promote cross-lingual alignment by enhancing the cross-lingual consistency of model responses. When responses are required to be in English, the Cross-lingual Distillation method introduced by Chai et al. (2024b) minimizes the difference between generated English responses when inputs are in different languages. For multilingual responses, She et al. (2024) employed high-performing multilingual translation models (e.g., NLLB-600M-distilled) to compute cross-lingual consistency between non-English and English responses. They further framed cross-lingual alignment as a preference optimization problem, utilizing the computed cross-lingual consistency as rewards for preference tuning.

### 6.4 Enhancement of Specific Multilingual Abilities

The direct multilingual tuning and cross-lingual alignment discussed above serve as foundational strategies for enhancing MLLMs’ capabilities in multilingual instruction following, preference alignment and complex reasoning (Zhao et al., 2024e). Beyond this, there is interest in more multilingual abilities of MLLMs, including diverse capabilities of understanding and generation. Those specific multilingual abilities can be enhanced using particular techniques, which we will address here.

We will focus on three of such abilities: adaptation to new languages, machine translation and cultural adaptation, as they inherently involve multiple languages or cultures. Adaptation to new languages primarily involves adapting English-centric MLLMs to other languages; machine translation focuses on translating between different languages; and cultural adaptation addresses the cultural diversity of our society, with language considered one of the proxies of culture.

#### 6.4.1 Adaptation to New Languages

Although the techniques employed for adaptation to new languages are general, primarily focusing on vocabulary expansion and continual pre-training, the target languages for adaptation vary across different works (Cui et al., 2023c; Kim et al., 2024a; Fujii et al., 2024; Lin et al., 2024; Nguyen et al., 2023a).

For instance, Cui et al. (2023c) aimed to extend English-based LLMs to Chinese. To enhance the efficiency of MLLMs in encoding and decoding Chinese text, they trained an additional tokenizer on Chinese corpora before conducting continual pre-training, expanding the original vocabulary of MLLMs.

Other studies extend the vocabulary and perform continual pre-training on different target languages, such as Korean (Kim et al., 2024a), Japanese (Fujii et al., 2024), or multiple languages (Lin et al., 2024; Nguyen et al., 2023a), and so forth.

Additionally, Kim et al. (2024a) proposed a seven-stage continuous pre-training strategy to enable LLMs to better learn new token representations.

Adaptation to new languages presents a significant challenge when data for the target languages are not available.

In one such case, to enable monolingual instruction-tuned MLLMs to perform multilingual mathematical reasoning tasks, Yoon et al. (2024) employed an external multilingual encoder and initialized a trainable linear layer as a bridge between the encoder and the MLLMs. They enhanced them with multilingual understanding capabilities by aligning the representation space of this encoder and the MLLMs.

In another case, considering the high computational and data resource costs of the strategy of first expanding the vocabulary and then performing continual pre-training, Husain et al. (2024) advocated for more efficient reuse of the original vocabulary rather than for expanding it. Given that English-centric vocabularies primarily consist of Roman letters, they proposed romanizing the text of new languages. Factors considered in this romanization scheme include the similarity between the romanized text and the typical writing style, compatibility with the original LLM tokenizer, and the lossiness of the conversion process.

Other studies conduct comparative analyses of key techniques for adapting to new languages (Yong et al., 2023b; Zhao et al., 2024a; Cui & Yao, 2024).

Specifically, Yong et al. (2023b) compared three languages adaptation strategies, viz. continual pre-training, MAD-X (Pfeiffer et al., 2020), and (Liu et al., 2022), and experimentally found that adapter-based methods are more effective than continual pre-training for MLLMs.

On the other hand, Zhao et al. (2024a) investigated the suitability of vocabulary expansion, concluding that such an extension may not be advantageous when the scale of the training dataset is below billions of tokens.

Similar to (Zhao et al., 2024a), Cui & Yao (2024) suggested that while expanding the vocabulary for new languages may accelerate encoding efficiency, it may not necessarily improve performance. They also gathered empirical evidence that training from a base model rather than an instruction-tuned model could result in better languages adaptation.

#### 6.4.2 Machine Translation

Many works use MLLMs to implements machine translation task (Zhu et al., 2024b; d; Cui et al., 2024; Zhu et al., 2024c; Chen et al., 2024a; Zhang et al., 2024a). To adapt MLLMs to machine translation, Yang et al. (2023c) utilized extensive parallel corpora (exceeding 300M) for fine-tuning.

However, it has been observed that using excessive parallel data (e.g., 5M or 20M) for translation task training on well pre-trained MLLMs can have adverse effects, potentially due to catastrophic forgetting (Xu et al., 2023b). To address this issue, Xu et al. (2023b) proposed a two-stage fine-tuning approach. Initially, they conducted continual pre-training on the languages involved in the translation task to enhance MLLMs’ proficiency in those languages. Then, they performed translation instruction tuning on a small scale (e.g., 58K) but with high-quality parallel data.

Other studies (Li et al., 2023e; Zhu et al., 2024a) conducted detailed analyses on multilingual translation instruction tuning, taking into account factors such as language similarity, the volume of pre-training and instruction tuning data, and the translation directions.

Although commonly used to enhance MLLMs’ translation capabilities, translation instruction tuning is regarded as having certain limitations. On the one hand, the quality of translation instruction data can restrict the performance of MLLMs (Xu et al., 2024a). On the other hand, MLLMs fine-tuned with translation instructions may overlook specific contextual knowledge, forget instructions, and consequently encounter issues such as hallucinations, over-translation or translation omissions (Wu et al., 2024c; Zeng et al., 2024; Chen et al., 2023a).

To address these challenges, some efforts have opted for preference tuning to further enhance the translation capabilities of MLLMs.

To build preference pairs, Xu et al. (2024a) employed different MLLMs to sample translation results, which were then scored by reference-free evaluation models.

To efficiently collect high-quality preference data, Xu et al. (2024b) aligned multilingual versions of books to acquire human translations, forming preference pairs alongside machine-generated translations.

Additionally, Wu et al. (2024c) utilized external word aligners to annotate the degree of word alignment for translation candidates, while Zeng et al. (2024) employed output comparison and preference comparison strategies to construct preference data.

Instead of performing preference tuning, Chen et al. (2023a) proposed to strengthen the global representations of instructions, aiming to mitigate MLLMs’ tendency to forget instructions.

The lack of cross-lingual alignment may also affect machine translation. To enhance cross-lingual alignment, Gao et al. (2024) introduced XConST, which employs Kullback-Leibler (KL) regularization on semantically equivalent parallel sentences.

Furthermore, Mao & Yu (2024) initially utilized external tools to extract statistical word alignment signals and then trained MLLMs to discern these signals.

Fu et al. (2024) proposed Relay Decoding, a method aligning two LLMs supporting the source and target languages by training a straightforward mapping layer.

Additionally, Yang et al. (2024a) proposed m3P, a multimodal neural machine translation framework to bridge language disparities through universal visual features.

#### 6.4.3 Cultural Adaptation

As research continues to advance MLLMs in their capabilities, assessing and enriching their cultural and value diversity becomes increasingly crucial, especially considering their extensive utilization across diverse populations worldwide (Sorensen et al., 2024).

Significant efforts have been dedicated to exploring the multicultural knowledge embedded within LLMs, as well as the extent of their cultural adaptation and alignment (AlKhamissi et al., 2024; Naous et al., 2023; Liu et al., 2023; Arora et al., 2022; Cao et al., 2023b; Masoud et al., 2023; Keleg & Magdy, 2023; Palta & Rudinger, 2023; Shwartz, 2022; Ramezani & Xu, 2023; Wang et al., 2023c; Kabra et al., 2023; Ma et al., 2022; Kovac et al., 2023; Fung et al., 2023; Kirk et al., 2024; Fung et al., 2024; Shen et al., 2024c; Yin et al., 2022; Shi et al., 2024c; Pistilli et al., 2024; Li et al., 2024d; Yao et al., 2024; Rao et al., 2024; Li et al., 2024b; Zhao et al., 2024b; Chiu et al., 2024; Huang & Yang, 2023; Zhou et al., 2024; Cao et al., 2023a; 2024).

However, efforts to fine-tune MLLMs for cultural adaptation are relatively sparse, or limited to English.

Recently, Li et al. (2024a) introduced CultureLLM, a methodology aimed at instilling MLLMs with cultural differences. First, they collected seed data from the World Values Survey (Survey, 2022), spanning 9 cultures, and utilized GPT-4 for data augmentation. Subsequently, they fine-tuned LLMs on the augmented data, creating culture-specific models for each culture alongside a unified model.

Furthermore, Xu et al. (2024c) proposed CultureSPA, a framework that achieves pluralistic cultural alignment in MLLMs by leveraging their internal cultural knowledge. This framework involves generating diverse questions from seed prompts, yielding both culture-unaware and culture-aware MLLM outputs, collecting culture-related QA pairs and conducting culture-joint and specific SFT. They validated the effectiveness of the method across 18 cultures spanning five continents.

With a proposed technique for extracting culture-related instruction data from unstructured data, Mukherjee et al. (2024) performed instruction tuning on this data to enhance the cultural reasoning abilities of MLLMs.

Despite recognizing that language often acts as a proxy for culture (Hershcovich et al., 2022), these works employed English for data collection, leaving exploration in multilingual contexts for future research.

Other works analyze the relationship between cultural adaptation and multilingual instruction tuning (Choenni et al., 2024; Wang et al., 2024a).

Specifically, Choenni et al. (2024) conducted multilingual instruction tuning using the translated Alpaca dataset, investigating the influence of language-specific instruction tuning and pre-training data on cultural adaptation, as well as the optimal method to elicit cultural knowledge from MLLMs.

In turn, Wang et al. (2024a) explored the impact of selected languages and data sources on the shift of cultural values during model fine-tuning.

## 7 Multilingual Evaluation

The evaluation of MLLMs are crucial to have a better understanding towards their performance and capabilities, being also important to assess human-values compliance and to address safety concerns (Zhang et al., 2024d; Park et al., 2024; Shen et al., 2024a; Doddapaneni et al., 2024). In multilingual contexts, the evaluation should be broaden towards diverse languages, where the performance towards each language should be understandable. In this section, we discuss the multilingual evaluation of MLLMs, focusing on tokenizers, benchmarks and datasets, multilingualism of MLLMs, and MLLMs as multilingual evaluators. The framework in this section is shown in Figure 4.

### 7.1 Multilingual Tokenizer Evaluation

A tokenizer plays an important role for supporting multilingual processes in LLMs, as tokenizers separate sentences into tokens and map them into numerical ids that are the input for MLLMs. Tokenizers that supports multilingual tokens are crucial to enhance the performance of MLLMs.

The fertility of a tokenizer is a metric aimed at assessing its quality, which is defined as the average number of sub-words produced per tokenized word (Rust et al., 2021). A lower fertility score indicates a better quality tokenizer. Fertility is also known as the tokenizer’s compression rate (Xue et al., 2022).

Fertility tests have been conducted to evaluate tokenizers on multilingual sentences across various MLLMs, including BLOOM (Scao et al., 2022) and OpenAI models (Ahuja et al., 2023a). Based on their findings, the fertility score of OpenAI tokenizers is higher in low-resource languages compared to BLOOM’s tokenizer.

Another way to evaluate a tokenizer is using parity, which is introduced in (Petrov et al., 2023) and it is motivated by the unequal treatment of tokenization across different languages. For instance, a Japanese kanji character can be tokenized into three tokens in GPT-2. To address this issue, the concept of tokenizer parity is introduced to evaluate how fairly tokenizers treat equivalent sentences in different languages. A tokenizer achieves parity when the ratio of the tokenization results for a sentence in language A compared to a sentence in language B is almost equal to 1.

A comprehensive evaluation of tokenizers was undertaken by Ali et al. (2023), dividing the assessment into intrinsic and extrinsic evaluations.

The intrinsic evaluation focuses solely on metrics like fertility and parity.

The extrinsic evaluation evaluates the impact of the tokenizer on the performance of LLMs, with respect to downstream tasks performance and computational costs.

The research found that tokenizers trained with a balanced share across languages achieve better fertility and parity scores. The impact of worst, higher fertility will increase computational costs.

### 7.2 Multilingual Evaluation Benchmarks and Datasets

| Name | #Languages | Language Family | Type |
| XNLI (Conneau et al., 2018) | 15 | Indo-European, Turkic, Afro-Asiatic, Austronesian, Austro-Asiatic, | Natural Language Inference |
| Tai-Kadai, Sino-Tibetan, Atlantic-Congo | |||
| Indic-XNLI (Aggarwal et al., 2022) | 11 | Indo-European, Dravidian | Natural Language Inference |
| GlueCoS (Khanuja et al., 2020) | 3 | Indo-European, Dravidian | Language Identification, POS Tagging, |
| NER, Sentiment Analysis, QA, NLI | |||
| XCOPA (Ponti et al., 2020) | 11 | Uralic, Indo-European, Austronesian, Quechuan, Atlantic-Congo, | Commonsense Reasoning |
| Dravidian, Kra-Dai, Turkic, Sino-Tibetan, Austro-Asiatic | |||
| XStoryCloze (Lin et al., 2022) | 10 | Indo-European, Sino-Tibetan, Afro-Asiatic, Austronesian, | Reasoning |
| Dravidian, Atlantic-Congo, Basque | |||
| PAWS-X (Yang et al., 2019) | 6 | Indo-European, Sino-Tibetan, Japonic, Koreanic | Paraphrase Identification |
| EN-ES-CS (Vilares et al., 2016) | 2 | Indo-European | Sentiment Analysis |
| XQuAD (Artetxe et al., 2020) | 11 | Indo-European, Sino-Tibetan, Austro-Asiatic | Question Answering |
| MLQA (Lewis et al., 2020) | 7 | Indo-European, Afro-Asiatic, Austro-Asiatic, Sino-Tibetan | Question Answering |
| TyDiQA-GOLDP (Clark et al., 2020) | 11 | Indo-European, Afro-Asiatic, Uralic, Japonic, | Question Answering |
| Atlantic-Congo, Koreanic, Dravidian, Kra-Dai | |||
| IndicQA (Doddapaneni et al., 2022) | 11 | Indo-European, Dravidian | Question Answering |
| PAN-X/WikiANN (Pan et al., 2017) | Indo-European, Afro-Asiatic, Turkic, Sino-Tibetan, Austronesian, | Named Entity Recognition, | |
| Uralic, Atlantic-Congo, Kartvelian, Japonic, Dravidian, | |||
| Koreanic, Kra-Dai, Nilo-Saharan, Austro-Asiatic, Mongolic | |||
| UD v2.2 (Nivre et al., 2018) | 71 | Indo-European, Afro-Asiatic, Basque, Sino-Tibetan, Uralic, Austronesian, | Sequence Labeling |
| Japonic, Turkic, Koreanic, Atlantic-Congo, Dravidian, Kra-Dai, Austro-Asiatic | |||
| XL-Sum (Hasan et al., 2021) | 47 | Afro-Asiatic, Turkic, Austro-Asiatic, Sino-Tibetan, Indo-European, Japonic, | Summarization |
| Koreanic, Dravidian, Atlantic-Congo, Austronesian, Austro-Asiatic, Kra-Dai | |||
| Jigsaw | 6 | Indo-European, Turkic | Toxic Classification |
| Wino-MT (Stanovsky et al., 2019) | 8 | Indo-European, Afro-Asiatic | Gender Bias in Translation |
| Belebele (Bandarkar et al., 2023) | 122 | Afro-Asiatic, Indo-European, Turkic, Mande, Sino-Tibetan, | Question Answering |
| Austronesian, Uralic, Basque, Atlantic-Congo, Tupian, Japonic, Dravidian, | |||
| Kartvelian, Mongolic, Austroasiatic, Koreanic, Kra-Dai, Nilo-Saharan | |||
| AfriQA (Ogundepo et al., 2023) | 10 | Atlantic-Congo, Afro-Asiatic | Question Answering |
| XRiSAWOZ (Moradshahi et al., 2023) | 5 | Indo-European, Koreanic, Sino-Tibetan | Dialogue |
| IN22 (Gala et al., 2023) | 22 | Indo-European, Austronesian, Dravidian | Translation |
| MaRVL (Liu et al., 2021) | 5 | Austronesia, Sino-Tibetan, Atlantic-Congo, Dravidian, Turkic | Image-Text Reasoning |
| XM-3600 (Thapliyal et al., 2022) | 36 | Afro-Asiatic, Indo-European, Uralic, Austronesian, Japonic, Koreanic, Quechuan | Image Captioning |
| MultiCoNER (Malmasi et al., 2022) | 11 | Sino-Tibetan, Indo-European, Koreanic, Turkic | NER |
| SMiLER (Seganti et al., 2021) | 14 | Koreanic, Indo-European, Afro-Asiatic | Relation Extraction |
| X-CSQA (Lin et al., 2021) | 15 | Indo-European, Japonic, Sino-Tibetan, Afro-Asiatic, Atlantic-Congo, Austro-Asiatic | Commonsense Reasoning |
| Wikipedia Cloze QA (Kakwani et al., 2020) | 11 | Indo-European, Dravidian | Question Answering |
| Flores-101 (Goyal et al., 2022) | 101 | Indo-European, Afro-Asiatic, Turkic, Sino-Tibetan, Austronesian, Uralic, | Translation |
| Atlantic-Congo, Kartvelian, Japonic, Dravidian, Koreanic, | |||
| Kra-Dai, Nilo-Saharan, Austro-Asiatic, Mongolic | |||
| M3Exam (Zhang et al., 2023e) | 9 | Indo-European, Sino-Tibetan, Austro-Asiatic, Kra-Dai, Atlantic-Congo | Question Answering |
| MultiQ (Holtermann et al., 2024) | 137 | Afro-Asiatic, Altaic, Austro-Asiatic, Austronesian, Aymaran, Basque, | Question Answering |
| Dravidian, Indo-European, Kra-Dai, Japonic, Koreanic, Kartvelian, Mande, | |||
| Atlantic-Congo, Quechuan, Sino-Tibetan, Tupian, Uralic, Turkic | |||
| SEAHORSE (Clark et al., 2020) | 6 | Indo-European, Turkic, Austro-Asiatic | Summarization |
| xDial-Eval (Zhang et al., 2023a) | 9 | Sino-Tibetan, Indo-European, Japonic, Koreanic, Afro-Asiatic | Dialogue |
| RTP-LX (de Wynter et al., 2024) | 27 | Afro-Asiatic, Indo-European, Austronesian, Kra-Dai, Atlantic-Congo, | Toxicity |
| Sino-Tibetan, Japonic, Koreanic, Turkic, Uralic | Toxicity | ||
| PolygloToxicityPrompts (Jain et al., 2024) | 17 | Afro-Asiatic, Sino-Tibetan, Indo-European, Austronesian, Japonic, Koreanic | Toxicity |
| XSAFETY (Wang et al., 2023d) | 10 | Indo-European, Sino-Tibetan, Afro-Asiatic, Japonic | Safety |
| MultiJail (Deng et al., 2023) | 9 | Sino-Tibetan, Indo-European, Austro-Asiatic, Afro-Asiatic, | Jailbreaking |
| Koreanic, Kra-Dai, Atlantic-Congo, Austronesian |

The evaluation of MLLMs should cover various dimensions, including accuracy, alignment with human values and safety. In the context of multilingual evaluation, these dimensions should be evaluated for each language, from high-resource to low-resource languages. In this section, we divide the evaluation benchmarks of MLLMs into a holistic evaluation and a task-specific, alignment and safety evaluation. We summarize the available datasets in Table 3.

#### 7.2.1 Multilingual Holistic Evaluation

Several studies have conducted comprehensive multilingual evaluations of MLLMs, covering diverse languages and tasks. Ahuja et al. (2023a) proposed MEGA, the first benchmark for evaluating generative AI in a multilingual context. This benchmark covers five categories of NLP tasks: classification, question answering, sequence labeling, natural language generation, and responsible AI. These categories are represented by 16 datasets encompassing 70 languages.

Concerning the classification task, the datasets used are XNLI (Conneau et al., 2018), Indic-XNLI (Aggarwal et al., 2022), GLUECos NLI (Khanuja et al., 2020), XCOPA (Ponti et al., 2020), XStoryCloze (Lin et al., 2022), PAWS-X (Yang et al., 2019), and EN-ES-CS (Vilares et al., 2016).

For the question answering (QA) task, the dataset used for this task are XQuAD (Artetxe et al., 2020), MLQA (Lewis et al., 2020), TyDiQA-GoldP (Clark et al., 2020), and IndicQA (Doddapaneni et al., 2022).

For sequence labeling task, PAN-X (Pan et al., 2017) and UDPOS (Nivre et al., 2018) datasets were used.

For the natural language generation, XL-Sum (Hasan et al., 2021) dataset was resorted to.

For responsible AI evaluation, Jigsaw (Ian Kivlichan, 2020) and WinoMT (Stanovsky et al., 2019) datasets were used.

In the next version, MEGAVERSE (Ahuja et al., 2023b) was distributed, an expanded version of MEGA multilingual evaluation benchmark. This benchmark was expanded to 22 datasets covering 81 languages.

The additional datasets include Belebele (Bandarkar et al., 2023), a multiple choice machine reading comprehension in 122 languages; AfriQA (Ogundepo et al., 2023), a question answering dataset for 10 African languages; XRiSAWOZ (Moradshahi et al., 2023), a task oriented dialogue modeling dataset originally in Chinese and translated into English, Hindi, French, Korean and English-Hindi code-mixed setting; IN22 (Gala et al., 2023), a translation benchmark for all 22 scheduled Indic languages; MaRVL (Liu et al., 2021), a multicultural reasoning over vision and language dataset in 5 distinct languages, comprises of image and its caption; XM-3600 (Thapliyal et al., 2022), a multilingual image captioning dataset consisting of 3600 geographically diverse images directly captioned in 36 different languages.

Another work explored the multilingual evaluation of ChatGPT (Lai et al., 2023a). This evaluation was conducted for 7 diverse NLP tasks in 37 diverse languages.

In part-of-speech (POS) tagging task, XGLUE-POS (Liang et al., 2020) dataset was used, which covers 18 languages. For named entity recognition (NER) task, MultiCoNER (Malmasi et al., 2022) dataset that supports 11 languages was resorted to. For relation extraction (RE) task, SMiLER (Seganti et al., 2021) dataset contains 14 languages was used. For natural language inference (NLI) and QA task, the dataset used is the same as MEGA benchmark, the XNLI and XQuAD dataset. For common sense reasoning (CSR) task, two datasets were included: X-CSQA (Talmor et al., 2019; Lin et al., 2021) in English and the translation into 15 languages, and Wikipedia Cloze QA from IndicNLPSuite (Kakwani et al., 2020) in 11 low-resource Indian languages.

#### 7.2.2 Multilingual Task-Specific Evaluation

LLMs are able to solve diverse kind of tasks, from question answering to dialogue generation and including translation, among many others. The trained MLLMs have strengths in supporting tasks in diverse kind of languages as they are trained in multilingual datasets. Recent studies have explored the multilingual evaluation of LLMs in specific tasks.

##### Translation Evaluation

Numerous studies have demonstrated that LLMs possess the ability to perform multilingual translation tasks. Research has specifically investigated the multilingual translation capabilities of models like BLOOM (Bawden & Yvon, 2023) and diverse kinds of LLMs (Zhu et al., 2023a).

BLOOM’s translation abilities were evaluated in (Bawden & Yvon, 2023) by resorting to several datasets and exploring various prompting strategies. This evaluation focused on English-French and English-Hindi parallel corpora and utilized the Flores-101 dataset (Goyal et al., 2022). The empirical results indicate that few-shot prompting strategies significantly improve translation quality. However, translating low-resource languages remains challenging, even when the language is included in the training data.

For a more extensive study, Zhu et al. (2023a) evaluated LLMs in 102 languages and 606 translation directions that are English-centric, French-centric and Chinese-centric. MLLMs were compared with machine translation models with varying model sizes, resorting to the Flores-101 dataset (Goyal et al., 2022).

##### Question Answering Evaluation

MLLMs can provide valuable knowledge and insights on the basis of the questions that are prompted. The most prominent datasets for the evaluation of this task are XQuAD (Artetxe et al., 2020) and MLQA (Lewis et al., 2020).

In MLLMs in multilingual context, several studies have undertaken evaluation of multilingual question answering. M3Exam was proposed by Zhang et al. (2023e), a benchmark to evaluate LLMs when they handle human exam questions in multilingual, multimodal and multilevel contexts. This dataset contains more than 12K questions in 9 languages by gathering the official exams from different countries. It has also questions in multimodal context where questions are provided in text and need to be answered based on images. It is divided into 3 levels: primary, middle and high school.

For more diverse languages evaluation, MultiQ was proposed in Holtermann et al. (2024), a benchmark to evaluate basic question answering task in 137 languages. This dataset comprises more than 27K prompts. It is originally from 200 English prompts, 100 prompts gathered from LMSYS-Chat-1M (Zheng et al., 2023a) and 100 prompts curated manually by prompting GPT-4 to provide a question and the answer. These questions are then translated automatically into 136 other languages.

##### Summarization Evaluation

Summarization is a common task in natural language processing (NLP), where the input is a document and the output is another one that is a shorter version of it.

SEAHORSE (Clark et al., 2020) is a dataset for multilingual and multifaceted summarization evaluation. This dataset consists of 96K summaries in 6 languages and 9 different outputs (1 human summaries and 8 language models). It includes also ratings from human annotators along 6 dimensions: comprehensibility, repetition, grammar, attribution, main ideas and conciseness. The articles for the summarization are based on XSum (Narayan et al., 2018), XL-Sum (Hasan et al., 2021), MLSum (Pagnoni et al., 2021), and WikiLingua (Ladhak et al., 2020).

##### Dialogue Evaluation

With MLLMs, dialogues are supported in more than one languages, being thus important to evaluate dialogues for different languages.

xDial-Eval (Zhang et al., 2023a) is a multilingual benchmark for evaluating the open-domain dialogue task. This dataset is sourced from the 12 turn-level and 6 dialogue-level datasets in English. The size of the collected datasets are 14930 annotated turns and 8691 annotated multi-turn dialogues. These datasets are translated into 9 diverse languages by utilizing machine translation, which was validated by human evaluation.

In another study, Ferron et al. (2023) evaluate the engagingness in dialogue, especially in multilingual contexts. An engaged reply is increasing the attention, interest and participation of the users. 5 subdimensions of engagingness are proposed: response diversity, interactional quality, interestingness, contextual specificity and othering. 4 turn- and dialogue-level datasets were used that cover 3 languages: English, Chinese, and Spanish.

#### 7.2.3 Multilingual Alignment Evaluation

The alignment of MLLMs towards human-values are crucial, by means of which LLMs should follow human preferences, not adhering to bias or producing toxic outputs (Wang et al., 2024b). It is thus important to evaluate MLLMs not only in their alignment with respect to English only, but also in multilingual terms.

##### Multilingual Ethics Evaluation

MLLMs should follow human ethics and preferences also in multilingual contexts. An ethical MLLM is a MLLM that is able to discern what is morally good or bad in multilingual contexts. Agarwal et al. (2024) proposed an evaluation benchmark in multilingual ethical reasoning across 7 different languages. The ethical categories are divided into virtue, deontology and consequentialism. This dataset is originally in English (Rao et al., 2023) and it was translated automatically into 6 languages. The experimental results show that high-resource languages has a strong ethical reasoning capability, but the capability is lower in low-resource languages.

##### Multilingual Toxicity Evaluation

Toxicity or toxic degeneration is defined as a disrespectful, rude or unreasonable text and make people leave a discussion (Jain et al., 2024). A benchmark for evaluating toxicity in English, named RealToxicityPrompts (RTP), was developed in (Gehman et al., 2020). For expanding the evaluation of toxicity into multilingual contexts, RTP-LX (RTP-Language eXpanded) dataset was curated (de Wynter et al., 2024). This dataset used RealToxicityPrompts as the seed corpus and added cultural-specific human-crafted prompts. It cover 27 languages with more than 1K prompts.

In another study, PolygloToxicityPrompts (PTP) was developed as the first large-scale multilingual benchmark to evaluate toxic degeneration in MLLMs (Jain et al., 2024). Differently from RTP-LX, this dataset is scraped from mC4 and THE PILE corpora. It consists of 425K prompts with the respective toxicity scores and covers 17 languages, with 25K prompts for each language.

##### Multilingual Bias Evaluation

Harmful biases need to be evaluated and mitigated, where a most concerning issue is the gender bias. In multilingual bias evaluation, Vashishtha et al. (2023) proposed two benchmarks to evaluate biases across languages. One of them is Discovery of Correlations (DisCo) (Webster et al., 2020) concerning a measurement of unfairness or bias in MLLMs to predict particular gender. Another one uses multilingual bias evaluation (MBE) score Kaneko et al. (2022) and it a bias evaluation dataset in 8 high-resource languages.

#### 7.2.4 Multilingual Safety Evaluation

The safety of MLLMs is a major concern before deployment for public use. It is crucial to evaluate their robustness and risk mitigation against adversarial attacks, malicious inputs and the generation of unsafe outputs. In a multilingual context, safety evaluations must be conducted for each language, encompassing both high-resource and low-resource languages, ensuring that MLLMs maintain consistent safety standards across all languages.

##### Multilingual Safety Benchmark

There are different kind of safety scenarios that must be addressed before deploying MLLMs, e.g. privacy, illegal activities and harmful contents. Wang et al. (2023d) created the XSAFETY dataset, a multilingual safety dataset encompassing 14 different safety scenarios. This dataset is curated from the translation of SAFETYPROMPTS (Sun et al., 2023) and SAFETEXT (Levy et al., 2022) datasets into 9 languages. They evaluated several MLLMs, including ChatGPT, PaLM2, LLaMA2-Chat-13B and Vicuna-13B. Results indicate that all MLLMs produce more unsafe responses in non-English languages.

##### Multilingual Jailbreaking and Red-Teaming

In real-life scenarios, users may enter malicious inputs, such as harmful instructions or adversarial attacks. Jailbreaking or red-teaming are methods recently used to evaluate the robustness of LLMs. If MLLMs are not jailbroken and continue to produce safe content after red-teaming attempts, they are considered to be robust.

Yong et al. (2023a) conducted multilingual jailbreaking towards GPT-4 with AdvBench dataset (Zou et al., 2023), which is translated into 12 languages that high-resource, medium-resource and low-resource. The results indicate that chances to jailbreak GPT-4 increases with the usage of low-resource languages.

Jailbreaking in multilingual contexts were also studied by Deng et al. (2023), who developed MultiJail, a dataset for jailbreaking sampled from Antrophic’s red teaming dataset (Ganguli et al., 2022) and translated into 9 languages that are high-resource, medium-resource and low-resource. The evaluation metric consists in counting attack success rates. Their findings suggest that low-resource languages can be potential targets for jailbreaking. In other words, the robustness of MLLMs is primarily focused on English, while other languages exhibit lower robustness.

Extensive empirical study of multilingual jailbreaking was undertaken in (Li et al., 2024e). The dataset used was originally in English, inspired by previous studies, and it was translated into 8 languages with a semantic-preserving algorithm, with the final dataset containing 365 multilingual questions. This study also performed MLLM representation analysis, addressing the attention visualization of malicious questions. An empirical result obtained indicate that MLLMs focus on specific keywords in questions without jailbreak templates, leading to non-responses, while questions with templates have a more dispersed attention.

### 7.3 Multilingualism Evaluation

Recent approaches to evaluating the multilingual capabilities of MLLMs involve testing them with evaluation datasets and assessing the generated text (Zhao et al., 2024d; Aggarwal et al., 2024). However, a deep understanding of MLLMs’ multilingualism remains underexplored.

Some studies have conducted extensive analyses and used interpretability methods to investigate models’ inner workings. For instance, Yuan et al. (2023a) proposed a benchmark to evaluate multilingual capabilities of MLLMs that when fine-tuned in one source language can be applied to other languages. It uses Embed FT, a method that only trains the embedding layer and keep the parameters frozen. Embed FT was trained with 101 language pairs, from and to English.

To proceed with the analysis, languages were divided into four multilingual quality quadrants based on the results. The selfish quadrant contains languages whose model’s capabilities only increase for the corresponding language pair. The reciprocal quadrant have languages whose training with bilingual data improves corresponding language pair and boost multilingual capabilities. The altruistic quadrant encompasses languages that primarily improve the multilingual performance. The idle quadrant contain languages that do not improve neither bilingual nor multilingual performance.

### 7.4 MLLMs as Multilingual Evaluator

Evaluating MLLMs is a complex and resource-intensive task, particularly when evaluating performance across numerous languages, diverse tasks, and various domains (Mu et al., 2024a). Curating high-quality evaluation datasets demands significant resources, in terms of financial support and human annotators. To mitigate this, several studies have explored the use of MLLMs as evaluators for MLLMs, leveraging their capabilities to evaluate MLLMs in multilingual context (Hada et al., 2024b; a).

Hada et al. (2024b) explored the potential of an LLM-based evaluator, having resorted to GPT-4 for multilingual evaluation. They evaluate tasks in open prompt, continue writing and summarization in 8 languages: English, French, German, Spanish, Chinese, Japanese, Italian, Portuguese and Czech. The evaluation metrics including linguistic acceptability (LA), output content quality (OCQ), task quality (TQ), problematic content (PC), and hallucination (H). They performed inner-annotator agreement (IAA) analysis between human annotators and GPT-4. The results indicate that GPT-4 demonstrates relatively high consistency for non-English languages. However, there is a bias compared to human judgments in low-resource and non-Latin script languages.

METAL, an end-to-end multilingual meta-evaluation framework, was proposed in Hada et al. (2024a) to assess MLLMs, as an evaluator in multilingual context. Firstly, the meta-evaluation dataset was created by prompting GPT-4 to generate data samples, whose evaluation is performed by native speakers. With this curated dataset, covering 10 languages, with a total of 1000 summaries, the evaluation was conducted by prompting MLLMs with its test examples and evaluating the outcome with five metrics, following previous work (Hada et al., 2024b).

## 8 Interpretability

Training an MLLM with multilingual data and subsequently fine-tuning it with multilingual instruction data can result in a highly effective MLLM. However, understanding the underlying mechanisms of such a model becomes challenging due to the diverse distribution of the training data and the complexity of the model’s structure. Exploring the interpretability of MLLMs, or just MLLMs in general, is crucial for several reasons.

1) It enhances the credibility and reliability of these models in practical use. When processing sensitive information or making important decisions, it is essential to explain the model’s behavior and inferential processes to ensure compliance with legal, ethical or social standards.

2) Interpretability helps identify biases and unfairness in the models. Due to imbalances in the quantity and quality of linguistic data, MLLMs may perform better in some languages than others, leading to linguistic bias. By interpreting the models’ internal decision-making processes, we can better identify and correct these biases, making the models fairer.

3) Analyzing the internal decision-making processes in depth allows us to discover patterns and regularities that can further improve model performance. For instance, understanding the mechanisms behind cross-lingual transfer in MLLMs can help enhance the performance of low-resource languages.

While the interpretability of MLLMs is challenging, the interpretability of MLLMs brings specific complexities. It centers on how the unique attributes of various languages and their interactions influence these models.

Our discussion on the interpretability of MLLMs will begin by exploring three main issues: how these models manage multilingual capabilities, how they perform cross-lingual transfer, and the reasons behind language bias.

The first issue is the basis for investigating the interpretability of MLLMs, while the latter two issues are phenomena that arise after modeling multilingual capabilities in MLLMs.

The framework of this section is shown in Figure 5, and the specific perspectives and approaches are shown in Figure 6.

### 8.1 Interpretability of Multilingual Capabilities

Studying the interpretability of MLLMs first requires understanding how these models specifically handle multilingual capabilities, how inputs from different languages are represented and processed (Zhao et al., 2024d). This involves examining how the model shares semantic information across languages and manages language-specific features. Previous studies have explored the multilingual capabilities of MLLMs through four perspectives: model, component, neuron and representation.

At the model level, the behavior and decision paths of the entire model is analyzed to understand its overall multilingual processing capabilities. At the component level, the focus is on the internal components, such as feed-forward network and attention module, and investigates their roles in a multilingual environment. The neuron level delves into finer granularity, exploring the functions of individual neurons in multilingual tasks. Finally, the representation level examines how the model learns multilingual representations, investigating how inputs from different languages are represented and distinguished in high-dimensional space. Through these levels of analysis, a comprehensive understanding is gained on how MLLMs operate and process multilingual information.

#### 8.1.1 Model-Wide Interpretation

To uncover the internal mechanisms of MLLMs modeling multilingual capabilities, some studies begins by examining the model as a whole. Generally, this aims to divide the model’s processing of multilingual data into distinct stages. Specifically, these studies are categorized into two perspectives: static reasoning and dynamic training. Static reasoning focuses on the trained model, while dynamic training examines the model during its training process.

##### Static Reasoning

Existing studies suggest that the layers closest to a model’s input or output exhibit more language-specific behaviors than the layers in the middle (Bhattacharya & Bojar, 2023). For instance, Zhao et al. (2024c) summarized the multilingual workflow of an MLLM by observing changes in the ratio of languages between layers. In this workflow, multilingual inputs are converted to English near the input layer. The model uses English for processing and integrates multilingual knowledge through specific structures in the intermediate layers. Near the output layer, the information is converted back to the original input language.

Additionally, some studies interpret the intermediate process as encoding into a conceptual space, noting that the model appears to think in concepts that favor English, as the English space seems to be closer to this conceptual space than other languages spaces (Wendler et al., 2024).

##### Dynamic Training

patterns in the dynamic training process of MLLMs were uncovered by Blevins et al. (2022). Compared to monolingual LLMs, MLLMs acquire information within each language in essentially the same order as monolingual models. However, this internal language learning occurs early in the training process. In contrast, cross-lingual information transfer is learned throughout pre-training, with the sequence of transferring linguistic information between specific languages varying considerably.

#### 8.1.2 Component-Based Interpretation

The components in MLLMs play a crucial role in modeling multilingual capabilities. In Bhattacharya & Bojar (2023), distinct patterns of multilingual processing in the sub-layers of the model’s feed-forward network were discovered. It divided FFNs into detectors and combiners, finding that the detectors in the early and late layers are multilingual, while the combiners in the middle layer also exhibit multilingual properties.

#### 8.1.3 Neuron-Level Interpretation

Due to the complexity of MLLM structures, where each component can have multiple functions, numerous studies have investigated how MLLMs model multilingual capabilities at a finer granularity–—the neuron level. A core region corresponding to multilingual capabilities in MLLMs was identified in Zhang et al. (2024e), as perturbing parameters in this region were demonstrated to affect performance across all languages. Additionally, it found a region related to specific monolingual capabilities, where altering parameters decreases the performance of the corresponding language.

#### 8.1.4 Representation-Driven Interpretation

Considering interpretability from the perspective of model-specific structures, including the overall model, components and neurons, is one of the research approaches. Another approach focuses solely on the intermediate representations produced by the model. This involves transforming high-dimensional model representations into low-dimensional, human-understandable features through specific operations, typically probe methods. These probe methods can be divided into two categories: parameter-free probing methods, such as direct feature dimensionality reduction; and parametric probing methods, such as those using diagnostic classifiers to classify features.

##### Parameter-free Probing Interpretation

The parameter-free probing approach primarily targets encoder-based MLLMs.

Studies have shown that after applying mean-centering operations on the representations of MLLMs, different languages occupy similar linear subspaces. Additionally, it has been found that these MLLMs encode information along orthogonal axes that are either language-sensitive or language-neutral, addressing both language-agnostic and language-specific information (Chang et al., 2022).

In contrast, some other studies aim at maximizing the use of language-agnostic information in representations while ignoring language-specific information. This is achieved through methods like clustering-based anisotropy enhancement (Rajaee & Pilehvar, 2022) or identifying low-rank subspaces that encode semantically irrelevant information (Xie et al., 2022), which can improve the performance of language-agnostic tasks, such as cross-lingual text similarity (Tiyajamorn et al., 2021).

Other studies have explored the reason of cross-lingual generic representations, identifying the significant role of shared parameters. Surprisingly, these studies also found similarities in the representations of models from different languages even when parameters are not shared (Conneau et al., 2020b).

##### Parametric Probing Interpretation

Some parametric probing methods focus on uncovering the underlying cross-lingual syntactic and morphological representations. By applying probing classifiers to syntax and morphology-related tasks, researchers have found that MLLMs learn syntactic and morphological content in different languages in a very similar way (Mikhailov et al., 2021). These models can even learn this content across languages in a monolingual-like manner (Starace et al., 2023).

The intrinsic probing approach suggests this may be because some neurons can encode universal syntactic and morphological information. However, since subsets of neurons from different languages do not completely overlap, the complexity of syntactic morphological information and the distance between languages may affect the learning of syntactic morphology across languages (Stanczak et al., 2022).

In more advanced research, some approaches explore lexical and semantic representations in MLLMs. Cross-lingual lexical fine-tuning followed by interpolation with static linguistic word embeddings was used in Vulic et al. (2023) to reveal cross-lingual lexical knowledge from MLLMs.

Additionally, de Varda & Marelli (2024) trained linear probes using extracted representations to identify neurons most relevant to the target lexicon, discovering significant cross-lingual overlap in these neurons.

A special case of lexical representation is factual knowledge representation. Existing MLLMs lack cross-lingual consistency in encoding factual knowledge, leading to different answers when the same factual question is asked in different languages (Jiang et al., 2020; Kassner et al., 2021; Qi et al., 2023). One potential solution is to explicitly transfer relatively rich factual knowledge from English to non-English languages (Xu et al., 2023d).

### 8.2 Interpretability of Cross-lingual Transfer

Cross-lingual transfer refers to the transfer of knowledge from one language to another. Despite the challenges posed by differences between languages, similar concepts can facilitate this process.

Studies have shown that zero-shot and few-shot cross-lingual transfer occurs in MLLMs (Xu et al., 2023c). While some attribute this phenomenon to shared sub-word tokens (Deshpande et al., 2022), others argue that parameter sharing (Conneau et al., 2020b) and network depth (K et al., 2020) are more crucial.

### 8.3 Interpretability of Linguistic Bias

Language bias refers to the tendency to favor or discriminate against certain groups in language use (Zhang et al., 2024b). This issue arises in MLLMs due to the imbalance in the distribution of high- and low-resource languages within their training data. Consequently, MLLMs can exhibit language bias, such as not fully understanding the cultural context of certain languages.

This bias may stem from the distribution imbalance between high- and low-resource datasets, causing the model to shift inputs from various languages to high-resource languages, particularly English, to process information (Zhao et al., 2024c).

Additionally, it has been suggested that this bias could be due to the close proximity between the space of English tokens and abstract concepts, further contributing to linguistic bias in MLLMs (Wendler et al., 2024).

## 9 Application

Due to its excellent multi-language understanding and cross-language generalization capabilities, MLLMs have demonstrated excellent performance in various downstream tasks. Therefore, MLLM is widely used in various professional domains, including biology and medicine, computer science, mathematics, law, etc.

These domain-specific MLLMs have demonstrated outstanding capabilities and promising perspectives in related domains, and have even surpassed human levels in some aspects. As a consequence, MLLMs provide a new approach for the integration of artificial intelligence and these domains. Nevertheless, the application of MLLMs in these domains remains challenging, mainly due to the reliance on specific expertise and data collection requirements.

At present, MLLMs abound more for English and Chinese, and there are fewer suitable for low-resource languages, which greatly hinders the development of generative AI on a global scale. In this section, we address the development trajectory and recent progress of MLLMs across different domains, focusing on their practical applications, as shown in Figure 7.

### 9.1 MLLMs for Biology and Medicine

The integration of MLLMs has shown tremendous potential in the health sector, particularly in applications such as medical Q&A, intelligent diagnosis, and psychological counseling (Qiu et al., 2024; Lifelo et al., 2024; García-Ferrero et al., 2024).

By bridging language barriers and enhancing data-driven research and clinical practice, MLLMs have significantly advanced the fields of biology and medicine. Their ability to understand and generate human language across multiple languages has demonstrated substantial promise in improving medical diagnostics, treatment, and research in diverse linguistic contexts globally.

BioBERT (Lee et al., 2020) is a domain-specific language representation model pre-trained on a large biomedical corpus. It significantly outperforms BERT in numerous biomedical text mining tasks.

DNABERT (Ji et al., 2021) is a pre-trained bidirectional encoder representation designed to capture a global and transferable understanding of genomic DNA sequences based on upstream and downstream nucleotide contexts. DNABERT-2 (Zhou et al., 2023b) is trained on multi-species genomes and is more efficient, powerful, and easy to use than its first generation.

Ming-MOE (Liao et al., 2024), a novel Mixture-of-Expert-based medical large language model designed to manage diverse and complex medical tasks without requiring task-specific annotations, thus enhancing its usability across extensive datasets.

DoctorGLM (Xiong et al., 2023) is a Chinese medical inquiry model based on ChatGLM-6B. It is pre-trained using various techniques on a collected Chinese medical dialogue database.

HuatuoGPT (Zhang et al., 2023b) leverage both distilled data from ChatGPT and real-world data from doctors in the supervised fine-tuned stage.

MedGPT (Kraljevic et al., 2021) is a large medical language model based on LLaMA-13B, fine-tuned through supervised learning for multiple tasks. It excels in various applications, including disease inquiry, differential diagnosis, recommending tests and examinations, summarizing medical records, interpreting examination results, and providing diagnostic outcomes and treatment plans.

ClinicalGPT (Wang et al., 2023a) is a language model specifically designed and optimized for clinical settings. Trained on real-world medical records and domain-specific knowledge, it excels in tasks such as medical knowledge question-answering, physical examinations, patient consultations and medical record analysis.

IvyGPT (Wang et al., 2023b), a model based on LLaMA, has been trained and fine-tuned using over 300,000 high-quality medical question-answer (QA) instances and reinforced learning from human feedback (RLHF). It demonstrates strong multi-turn dialogue capabilities, providing more detailed and human-like diagnostic and treatment responses.

BianQue (Chen et al., 2023b) utilizing ChatYuan as its base model, is a large-scale medical dialogue model fine-tuned through a combination of instruction and multi-turn inquiry dialogues. It has been refined on a mixed dataset of over 9 million Chinese medical question-answering instructions and multi-turn dialogues.

SoulChat (Chen et al., 2023c) employs the ChatGLM-6B model as its initialization framework and has undergone comprehensive fine-tuning on both single-turn and multi-turn mental health counseling dialogue datasets. It is capable of demonstrating empathy, encouraging users to express themselves, and providing reasonable advice.

Med-PaLM (Singhal et al., 2023), a large language model from Google Research, designed to provide high quality answers to medical questions. It was the first AI system to overcome the pass mark (>60%) in the U.S.A.

ChatDoctor (Li et al., 2023h) is a medical assistant utilizing the LLaMA model, trained with integrated medical knowledge. It has been fine-tuned on over 100,000 real doctor-patient conversations. It not only demonstrates fluent conversational abilities but also exhibits a high level of understanding and diagnostic accuracy in the medical domain.

### 9.2 MLLMs for Computer Science

Recently, there have been significant advancements in the application of large language models within the domain of computer science, particularly in tasks such as code generation and text-to-SQL conversion. The application of large language models is reshaping the domain of computer science. The paradigm has shifted from manually writing code to generating it and making human corrections. In this subsection, we will explore the development history and recent advancements of MLLMs in the domain of computer science.

Encoder-Only: Due to the Encoder-Only architecture’s ability to effectively capture the global dependencies and features of input sequences, it excels in code detection and classification tasks.

Feng et al. (2020) proposed CodeBERT, a pre-trained model specifically designed for programming languages. This model comprises 124 million parameters and supports 6 different programming languages.

In the same year, CuBERT (Kanade et al., 2020) was introduced, focusing on training BERT on source code to obtain contextual embeddings. This model was used to identify code block defects and detect duplicate code blocks.

Encoder-Decoder: The Encoder-Decoder architecture is a neural network model widely used in natural language processing and other sequence-to-sequence tasks. It has now been extensively applied to tasks such as text generation, question answering, and code generation.

PLBART (Ahmad et al., 2021) adopts the encoder-decoder BART architecture and is pre-trained on an extensive collection of programming languages (PL) and natural languages (NL) via denoising autoencoding. It is primarily used for code summarization, text-to-code generation, code-to-code translation, and code refinement.

CodeT5 (Wang et al., 2021) supports eight common programming languages and employs the same pre-training approach as T5 (Raffel et al., 2020a). It focuses on masked span prediction, denoising sequence reconstruction and masked identifier prediction with a bimodal dual generation strategy. This strategy encourages better alignment between NL and PL, allowing CodeT5 to significantly outperform PLBART across all generation tasks.

CodeT5+ (Wang et al., 2023g) is an enhanced version of CodeT5, employing a larger parameter scale to deliver enhanced performance and more accurate code comprehension. It mainly features three parameter levels: 2B, 6B and 16B. Across various code-related tasks such as code completion, code recommendation and code classification, it demonstrates superior performance compared to CodeT5.

In 2022, DeepMind announced the launch of AlphaCode (Li et al., 2022), a code generation system that can create novel solutions to problems requiring deeper reasoning, using a Transformer-based model. It was trained on over 715 GB of data from GitHub and Codeforces issues and solutions, and it supports twelve of the most common programming languages.

Decoder-Only: The Decoder-Only architecture is currently the most commonly used MLLM architecture, typically employed for sequence generation tasks such as text generation and machine translation.

Codex (Chen et al., 2021), a model provided exclusively through OpenAI’s API, is a descendant of GPT-3. It is available in three sizes: 300M, 2.5B and 12B, and it powers GitHub Copilot, a well-known and robust model renowned for its performance. While excelling in Python, it also demonstrates proficiency in a variety of other programming languages, for tasks such as code translation, code explanation and code refactoring. However, unlike other models, Codex is not publicly downloadable.

PolyCoder (Xu et al., 2022) is an MLLM based on the GPT-2 architecture. It has been trained on 249GB of code across twelve common programming languages and is available in three sizes: 160M, 0.4B and 2.7B. In the C programming language, PolyCoder outperforms all previous models, including Codex.

CodeParrot is a GPT-2 model trained exclusively on 180GB of Python code. It is available in two sizes, 110M and 1.5B parameters.

CodeGen (Nijkamp et al., 2023) is an autoregressive language model designed for program synthesis, sequentially trained on The Pile, BigQuery and BigPython datasets. Its goal is to enhance developer productivity by converting metadata into readable and maintainable source code.

Incoder (Fried et al., 2023) is trained on code using a causal masking objective, enabling code insertion/filling as well as standard left-to-right generation. It is trained on public open-source repositories with permissive, non-Copyleft licenses from GitHub, GitLab, and StackOverflow. These repositories predominantly contain Python and JavaScript but also include code in twenty eight other languages sourced from StackOverflow. Incoder is available in two variants, 1.3B and 6B parameters.

Codey, a fine-tuned model of PaLM2,262626https://lablab.ai/tech/google/codey#codey-google-ais-revolutionary-coding-assistant is capable of performing varying coding tasks.
It is fine-tuned with extensive high-quality code and coding documentation. Google claims that Codey can code in more than twenty programming languages.
It is used to enhance Google products like Google Colab, Android Studio, and more.

The CodeLlama (Rozière et al., 2023) release introduces a series of models with 7B, 13B and 34B parameters. These base models are initialized from Llama 2 and then trained on 500B tokens of code. Meta subsequently fine-tuned these base models into two distinct flavors: a Python specialist variant (with an additional 100B tokens) and an instruction fine-tuned version capable of comprehending natural language instructions. These models exhibit state-of-the-art performance across various programming languages.

StarCoder (Li et al., 2023f) is trained on permissively licensed data from GitHub, encompassing over eighty programming languages, Git commits, GitHub issues and Jupyter notebooks. Similar to LLaMA, it is a 15 billion parameter model trained on 1 trillion tokens. StarCode outperforms existing open code language models on popular programming benchmarks and matches or surpasses closed models such as Codex.

ChatGPT and GPT-4 are advanced language models developed by OpenAI. They utilize Reinforcement Learning with Human Feedback (RLHF) to enhance their program synthesis capabilities. These models have demonstrated proficiency in code generation tasks, often surpassing human-level performance.

CodeGeeX (Zheng et al., 2023c) and CodeGeeX2 (Zheng et al., 2023d) are multilingual code generation models developed by Tsinghua University. Unlike CodeGeeX, CodeGeeX2 is built on the ChatGLM2 architecture with added code pre-training. Leveraging the superior performance of ChatGLM2, CodeGeeX2 achieves performance improvements across multiple benchmarks. Remarkably, with only 6B parameters, it surpasses the 15B parameter StarCoder-15B by nearly 10%.

The original training data for CodeShell (Xie et al., 2024) is based on self-collected GitHub data, the Stack and StarCoder datasets, and a small amount of high-quality Chinese and English data. It underwent cold start training on 500B tokens, with a context window length of 8192. CodeShell’s code generation performance surpasses that of CodeLlama-7B and StarCoder-7B.

CodeGemma (Mesnard et al., 2024) is a family of code-specialist LLM models by Google, based on the pre-trained 2B and 7B Gemma (Mesnard et al., 2024) checkpoints. CodeGemma are further trained on an additional 500B tokens of primarily English language data, mathematics and code to improve on logical and mathematical reasoning, and are suitable for code completion and generation.

CodeQwen1.5 (Bai et al., 2023a) is the code-specific version of Qwen1.5, pre-trained on a vast corpus of code data. With a context length of 64K tokens and support for ninety two programming languages, it exhibits robust code generation capabilities across a range of benchmark tests. Additionally, it demonstrates outstanding performance in tasks such as text-to-SQL conversion and error correction.

### 9.3 MLLMs for Mathematics

In recent years, there has been a significant surge in the development of large language models aimed at automating the process of solving mathematical problems, which given the wide-ranging and diverse nature of mathematical problem, presents a challenge to the development of this emerging field.

Multilingual large language models perform exceptionally well in tasks such as text completion and machine translation. However, they exhibit notable limitations in solving, explaining, answering and recommending solutions for mathematical problems.

In this subsection, we will discuss recent advancements of MLLMs in the domain of mathematics.

MathGPT (Scarlatos & Lan, 2023) is a large-scale model targeting global mathematics enthusiasts and research institutions with problem-solving and teaching algorithms at its core. Its mathematical computation capabilities span across primary, middle and high school levels. In evaluations across six publicly available math test sets–—CEval-Math, AGIEval-Math, APE5K, CMMLU-Math, high school math exams, and Math401–—MathGPT has achieved the highest scores in multiple tests, even surpassing those of GPT-4.

The ChatGLM-Math (Xu et al., 2024d) model is an LLM that enhances mathematical problem-solving abilities through a self-critical pipeline. This model not only improves mathematical skills but also preserves and enhances language capabilities, resulting in performance enhancements across various tasks.

The DeepSeekMath (Shao et al., 2024) model underwent pretraining with a total of 500 billion tokens, which included mathematical-related texts from Common Crawl as well as natural language and code data. Built upon the foundational architecture of DeepSeek-Coder-v1.5 7B, the model received specialized instruction tuning and reinforcement learning training to enhance its mathematical problem-solving and tool utilization abilities. Additionally, DeepSeekMath 7B achieved performance levels similar to Gemini-Ultra and GPT-4 in the highly competitive MATH Challenge.

MetaMath (Yu et al., 2023), fine-tuned on the MetaMathQA dataset using the LLaMA-2 architecture, is a large language model specialized in mathematical reasoning (both forward and backward). MetaMath-70B has surpassed ChatGPT in performance on mathematical reasoning datasets, achieving state-of-the-art results.

MAmmoTH (Yue et al., 2023b) underwent instruction fine-tuning on the MathInstruct dataset, which covers various mathematical domains and complexities, blending Concept of Thinking (CoT) with Programming of Thinking (PoT). The MAmmoTH-34B model has surpassed the CoT results of GPT-4 on competition-level datasets.

WizardMath (Luo et al., 2023a) combines reinforcement learning with math-specific instruction data to enhance the abilities of LLMs in mathematical reasoning. It surpasses advanced models such as ChatGPT, Claude Instant-1, PaLM-2, and Minerva on the GSM8K dataset.

GAIRMath-Abel (Chern et al., 2023) achieves state-of-the-art performance across open-source LLMs solely through the utilization of Parental Oversight, a Babysitting Strategy for Supervised Fine-tuning, without requiring tools, continued pretraining, reward modeling or reinforcement learning from human feedback (RLHF).

Orca-Math (Mitra et al., 2024), a small language model constructed with 700M parameters, is fine-tuned on the Mistral-7B architecture. It redefines the traditional approach to teaching mathematical word problems through creative synthetic datasets and iterative learning mechanisms. Notably, Orca-Math has achieved significant advancements on the GSM8K benchmark.

### 9.4 MLLMs for Law

While general multilingual large language models have the capacity to address a broad spectrum of knowledge domains, the legal field is a highly specialized domain that necessitates both utmost accuracy and timely relevance. General MLLMs may be deficient in their grasp of legal statutes, judicial interpretations and nuances particular to specific jurisdictions, thereby leading to imprecise responses or the omission of crucial information.

Consequently, the development of specialized legal MLLMs becomes imperative to overcome these limitations and cater to the specific demands of the legal sector.

In this section, we address the most recent advancements of MLLMs within the domain of law.

LegalBERT (Chalkidis et al., 2020), based on the BERT model, is the first large-scale model pre-trained on legal texts. It was trained on the entire Harvard Law School case corpus, which comprises 37GB of legal data and includes 3,446,187 legal decisions from both federal and state courts. The model demonstrates exceptional performance on downstream legal tasks.

Lawformer (Xiao et al., 2021), based on Longformer, is pre-trained on a large-scale corpus of Chinese legal texts. As the first Chinese legal pre-training model, it does not use standard self-attention but instead combines local sliding windows with a global attention mechanism to capture long-range dependencies. It demonstrates exceptional performance on long-text legal tasks.

LawGPT (Nguyen, 2023) builds on existing general-purpose Chinese language models by expanding the vocabulary with legal-specific terms and conducting pre-training on a large-scale corpus of Chinese legal texts. This enhances the model’s foundational semantic understanding in the legal domain. Additionally, LawGPT undergoes instruction fine-tuning on legal domain dialogue question-answering datasets and the Chinese judicial examination dataset, improving its comprehension and execution of legal content.

Fudan University has introduced DISC-LawLLM (Yue et al., 2023a), a large-scale model designed to provide users with professional, intelligent and comprehensive legal services. It surpasses ChatGPT in performance on the latest legal evaluation benchmark, Lawbench, and is second only to GPT-4.

The ChatLaw (Cui et al., 2023b) model comes in two versions, one based on the Ziya-13B model and the other on the Anima-33B model. It is pre-trained on extensive legal dialogue data, including legal news, forums, statutes, judicial interpretations, legal consultations, judicial exam questions and court judgments. ChatLaw demonstrates excellent performance on various legal tasks.

LexiLaw (Li et al., 2023b), built on the ChatGLM-6B architecture, has been fine-tuned on legal domain datasets to enhance its performance and professionalism in providing legal consultation and support.

Lawyer LLaMA (Huang et al., 2023), based on the LLaMA-13B architecture, underwent continual pretraining on a large-scale legal corpus. Subsequently, the model was instruction fine-tuned using collected legal data, enabling a deep understanding of common areas in Chinese law, including civil law, criminal law, administrative law, and procedural law.

legal-ELECTRA (Hua et al., 2022) is built upon the ELECTRA model and further pre-trained on a dataset of 20M high-quality judicial documents, enabling the model to better comprehend legal texts.

## 10 Challenges and Future Directions

The rapid advancement of MLLMs has opened exciting new avenues for cross-lingual language understanding and generation. However, several critical challenges remain, hindering the full realization of their potential and necessitating further research and development.

### 10.1 Democratizing Language Technology

Democratizing language technology and effectively addressing linguistic diversity represent paramount challenges in MLLM research. Current models often exhibit a significant performance disparity, favoring high-resource languages like English due to data abundance. Bridging this gap requires innovative approaches to cross-lingual transfer learning, enabling efficient knowledge transfer from resource-rich to resource-scarce languages. This involves exploring advanced techniques such as adapter-based transfer, meta-learning for cross-lingual adaptation, and developing robust interlingua representations that capture universal linguistic properties. Furthermore, addressing the scarcity of data in low-resource languages necessitates sophisticated data augmentation and generation methods. Techniques like back-translation, paraphrasing, and leveraging synthetic data generation through controlled text generation can supplement limited datasets, though careful consideration must be given to cultural sensitivity and appropriateness during data generation. Ultimately, democratizing language technology hinges on developing MLLMs that are genuinely inclusive, capable of understanding and generating text across a wide spectrum of languages, regardless of their resource availability.

Beyond supporting a multitude of languages, a crucial aspect of linguistic diversity lies in the rich tapestry of language variations and dialects. MLLMs frequently struggle with these nuances, as they deviate significantly from standardized language forms commonly used in training data. Therefore, future research must focus on incorporating dialectal information during both training and evaluation. This may involve developing specialized modules within MLLM architectures that are sensitive to dialectal variations or by leveraging techniques like code-switching and dialect adaptation during fine-tuning. Creating dedicated evaluation benchmarks that accurately assess MLLM performance on dialectal data is also essential, emphasizing aspects such as comprehension, fluency, and cultural relevance. Furthermore, code-switching and language mixing, prevalent phenomena in multilingual communities, pose additional complexities. Effectively handling these intricate language patterns necessitates developing models capable of explicitly recognizing and adapting to code-switching dynamics, potentially through dedicated modules and by training on large-scale code-switching datasets.

### 10.2 Towards Culturally-Aware and Adaptive MLLMs

Developing truly multilingual AI necessitates moving beyond purely linguistic considerations and embracing the rich tapestry of cultural nuances embedded within language. Towards this goal, building culturally-aware and adaptive MLLMs is a critical research frontier. Current MLLMs, often trained predominantly on English data, risk perpetuating cultural biases and failing to capture the diverse perspectives and values reflected in different languages and cultures. Future research must prioritize integrating cultural knowledge into MLLMs, ensuring they are not only multilingual but also multicultural in their understanding. Training on culturally diverse datasets, encompassing various perspectives and worldviews, is essential. This involves carefully curating data that represents different cultural contexts, including literature, social media conversations, and news articles from diverse sources. Moreover, developing evaluation metrics that capture cultural nuances is crucial. These metrics should move beyond simple accuracy and fluency, assessing aspects such as cultural sensitivity, appropriateness, and the ability to adapt communication styles based on cultural context. This shift towards culturally-aware evaluation will encourage the development of MLLMs that are not only proficient in multiple languages but also respectful and understanding of the diverse cultures they represent.

Adaptability is another key dimension of culturally-aware MLLMs. Language is not static; it evolves and adapts to changing cultural contexts. Therefore, MLLMs should be equipped to learn and adapt to these evolving cultural dynamics. This involves developing models that can dynamically adjust their understanding and generation based on cultural cues, potentially by incorporating mechanisms for continuous learning and adaptation. For example, an adaptive MLLM could learn to recognize and respond appropriately to culturally-specific humor, idioms, and social norms. Furthermore, research should explore methods for personalizing MLLMs to individual cultural preferences, enabling users to tailor the model’s behavior to align with their own cultural background and values. This personalization can enhance user experience and promote inclusivity, ensuring that MLLMs are accessible and beneficial to diverse cultural communities worldwide. Building culturally-aware and adaptive MLLMs requires a multidisciplinary approach, drawing on insights from linguistics, anthropology, sociology, and other fields to develop models that are truly representative of the rich diversity of human culture.

### 10.3 Ensuring Safety, Fairness, and Interpretability

Ensuring safety, fairness, and interpretability in MLLMs is paramount, especially as these powerful models become increasingly integrated into diverse real-world applications. Their widespread deployment necessitates a rigorous focus on responsible AI principles to mitigate potential risks and ensure ethical and equitable outcomes. Safety, in the context of MLLMs, encompasses safeguarding against adversarial attacks, malicious inputs, and the generation of unsafe or harmful content. This requires developing robust defense mechanisms, including techniques like adversarial training and input sanitization, to enhance MLLM resilience against malicious exploitation. Furthermore, establishing clear safety guidelines and protocols for MLLM deployment is crucial, outlining best practices for data handling, model training, and user interaction. These guidelines should address potential risks associated with specific application domains, such as healthcare, finance, and legal services, ensuring that MLLMs are used responsibly and ethically.

Fairness is another critical concern, particularly given the potential for MLLMs to perpetuate or amplify existing societal biases. Bias can manifest in various forms, including gender bias, racial bias, and cultural bias, leading to discriminatory or unfair outcomes. Addressing this challenge requires a multi-pronged approach, encompassing both data and model-level interventions. Developing methods for detecting and mitigating biases in training data, such as debiasing techniques and data augmentation strategies, is essential. Furthermore, incorporating fairness metrics into evaluation frameworks can help quantify and monitor bias in MLLM outputs, encouraging the development of fairer and more equitable models. Promoting fairness also necessitates fostering responsible data collection and annotation practices, ensuring that data used to train MLLMs is representative and inclusive of diverse populations.

Interpretability and explainability are crucial for building trust and ensuring accountability in MLLMs. The complex nature of these models, often referred to as “black boxes" makes it challenging to understand their internal decision-making processes. Enhancing interpretability involves developing techniques for visualizing cross-lingual representations, analyzing attention patterns, and providing human-understandable explanations for MLLM outputs. This requires moving beyond simply generating text to providing insights into how the model arrived at a particular conclusion or prediction. For example, methods like probing tasks and attention visualization can help reveal the underlying linguistic knowledge and reasoning processes employed by MLLMs. Furthermore, research should focus on developing methods for explaining cross-lingual transfer, providing insights into how knowledge is transferred between languages and how this transfer can be optimized for improved performance. Ultimately, achieving interpretability and explainability in MLLMs is essential for promoting transparency, fostering trust, and enabling effective human-model collaboration.

### 10.4 Towards Efficient and Sustainable MLLMs

The computational demands of MLLMs present significant challenges to their widespread accessibility and sustainability. Training and deploying these massive models require substantial resources, raising concerns about their environmental impact and limiting their use by researchers and practitioners with constrained budgets. Addressing this challenge necessitates a multifaceted approach focusing on efficiency improvements across various dimensions. Research should prioritize developing more efficient training algorithms, including techniques like sparse training, pruning, and quantization, as well as exploring lightweight model architectures and model compression methods such as knowledge distillation. Furthermore, investigating hardware acceleration strategies, including specialized hardware and distributed training, can significantly reduce computational costs. Enhancing data efficiency through few-shot and zero-shot learning, alongside exploring alternative data sources like synthetic data and unsupervised learning, can further reduce resource requirements. Ultimately, building efficient and sustainable MLLMs demands a holistic approach, optimizing not only model architecture and training but also hardware, data usage, and the overall environmental footprint, ensuring these powerful tools are both accessible and environmentally responsible.

Future research should focus on developing more efficient training algorithms, exploring model compression techniques, and investigating hardware acceleration strategies to make MLLMs more accessible and computationally sustainable. This includes exploring lightweight architectures, knowledge distillation methods, and distributed training strategies to reduce computational costs while maintaining performance.

## 11 Conclusion

The development pace of MLLMs has been astonishing, showcasing remarkable progress across numerous tasks. However, despite ushering in a new era of artificial intelligence, our understanding of this novel form of intelligence remains relatively limited.

It is crucial to delineate the boundaries of MLLMs’ capabilities, understand their performance in various domains, and explore how to harness their potential more effectively. This necessitates a comprehensive evaluation framework to guide the direction of MLLM development.

This survey has provided a systematic and thorough elaboration on the core capabilities of MLLMs, encompassing critical aspects like cross-lingual knowledge, reasoning, alignment with human values and safety. Furthermore, it has delved into the interpretability of multilingual capabilities, cross-lingual transfer, and language bias within these models, transforming them from black boxes to white boxes.

Most importantly, the survey has explored the potential applications of MLLMs across diverse domains, including biology, medicine, computer science, mathematics and law. It has discussed how these models have driven innovation and improvements in these specialized fields, while also highlighting the challenges and opportunities in deploying MLLMs within diverse language communities and application scenarios.

## References

-
Agarwal et al. (2024)
Utkarsh Agarwal, Kumar Tanmay, Aditi Khandelwal, and Monojit Choudhury.
Ethical reasoning and moral value alignment of llms depend on the language we prompt them in.
In Nicoletta Calzolari, Min-Yen Kan, Véronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (eds.),
*Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy*, pp. 6330–6340. ELRA and ICCL, 2024. URL https://aclanthology.org/2024.lrec-main.560. -
Aggarwal et al. (2022)
Divyanshu Aggarwal, Vivek Gupta, and Anoop Kunchukuttan.
Indicxnli: Evaluating multilingual inference for indian languages.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 10994–11006. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.755. URL https://doi.org/10.18653/v1/2022.emnlp-main.755. -
Aggarwal et al. (2024)
Divyanshu Aggarwal, Ashutosh Sathe, Ishaan Watts, and Sunayana Sitaram.
Maple: Multilingual evaluation of parameter efficient finetuning of large language models.
*arXiv preprint arXiv:2401.07598*, 2024. -
Ahmad et al. (2021)
Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang.
Unified pre-training for program understanding and generation.
In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.),
*Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021*, pp. 2655–2668. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.NAACL-MAIN.211. URL https://doi.org/10.18653/v1/2021.naacl-main.211. -
Ahuja et al. (2023a)
Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Uttama Nambi, Tanuja Ganu, Sameer Segal, Mohamed Ahmed, Kalika Bali, and Sunayana Sitaram.
MEGA: multilingual evaluation of generative AI.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pp. 4232–4267. Association for Computational Linguistics, 2023a. doi: 10.18653/V1/2023.EMNLP-MAIN.258. URL https://doi.org/10.18653/v1/2023.emnlp-main.258. -
Ahuja et al. (2023b)
Sanchit Ahuja, Divyanshu Aggarwal, Varun Gumma, Ishaan Watts, Ashutosh Sathe, Millicent Ochieng, Rishav Hada, Prachi Jain, Maxamed Axmed, Kalika Bali, and Sunayana Sitaram.
MEGAVERSE: benchmarking large language models across languages, modalities, models and tasks.
*CoRR*, abs/2311.07463, 2023b. doi: 10.48550/ARXIV.2311.07463. URL https://doi.org/10.48550/arXiv.2311.07463. - AI@Meta (2024) AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md.
-
Ainslie et al. (2023)
Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai.
Gqa: Training generalized multi-query transformer models from multi-head checkpoints.
*arXiv preprint arXiv:2305.13245*, 2023. -
Ali et al. (2023)
Mehdi Ali, Michael Fromm, Klaudia Thellmann, Richard Rutmann, Max Lübbering, Johannes Leveling, Katrin Klug, Jan Ebert, Niclas Doll, Jasper Schulze Buschhoff, Charvi Jain, Alexander Arno Weber, Lena Jurkschat, Hammam Abdelwahab, Chelsea John, Pedro Ortiz Suarez, Malte Ostendorff, Samuel Weinbach, Rafet Sifa, Stefan Kesselheim, and Nicolas Flores-Herr.
Tokenizer choice for LLM training: Negligible or crucial?
*CoRR*, abs/2310.08754, 2023. doi: 10.48550/ARXIV.2310.08754. URL https://doi.org/10.48550/arXiv.2310.08754. -
Aljundi et al. (2018)
Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars.
Memory aware synapses: Learning what (not) to forget.
In Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss (eds.),
*Computer Vision - ECCV 2018 - 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part III*, volume 11207 of*Lecture Notes in Computer Science*, pp. 144–161. Springer, 2018. doi: 10.1007/978-3-030-01219-9\_9. URL https://doi.org/10.1007/978-3-030-01219-9_9. -
AlKhamissi et al. (2024)
Badr AlKhamissi, Muhammad N. ElNokrashy, Mai AlKhamissi, and Mona T. Diab.
Investigating cultural alignment of large language models.
*CoRR*, abs/2402.13231, 2024. doi: 10.48550/ARXIV.2402.13231. URL https://doi.org/10.48550/arXiv.2402.13231. -
Almazrouei et al. (2023)
Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, et al.
The falcon series of open language models.
*arXiv preprint arXiv:2311.16867*, 2023. -
Arora et al. (2022)
Arnav Arora, Lucie-Aimée Kaffee, and Isabelle Augenstein.
Probing pre-trained language models for cross-cultural differences in values.
*CoRR*, abs/2203.13722, 2022. doi: 10.48550/ARXIV.2203.13722. URL https://doi.org/10.48550/arXiv.2203.13722. -
Artetxe et al. (2020)
Mikel Artetxe, Sebastian Ruder, and Dani Yogatama.
On the cross-lingual transferability of monolingual representations.
In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (eds.),
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020*, pp. 4623–4637. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.ACL-MAIN.421. URL https://doi.org/10.18653/v1/2020.acl-main.421. -
Askell et al. (2021)
Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Benjamin Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan.
A general language assistant as a laboratory for alignment.
*CoRR*, abs/2112.00861, 2021. URL https://arxiv.org/abs/2112.00861. -
Azar et al. (2024)
Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Rémi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello.
A general theoretical paradigm to understand learning from human preferences.
In Sanjoy Dasgupta, Stephan Mandt, and Yingzhen Li (eds.),
*International Conference on Artificial Intelligence and Statistics, 2-4 May 2024, Palau de Congressos, Valencia, Spain*, volume 238 of*Proceedings of Machine Learning Research*, pp. 4447–4455. PMLR, 2024. URL https://proceedings.mlr.press/v238/gheshlaghi-azar24a.html. -
Bai et al. (2023a)
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu.
Qwen technical report.
*CoRR*, abs/2309.16609, 2023a. doi: 10.48550/ARXIV.2309.16609. URL https://doi.org/10.48550/arXiv.2309.16609. -
Bai et al. (2023b)
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al.
Qwen technical report.
*arXiv preprint arXiv:2309.16609*, 2023b. -
Bai et al. (2022)
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Chris Olah, Benjamin Mann, and Jared Kaplan.
Training a helpful and harmless assistant with reinforcement learning from human feedback.
*CoRR*, abs/2204.05862, 2022. doi: 10.48550/ARXIV.2204.05862. URL https://doi.org/10.48550/arXiv.2204.05862. -
Bandarkar et al. (2023)
Lucas Bandarkar, Davis Liang, Benjamin Muller, Mikel Artetxe, Satya Narayan Shukla, Donald Husa, Naman Goyal, Abhinandan Krishnan, Luke Zettlemoyer, and Madian Khabsa.
The belebele benchmark: a parallel reading comprehension dataset in 122 language variants.
*CoRR*, abs/2308.16884, 2023. doi: 10.48550/ARXIV.2308.16884. URL https://doi.org/10.48550/arXiv.2308.16884. -
Bang et al. (2021)
Jihwan Bang, Heesu Kim, Youngjoon Yoo, Jung-Woo Ha, and Jonghyun Choi.
Rainbow memory: Continual learning with a memory of diverse samples.
In
*IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021*, pp. 8218–8227. Computer Vision Foundation / IEEE, 2021. doi: 10.1109/CVPR46437.2021.00812. URL https://openaccess.thecvf.com/content/CVPR2021/html/Bang_Rainbow_Memory_Continual_Learning_With_a_Memory_of_Diverse_Samples_CVPR_2021_paper.html. -
Bañón et al. (2020)
Marta Bañón, Pinzhen Chen, Barry Haddow, Kenneth Heafield, Hieu Hoang, Miquel Esplà-Gomis, Mikel L. Forcada, Amir Kamran, Faheem Kirefu, Philipp Koehn, Sergio Ortiz-Rojas, Leopoldo Pla Sempere, Gema Ramírez-Sánchez, Elsa Sarrías, Marek Strelec, Brian Thompson, William Waites, Dion Wiggins, and Jaume Zaragoza.
Paracrawl: Web-scale acquisition of parallel corpora.
In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (eds.),
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020*, pp. 4555–4567. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.ACL-MAIN.417. URL https://doi.org/10.18653/v1/2020.acl-main.417. -
Bawden & Yvon (2023)
Rachel Bawden and François Yvon.
Investigating the translation performance of a large multilingual language model: the case of BLOOM.
In Mary Nurminen, Judith Brenner, Maarit Koponen, Sirkku Latomaa, Mikhail Mikhailov, Frederike Schierl, Tharindu Ranasinghe, Eva Vanmassenhove, Sergi Alvarez Vidal, Nora Aranberri, Mara Nunziatini, Carla Parra Escartín, Mikel L. Forcada, Maja Popovic, Carolina Scarton, and Helena Moniz (eds.),
*Proceedings of the 24th Annual Conference of the European Association for Machine Translation, EAMT 2023, Tampere, Finland, 12-15 June 2023*, pp. 157–170. European Association for Machine Translation, 2023. URL https://aclanthology.org/2023.eamt-1.16. -
Bendale et al. (2024)
Abhijit Bendale, Michael Sapienza, Steven Ripplinger, Simon Gibbs, Jaewon Lee, and Pranav Mistry.
Sutra: Scalable multilingual language model architecture.
*arXiv preprint arXiv:2405.06694*, 2024. -
Bhattacharya & Bojar (2023)
Sunit Bhattacharya and Ondrej Bojar.
Unveiling multilinguality in transformer models: Exploring language specificity in feed-forward networks.
In Yonatan Belinkov, Sophie Hao, Jaap Jumelet, Najoung Kim, Arya McCarthy, and Hosein Mohebbi (eds.),
*Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, BlackboxNLP@EMNLP 2023, Singapore, December 7, 2023*, pp. 120–126. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.BLACKBOXNLP-1.9. URL https://doi.org/10.18653/v1/2023.blackboxnlp-1.9. -
Bi et al. (2024)
Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al.
Deepseek llm: Scaling open-source language models with longtermism.
*arXiv preprint arXiv:2401.02954*, 2024. - Black et al. (2021) Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, March 2021. URL https://doi.org/10.5281/zenodo.5297715.
-
Blevins et al. (2022)
Terra Blevins, Hila Gonen, and Luke Zettlemoyer.
Analyzing the mono- and cross-lingual pretraining dynamics of multilingual language models.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 3575–3590. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.234. URL https://doi.org/10.18653/v1/2022.emnlp-main.234. -
Blevins et al. (2024)
Terra Blevins, Tomasz Limisiewicz, Suchin Gururangan, Margaret Li, Hila Gonen, Noah A Smith, and Luke Zettlemoyer.
Breaking the curse of multilinguality with cross-lingual expert language models.
*arXiv preprint arXiv:2401.10440*, 2024. -
Bradley & Terry (1952)
Ralph Allan Bradley and Milton E Terry.
Rank analysis of incomplete block designs: I. the method of paired comparisons.
*Biometrika*, 39(3/4):324–345, 1952. -
Brown et al. (2020a)
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.
Language models are few-shot learners.
*Advances in neural information processing systems*, 33:1877–1901, 2020a. -
Brown et al. (2020b)
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.
Language models are few-shot learners.
In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.),
*Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020b. URL https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html. -
Buzzega et al. (2020)
Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara.
Dark experience for general continual learning: a strong, simple baseline.
In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.),
*Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html. -
Cahyawijaya et al. (2023)
Samuel Cahyawijaya, Holy Lovenia, Tiezheng Yu, Willy Chung, and Pascale Fung.
Instruct-align: Teaching novel languages with to llms through alignment-based cross-lingual instruction.
*CoRR*, abs/2305.13627, 2023. doi: 10.48550/ARXIV.2305.13627. URL https://doi.org/10.48550/arXiv.2305.13627. -
Cai et al. (2024)
Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al.
Internlm2 technical report.
*arXiv preprint arXiv:2403.17297*, 2024. -
Cao et al. (2023a)
Yong Cao, Yova Kementchedjhieva, Ruixiang Cui, Antonia Karamolegkou, Li Zhou, Megan Dare, Lucia Donatelli, and Daniel Hershcovich.
Cultural adaptation of recipes.
*CoRR*, abs/2310.17353, 2023a. doi: 10.48550/ARXIV.2310.17353. URL https://doi.org/10.48550/arXiv.2310.17353. -
Cao et al. (2023b)
Yong Cao, Li Zhou, Seolhwa Lee, Laura Cabello, Min Chen, and Daniel Hershcovich.
Assessing cross-cultural alignment between chatgpt and human societies: An empirical study.
*CoRR*, abs/2303.17466, 2023b. doi: 10.48550/ARXIV.2303.17466. URL https://doi.org/10.48550/arXiv.2303.17466. -
Cao et al. (2024)
Yong Cao, Min Chen, and Daniel Hershcovich.
Bridging cultural nuances in dialogue agents through cultural value surveys.
In Yvette Graham and Matthew Purver (eds.),
*Findings of the Association for Computational Linguistics: EACL 2024, St. Julian’s, Malta, March 17-22, 2024*, pp. 929–945. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024.findings-eacl.63. -
Chai et al. (2024a)
Linzheng Chai, Jian Yang, Tao Sun, Hongcheng Guo, Jiaheng Liu, Bing Wang, Xiannian Liang, Jiaqi Bai, Tongliang Li, Qiyao Peng, et al.
xcot: Cross-lingual instruction tuning for cross-lingual chain-of-thought reasoning.
*arXiv preprint arXiv:2401.07037*, 2024a. -
Chai et al. (2024b)
Linzheng Chai, Jian Yang, Tao Sun, Hongcheng Guo, Jiaheng Liu, Bing Wang, Xinnian Liang, Jiaqi Bai, Tongliang Li, Qiyao Peng, and Zhoujun Li.
xcot: Cross-lingual instruction tuning for cross-lingual chain-of-thought reasoning.
*CoRR*, abs/2401.07037, 2024b. doi: 10.48550/ARXIV.2401.07037. URL https://doi.org/10.48550/arXiv.2401.07037. -
Chalkidis et al. (2020)
Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos.
LEGAL-BERT: the muppets straight out of law school.
*CoRR*, abs/2010.02559, 2020. URL https://arxiv.org/abs/2010.02559. -
Chang et al. (2022)
Tyler A. Chang, Zhuowen Tu, and Benjamin K. Bergen.
The geometry of multilingual language model representations.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 119–136. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.9. URL https://doi.org/10.18653/v1/2022.emnlp-main.9. -
Chang et al. (2024)
Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al.
A survey on evaluation of large language models.
*ACM Transactions on Intelligent Systems and Technology*, 15(3):1–45, 2024. -
Chaudhry et al. (2019)
Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet Kumar Dokania, Philip H. S. Torr, and Marc’Aurelio Ranzato.
Continual learning with tiny episodic memories.
*CoRR*, abs/1902.10486, 2019. URL http://arxiv.org/abs/1902.10486. -
Chen et al. (2024a)
Andong Chen, Lianzhang Lou, Kehai Chen, Xuefeng Bai, Yang Xiang, Muyun Yang, Tiejun Zhao, and Min Zhang.
Dual-reflect: Enhancing large language models for reflective translation through dual learning feedback mechanisms.
*arXiv preprint arXiv:2406.07232*, 2024a. -
Chen et al. (2021)
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba.
Evaluating large language models trained on code.
*CoRR*, abs/2107.03374, 2021. URL https://arxiv.org/abs/2107.03374. -
Chen et al. (2024b)
Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Andrey Kutuzov, Barry Haddow, and Kenneth Heafield.
Monolingual or multilingual instruction tuning: Which makes a better alpaca.
In Yvette Graham and Matthew Purver (eds.),
*Findings of the Association for Computational Linguistics: EACL 2024, St. Julian’s, Malta, March 17-22, 2024*, pp. 1347–1356. Association for Computational Linguistics, 2024b. URL https://aclanthology.org/2024.findings-eacl.90. -
Chen et al. (2023a)
Yijie Chen, Yijin Liu, Fandong Meng, Yufeng Chen, Jinan Xu, and Jie Zhou.
Improving translation faithfulness of large language models via augmenting instructions.
*CoRR*, abs/2308.12674, 2023a. doi: 10.48550/ARXIV.2308.12674. URL https://doi.org/10.48550/arXiv.2308.12674. -
Chen et al. (2023b)
Yirong Chen, Zhenyu Wang, Xiaofen Xing, Huimin Zheng, Zhipei Xu, Kai Fang, Junhong Wang, Sihang Li, Jieling Wu, Qi Liu, and Xiangmin Xu.
Bianque: Balancing the questioning and suggestion ability of health llms with multi-turn health conversations polished by chatgpt.
*CoRR*, abs/2310.15896, 2023b. doi: 10.48550/ARXIV.2310.15896. URL https://doi.org/10.48550/arXiv.2310.15896. -
Chen et al. (2023c)
Yirong Chen, Xiaofen Xing, Jingkai Lin, Huimin Zheng, Zhenyu Wang, Qi Liu, and Xiangmin Xu.
Soulchat: Improving llms’ empathy, listening, and comfort abilities through fine-tuning with multi-turn empathy conversations.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 1170–1183. Association for Computational Linguistics, 2023c. doi: 10.18653/V1/2023.FINDINGS-EMNLP.83. URL https://doi.org/10.18653/v1/2023.findings-emnlp.83. - Chen et al. (2023d) Zhihong Chen, Junying Chen, Hongbo Zhang, Feng Jiang, Guiming Chen, Fei Yu, Tiannan Wang, Juhao Liang, Chen Zhang, Zhiyi Zhang, Jianquan Li, Xiang Wan, Haizhou Li, and Benyou Wang. Llm zoo: democratizing chatgpt. https://github.com/FreedomIntelligence/LLMZoo, 2023d.
-
Chen et al. (2023e)
Zhihong Chen, Feng Jiang, Junying Chen, Tiannan Wang, Fei Yu, Guiming Chen, Hongbo Zhang, Juhao Liang, Chen Zhang, Zhiyi Zhang, Jianquan Li, Xiang Wan, Benyou Wang, and Haizhou Li.
Phoenix: Democratizing chatgpt across languages.
*arXiv preprint arXiv:2304.10453*, 2023e. -
Chen et al. (2023f)
Zhihong Chen, Feng Jiang, Junying Chen, Tiannan Wang, Fei Yu, Guiming Chen, Hongbo Zhang, Juhao Liang, Chen Zhang, Zhiyi Zhang, Jianquan Li, Xiang Wan, Benyou Wang, and Haizhou Li.
Phoenix: Democratizing chatgpt across languages.
*CoRR*, abs/2304.10453, 2023f. doi: 10.48550/ARXIV.2304.10453. URL https://doi.org/10.48550/arXiv.2304.10453. - Chern et al. (2023) Ethan Chern, Haoyang Zou, Xuefeng Li, Jiewen Hu, Kehua Feng, Junlong Li, and Pengfei Liu. Generative ai for math: Abel. https://github.com/GAIR-NLP/abel, 2023.
-
Chirkova & Nikoulina (2024)
Nadezhda Chirkova and Vassilina Nikoulina.
Zero-shot cross-lingual transfer in instruction tuning of large language model.
*CoRR*, abs/2402.14778, 2024. doi: 10.48550/ARXIV.2402.14778. URL https://doi.org/10.48550/arXiv.2402.14778. -
Chiu et al. (2024)
Yu Ying Chiu, Liwei Jiang, Maria Antoniak, Chan Young Park, Shuyue Stella Li, Mehar Bhatia, Sahithya Ravi, Yulia Tsvetkov, Vered Shwartz, and Yejin Choi.
Culturalteaming: Ai-assisted interactive red-teaming for challenging llms’ (lack of) multicultural knowledge.
*CoRR*, abs/2404.06664, 2024. doi: 10.48550/ARXIV.2404.06664. URL https://doi.org/10.48550/arXiv.2404.06664. -
Cho et al. (2014)
Kyunghyun Cho, Bart van Merrienboer, Çaglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio.
Learning phrase representations using RNN encoder-decoder for statistical machine translation.
In Alessandro Moschitti, Bo Pang, and Walter Daelemans (eds.),
*Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL*, pp. 1724–1734. ACL, 2014. doi: 10.3115/V1/D14-1179. URL https://doi.org/10.3115/v1/d14-1179. -
Choenni et al. (2024)
Rochelle Choenni, Anne Lauscher, and Ekaterina Shutova.
The echoes of multilinguality: Tracing cultural value shifts during lm fine-tuning.
*CoRR*, abs/2405.12744, 2024. doi: 10.48550/ARXIV.2405.12744. URL https://doi.org/10.48550/arXiv.2405.12744. - Chowdhery et al. (2022) Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022. URL https://arxiv.org/abs/2204.02311.
-
Chowdhery et al. (2023)
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel.
Palm: Scaling language modeling with pathways.
*J. Mach. Learn. Res.*, 24:240:1–240:113, 2023. URL http://jmlr.org/papers/v24/22-1144.html. -
Clark et al. (2020)
Jonathan H. Clark, Jennimaria Palomaki, Vitaly Nikolaev, Eunsol Choi, Dan Garrette, Michael Collins, and Tom Kwiatkowski.
Tydi QA: A benchmark for information-seeking question answering in typologically diverse languages.
*Trans. Assoc. Comput. Linguistics*, 8:454–470, 2020. doi: 10.1162/TACL\_A\_00317. URL https://doi.org/10.1162/tacl_a_00317. - Computer (2023) Together Computer. Redpajama: an open dataset for training large language models, October 2023. URL https://github.com/togethercomputer/RedPajama-Data.
-
Conneau & Lample (2019a)
Alexis Conneau and Guillaume Lample.
Cross-lingual language model pretraining.
In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett (eds.),
*Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada*, pp. 7057–7067, 2019a. URL https://proceedings.neurips.cc/paper/2019/hash/c04c19c2c2474dbf5f7ac4372c5b9af1-Abstract.html. -
Conneau & Lample (2019b)
Alexis Conneau and Guillaume Lample.
Cross-lingual language model pretraining.
*Advances in neural information processing systems*, 32, 2019b. -
Conneau et al. (2018)
Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov.
XNLI: evaluating cross-lingual sentence representations.
In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.),
*Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018*, pp. 2475–2485. Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1269. URL https://doi.org/10.18653/v1/d18-1269. -
Conneau et al. (2019)
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov.
Unsupervised cross-lingual representation learning at scale.
*arXiv preprint arXiv:1911.02116*, 2019. -
Conneau et al. (2020a)
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov.
Unsupervised cross-lingual representation learning at scale.
In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (eds.),
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020*, pp. 8440–8451. Association for Computational Linguistics, 2020a. doi: 10.18653/V1/2020.ACL-MAIN.747. URL https://doi.org/10.18653/v1/2020.acl-main.747. -
Conneau et al. (2020b)
Alexis Conneau, Shijie Wu, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov.
Emerging cross-lingual structure in pretrained language models.
In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (eds.),
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020*, pp. 6022–6034. Association for Computational Linguistics, 2020b. doi: 10.18653/V1/2020.ACL-MAIN.536. URL https://doi.org/10.18653/v1/2020.acl-main.536. - Conover et al. (2023a) Mike Conover, Matt Hayes, Ankit Mathur, Xiangrui Meng, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free dolly: Introducing the world’s first truly open instruction-tuned llm. 2023a. URL https://www.databricks.com/.
- Conover et al. (2023b) Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free dolly: Introducing the world’s first truly open instruction-tuned llm, 2023b. URL https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm.
-
Costa-jussà et al. (2022)
Marta R Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al.
No language left behind: Scaling human-centered machine translation.
*arXiv preprint arXiv:2207.04672*, 2022. -
Cui et al. (2023a)
Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun.
Ultrafeedback: Boosting language models with high-quality feedback.
*CoRR*, abs/2310.01377, 2023a. doi: 10.48550/ARXIV.2310.01377. URL https://doi.org/10.48550/arXiv.2310.01377. -
Cui et al. (2023b)
Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and Li Yuan.
Chatlaw: Open-source legal large language model with integrated external knowledge bases.
*CoRR*, abs/2306.16092, 2023b. doi: 10.48550/ARXIV.2306.16092. URL https://doi.org/10.48550/arXiv.2306.16092. -
Cui et al. (2024)
Menglong Cui, Jiangcun Du, Shaolin Zhu, and Deyi Xiong.
Efficiently exploring large language models for document-level machine translation with in-context learning.
*arXiv preprint arXiv:2406.07081*, 2024. -
Cui & Yao (2024)
Yiming Cui and Xin Yao.
Rethinking LLM language adaptation: A case study on chinese mixtral.
*CoRR*, abs/2403.01851, 2024. doi: 10.48550/ARXIV.2403.01851. URL https://doi.org/10.48550/arXiv.2403.01851. -
Cui et al. (2023c)
Yiming Cui, Ziqing Yang, and Xin Yao.
Efficient and effective text encoding for chinese llama and alpaca.
*CoRR*, abs/2304.08177, 2023c. doi: 10.48550/ARXIV.2304.08177. URL https://doi.org/10.48550/arXiv.2304.08177. -
Dai et al. (2024)
Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al.
Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models.
*arXiv preprint arXiv:2401.06066*, 2024. -
de Varda & Marelli (2024)
Andrea Gregor de Varda and Marco Marelli.
The emergence of semantic units in massively multilingual models.
In Nicoletta Calzolari, Min-Yen Kan, Véronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (eds.),
*Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy*, pp. 15910–15921. ELRA and ICCL, 2024. URL https://aclanthology.org/2024.lrec-main.1382. -
de Wynter et al. (2024)
Adrian de Wynter, Ishaan Watts, Nektar Ege Altintoprak, Tua Wongsangaroonsri, Minghui Zhang, Noura Farra, Lena Baur, Samantha Claudet, Pavel Gajdusek, Can Gören, Qilong Gu, Anna Kaminska, Tomasz Kaminski, Ruby Kuo, Akiko Kyuba, Jongho Lee, Kartik Mathur, Petter Merok, Ivana Milovanovic, Nani Paananen, Vesa-Matti Paananen, Anna Pavlenko, Bruno Pereira Vidal, Luciano Strika, Yueh Tsao, Davide Turcato, Oleksandr Vakhno, Judit Velcsov, Anna Vickers, Stéphanie Visser, Herdyan Widarmanto, Andrey Zaikin, and Si-Qing Chen.
RTP-LX: can llms evaluate toxicity in multilingual scenarios?
*CoRR*, abs/2404.14397, 2024. doi: 10.48550/ARXIV.2404.14397. URL https://doi.org/10.48550/arXiv.2404.14397. -
DeepSeek-AI et al. (2024a)
DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, Hao Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, Tao Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen
Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, and Xiaowen Sun.
Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model.
*CoRR*, abs/2405.04434, 2024a. doi: 10.48550/ARXIV.2405.04434. URL https://doi.org/10.48550/arXiv.2405.04434. -
DeepSeek-AI et al. (2024b)
DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, Hao Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, Tao Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen
Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, and Xiaowen Sun.
Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model.
*CoRR*, abs/2405.04434, 2024b. doi: 10.48550/ARXIV.2405.04434. URL https://doi.org/10.48550/arXiv.2405.04434. -
Deng et al. (2023)
Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, and Lidong Bing.
Multilingual jailbreak challenges in large language models.
*CoRR*, abs/2310.06474, 2023. doi: 10.48550/ARXIV.2310.06474. URL https://doi.org/10.48550/arXiv.2310.06474. -
Deshpande et al. (2022)
Ameet Deshpande, Partha Talukdar, and Karthik Narasimhan.
When is BERT multilingual? isolating crucial ingredients for cross-lingual transfer.
In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz (eds.),
*Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022*, pp. 3610–3623. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.NAACL-MAIN.264. URL https://doi.org/10.18653/v1/2022.naacl-main.264. -
Dettmers et al. (2023)
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer.
Qlora: Efficient finetuning of quantized llms.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.),
*Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html. -
Devlin et al. (2018)
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
Bert: Pre-training of deep bidirectional transformers for language understanding.
*arXiv preprint arXiv:1810.04805*, 2018. -
Devlin et al. (2019)
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
BERT: pre-training of deep bidirectional transformers for language understanding.
In Jill Burstein, Christy Doran, and Thamar Solorio (eds.),
*Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)*, pp. 4171–4186. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1423. URL https://doi.org/10.18653/v1/n19-1423. -
Doddapaneni et al. (2022)
Sumanth Doddapaneni, Rahul Aralikatte, Gowtham Ramesh, Shreya Goyal, Mitesh M. Khapra, Anoop Kunchukuttan, and Pratyush Kumar.
Indicxtreme: A multi-task benchmark for evaluating indic languages.
*CoRR*, abs/2212.05409, 2022. doi: 10.48550/ARXIV.2212.05409. URL https://doi.org/10.48550/arXiv.2212.05409. -
Doddapaneni et al. (2024)
Sumanth Doddapaneni, Mohammed Safi Ur Rahman Khan, Dilip Venkatesh, Raj Dabre, Anoop Kunchukuttan, and Mitesh M Khapra.
Cross-lingual auto evaluation for assessing multilingual llms.
*arXiv preprint arXiv:2410.13394*, 2024. -
Eisele & Chen (2010)
Andreas Eisele and Yu Chen.
Multiun: A multilingual corpus from united nation documents.
In Nicoletta Calzolari, Khalid Choukri, Bente Maegaard, Joseph Mariani, Jan Odijk, Stelios Piperidis, Mike Rosner, and Daniel Tapias (eds.),
*Proceedings of the International Conference on Language Resources and Evaluation, LREC 2010, 17-23 May 2010, Valletta, Malta*. European Language Resources Association, 2010. URL http://www.lrec-conf.org/proceedings/lrec2010/summaries/686.html. -
Ethayarajh et al. (2024)
Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela.
KTO: model alignment as prospect theoretic optimization.
*CoRR*, abs/2402.01306, 2024. doi: 10.48550/ARXIV.2402.01306. URL https://doi.org/10.48550/arXiv.2402.01306. -
Faisal & Anastasopoulos (2024)
Fahim Faisal and Antonios Anastasopoulos.
An efficient approach for studying cross-lingual transfer in multilingual language models.
*arXiv preprint arXiv:2403.20088*, 2024. -
Fedus et al. (2022)
William Fedus, Barret Zoph, and Noam Shazeer.
Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity.
*Journal of Machine Learning Research*, 23(120):1–39, 2022. -
Feng et al. (2020)
Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou.
Codebert: A pre-trained model for programming and natural languages.
In Trevor Cohn, Yulan He, and Yang Liu (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020*, volume EMNLP 2020 of*Findings of ACL*, pp. 1536–1547. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.FINDINGS-EMNLP.139. URL https://doi.org/10.18653/v1/2020.findings-emnlp.139. -
Ferron et al. (2023)
Amila Ferron, Amber Shore, Ekata Mitra, and Ameeta Agrawal.
MEEP: is this engaging? prompting large language models for dialogue evaluation in multilingual settings.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 2078–2100. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-EMNLP.137. URL https://doi.org/10.18653/v1/2023.findings-emnlp.137. -
Fried et al. (2023)
Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Scott Yih, Luke Zettlemoyer, and Mike Lewis.
Incoder: A generative model for code infilling and synthesis.
In
*The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. URL https://openreview.net/pdf?id=hQwb-lbM6EL. -
Fu et al. (2024)
Chengpeng Fu, Xiaocheng Feng, Yichong Huang, Wenshuai Huo, Baohang Li, Hui Wang, Bin Qin, and Ting Liu.
Relay decoding: Concatenating large language models for machine translation.
*CoRR*, abs/2405.02933, 2024. doi: 10.48550/ARXIV.2405.02933. URL https://doi.org/10.48550/arXiv.2405.02933. - Fu et al. (2023) Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, and Christopher Re. Hungry hungry hippos: Towards language modeling with state space models, 2023. URL https://arxiv.org/abs/2212.14052.
-
Fujii et al. (2024)
Kazuki Fujii, Taishi Nakamura, Mengsay Loem, Hiroki Iida, Masanari Ohi, Kakeru Hattori, Hirai Shota, Sakae Mizuki, Rio Yokota, and Naoaki Okazaki.
Continual pre-training for cross-lingual LLM adaptation: Enhancing japanese language capabilities.
*CoRR*, abs/2404.17790, 2024. doi: 10.48550/ARXIV.2404.17790. URL https://doi.org/10.48550/arXiv.2404.17790. -
Fung et al. (2023)
Yi Fung, Tuhin Chakrabarty, Hao Guo, Owen Rambow, Smaranda Muresan, and Heng Ji.
NORMSAGE: multi-lingual multi-cultural norm discovery from conversations on-the-fly.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pp. 15217–15230. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.941. URL https://doi.org/10.18653/v1/2023.emnlp-main.941. -
Fung et al. (2024)
Yi Fung, Ruining Zhao, Jae Doo, Chenkai Sun, and Heng Ji.
Massively multi-cultural knowledge acquisition & LM benchmarking.
*CoRR*, abs/2402.09369, 2024. doi: 10.48550/ARXIV.2402.09369. URL https://doi.org/10.48550/arXiv.2402.09369. -
Gala et al. (2023)
Jay P. Gala, Pranjal A. Chitale, Raghavan AK, Sumanth Doddapaneni, Varun Gumma, Aswanth Kumar, Janki Nawale, Anupama Sujatha, Ratish Puduppully, Vivek Raghavan, Pratyush Kumar, Mitesh M. Khapra, Raj Dabre, and Anoop Kunchukuttan.
Indictrans2: Towards high-quality and accessible machine translation models for all 22 scheduled indian languages.
*CoRR*, abs/2305.16307, 2023. doi: 10.48550/ARXIV.2305.16307. URL https://doi.org/10.48550/arXiv.2305.16307. -
Ganguli et al. (2022)
Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El Showk, Stanislav Fort, Zac Hatfield-Dodds, Tom Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom Brown, Nicholas Joseph, Sam McCandlish, Chris Olah, Jared Kaplan, and Jack Clark.
Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned.
*CoRR*, abs/2209.07858, 2022. doi: 10.48550/ARXIV.2209.07858. URL https://doi.org/10.48550/arXiv.2209.07858. -
Gao et al. (2024)
Pengzhi Gao, Zhongjun He, Hua Wu, and Haifeng Wang.
Towards boosting many-to-many multilingual machine translation with large language models.
*CoRR*, abs/2401.05861, 2024. doi: 10.48550/ARXIV.2401.05861. URL https://doi.org/10.48550/arXiv.2401.05861. -
García-Ferrero et al. (2024)
Iker García-Ferrero, Rodrigo Agerri, Aitziber Atutxa Salazar, Elena Cabrio, Iker de la Iglesia, Alberto Lavelli, Bernardo Magnini, Benjamin Molinet, Johana Ramirez-Romero, German Rigau, et al.
Medical mt5: an open-source multilingual text-to-text llm for the medical domain.
*arXiv preprint arXiv:2404.07613*, 2024. -
Gehman et al. (2020)
Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A. Smith.
Realtoxicityprompts: Evaluating neural toxic degeneration in language models.
In Trevor Cohn, Yulan He, and Yang Liu (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020*, volume EMNLP 2020 of*Findings of ACL*, pp. 3356–3369. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.FINDINGS-EMNLP.301. URL https://doi.org/10.18653/v1/2020.findings-emnlp.301. -
Gooding & Mansoor (2023)
Sian Gooding and Hassan Mansoor.
The impact of preference agreement in reinforcement learning from human feedback: A case study in summarization.
*CoRR*, abs/2311.04919, 2023. doi: 10.48550/ARXIV.2311.04919. URL https://doi.org/10.48550/arXiv.2311.04919. -
Goyal et al. (2022)
Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan.
The flores-101 evaluation benchmark for low-resource and multilingual machine translation.
*Trans. Assoc. Comput. Linguistics*, 10:522–538, 2022. doi: 10.1162/TACL\_A\_00474. URL https://doi.org/10.1162/tacl_a_00474. -
Gu & Dao (2023)
Albert Gu and Tri Dao.
Mamba: Linear-time sequence modeling with selective state spaces.
*arXiv preprint arXiv:2312.00752*, 2023. - Gu et al. (2022) Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces, 2022. URL https://arxiv.org/abs/2111.00396.
-
Gurgurov et al. (2024)
Daniil Gurgurov, Tanja Bäumel, and Tatiana Anikina.
Multilingual large language models and curse of multilinguality.
*arXiv preprint arXiv:2406.10602*, 2024. -
Hada et al. (2024a)
Rishav Hada, Varun Gumma, Mohamed Ahmed, Kalika Bali, and Sunayana Sitaram.
METAL: towards multilingual meta-evaluation.
*CoRR*, abs/2404.01667, 2024a. doi: 10.48550/ARXIV.2404.01667. URL https://doi.org/10.48550/arXiv.2404.01667. -
Hada et al. (2024b)
Rishav Hada, Varun Gumma, Adrian de Wynter, Harshita Diddee, Mohamed Ahmed, Monojit Choudhury, Kalika Bali, and Sunayana Sitaram.
Are large language model-based evaluators the solution to scaling up multilingual evaluation?
In Yvette Graham and Matthew Purver (eds.),
*Findings of the Association for Computational Linguistics: EACL 2024, St. Julian’s, Malta, March 17-22, 2024*, pp. 1051–1070. Association for Computational Linguistics, 2024b. URL https://aclanthology.org/2024.findings-eacl.71. -
Hasan et al. (2021)
Tahmid Hasan, Abhik Bhattacharjee, Md. Saiful Islam, Kazi Samin Mubasshir, Yuan-Fang Li, Yong-Bin Kang, M. Sohel Rahman, and Rifat Shahriyar.
Xl-sum: Large-scale multilingual abstractive summarization for 44 languages.
In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.),
*Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021*, volume ACL/IJCNLP 2021 of*Findings of ACL*, pp. 4693–4703. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.FINDINGS-ACL.413. URL https://doi.org/10.18653/v1/2021.findings-acl.413. -
Hershcovich et al. (2022)
Daniel Hershcovich, Stella Frank, Heather C. Lent, Miryam de Lhoneux, Mostafa Abdou, Stephanie Brandl, Emanuele Bugliarello, Laura Cabello Piqueras, Ilias Chalkidis, Ruixiang Cui, Constanza Fierro, Katerina Margatina, Phillip Rust, and Anders Søgaard.
Challenges and strategies in cross-cultural NLP.
In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.),
*Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022*, pp. 6997–7013. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.ACL-LONG.482. URL https://doi.org/10.18653/v1/2022.acl-long.482. -
Holtermann et al. (2024)
Carolin Holtermann, Paul Röttger, Timm Dill, and Anne Lauscher.
Evaluating the elementary multilingual capabilities of large language models with multiq.
*CoRR*, abs/2403.03814, 2024. doi: 10.48550/ARXIV.2403.03814. URL https://doi.org/10.48550/arXiv.2403.03814. -
Hu et al. (2024)
Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al.
Minicpm: Unveiling the potential of small language models with scalable training strategies.
*arXiv preprint arXiv:2404.06395*, 2024. -
Hua et al. (2022)
Wenyue Hua, Yuchen Zhang, Zhe Chen, Josie Li, and Melanie Weber.
Legalrelectra: Mixed-domain language modeling for long-range legal text comprehension.
*CoRR*, abs/2212.08204, 2022. doi: 10.48550/ARXIV.2212.08204. URL https://doi.org/10.48550/arXiv.2212.08204. -
Huang & Yang (2023)
Jing Huang and Diyi Yang.
Culturally aware natural language inference.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 7591–7609. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-EMNLP.509. URL https://doi.org/10.18653/v1/2023.findings-emnlp.509. -
Huang et al. (2023)
Quzhe Huang, Mingxu Tao, Zhenwei An, Chen Zhang, Cong Jiang, Zhibin Chen, Zirui Wu, and Yansong Feng.
Lawyer llama technical report.
*CoRR*, abs/2305.15062, 2023. doi: 10.48550/ARXIV.2305.15062. URL https://doi.org/10.48550/arXiv.2305.15062. - Huerta-Enochian & Ko (2024) Mathew Huerta-Enochian and Seung Yong Ko. Instruction fine-tuning: Does prompt loss matter?, 2024.
-
Husain et al. (2024)
Jaavid Aktar Husain, Raj Dabre, Aswanth Kumar, Ratish Puduppully, and Anoop Kunchukuttan.
Romansetu: Efficiently unlocking multilingual capabilities of large language models models via romanization.
*CoRR*, abs/2401.14280, 2024. doi: 10.48550/ARXIV.2401.14280. URL https://doi.org/10.48550/arXiv.2401.14280. -
Hutchins (1999)
John Hutchins.
Retrospect and prospect in computer-based translation.
In
*Proceedings of Machine Translation Summit VII*, pp. 30–36, Singapore, Singapore, September 13-17 1999. URL https://aclanthology.org/1999.mtsummit-1.5. - Ian Kivlichan (2020) Julia Elliott et al. Ian Kivlichan, Jeffrey Sorensen. Jigsaw multilingual toxic comment classification, 2020. URL https://kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification.
- Jain et al. (2024) Devansh Jain, Priyanshu Kumar, Samuel Gehman, Xuhui Zhou, Thomas Hartvigsen, and Maarten Sap. Polyglotoxicityprompts: Multilingual evaluation of neural toxic degeneration in large language models, 2024.
-
Ji & Chen (2024)
Shaoxiong Ji and Pinzhen Chen.
Lucky 52: How many languages are needed to instruction fine-tune large language models?
*CoRR*, abs/2404.04850, 2024. doi: 10.48550/ARXIV.2404.04850. URL https://doi.org/10.48550/arXiv.2404.04850. -
Ji et al. (2021)
Yanrong Ji, Zhihan Zhou, Han Liu, and Ramana V. Davuluri.
DNABERT: pre-trained bidirectional encoder representations from transformers model for dna-language in genome.
*Bioinform.*, 37(15):2112–2120, 2021. doi: 10.1093/BIOINFORMATICS/BTAB083. URL https://doi.org/10.1093/bioinformatics/btab083. -
Jiang et al. (2023)
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al.
Mistral 7b.
*arXiv preprint arXiv:2310.06825*, 2023. -
Jiang et al. (2024)
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al.
Mixtral of experts.
*arXiv preprint arXiv:2401.04088*, 2024. -
Jiang et al. (2020)
Zhengbao Jiang, Antonios Anastasopoulos, Jun Araki, Haibo Ding, and Graham Neubig.
X-FACTR: multilingual factual knowledge retrieval from pretrained language models.
In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.),
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pp. 5943–5959. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.EMNLP-MAIN.479. URL https://doi.org/10.18653/v1/2020.emnlp-main.479. -
K et al. (2020)
Karthikeyan K, Zihan Wang, Stephen Mayhew, and Dan Roth.
Cross-lingual ability of multilingual BERT: an empirical study.
In
*8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net, 2020. URL https://openreview.net/forum?id=HJeT3yrtDr. -
Kabra et al. (2023)
Anubha Kabra, Emmy Liu, Simran Khanuja, Alham Fikri Aji, Genta Indra Winata, Samuel Cahyawijaya, Aremu Anuoluwapo, Perez Ogayo, and Graham Neubig.
Multi-lingual and multi-cultural figurative language understanding.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 8269–8284. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-ACL.525. URL https://doi.org/10.18653/v1/2023.findings-acl.525. -
Kakwani et al. (2020)
Divyanshu Kakwani, Anoop Kunchukuttan, Satish Golla, Gokul N.C., Avik Bhattacharyya, Mitesh M. Khapra, and Pratyush Kumar.
IndicNLPSuite: Monolingual corpora, evaluation benchmarks and pre-trained multilingual language models for Indian languages.
In Trevor Cohn, Yulan He, and Yang Liu (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2020*, pp. 4948–4961, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.445. URL https://aclanthology.org/2020.findings-emnlp.445. -
Kanade et al. (2020)
Aditya Kanade, Petros Maniatis, Gogul Balakrishnan, and Kensen Shi.
Learning and evaluating contextual embedding of source code.
In
*Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event*, volume 119 of*Proceedings of Machine Learning Research*, pp. 5110–5121. PMLR, 2020. URL http://proceedings.mlr.press/v119/kanade20a.html. -
Kaneko et al. (2022)
Masahiro Kaneko, Aizhan Imankulova, Danushka Bollegala, and Naoaki Okazaki.
Gender bias in masked language models for multiple languages.
In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz (eds.),
*Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022*, pp. 2740–2750. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.NAACL-MAIN.197. URL https://doi.org/10.18653/v1/2022.naacl-main.197. -
Kassner et al. (2021)
Nora Kassner, Philipp Dufter, and Hinrich Schütze.
Multilingual LAMA: investigating knowledge in multilingual pretrained language models.
In Paola Merlo, Jörg Tiedemann, and Reut Tsarfaty (eds.),
*Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021*, pp. 3250–3258. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EACL-MAIN.284. URL https://doi.org/10.18653/v1/2021.eacl-main.284. -
Keleg & Magdy (2023)
Amr Keleg and Walid Magdy.
DLAMA: A framework for curating culturally diverse facts for probing the knowledge of pretrained language models.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 6245–6266. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-ACL.389. URL https://doi.org/10.18653/v1/2023.findings-acl.389. -
Kew et al. (2023)
Tannon Kew, Florian Schottmann, and Rico Sennrich.
Turning english-centric llms into polyglots: How much multilinguality is needed?
*CoRR*, abs/2312.12683, 2023. doi: 10.48550/ARXIV.2312.12683. URL https://doi.org/10.48550/arXiv.2312.12683. -
Khanuja et al. (2020)
Simran Khanuja, Sandipan Dandapat, Anirudh Srinivasan, Sunayana Sitaram, and Monojit Choudhury.
Gluecos: An evaluation benchmark for code-switched NLP.
In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (eds.),
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020*, pp. 3575–3585. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.ACL-MAIN.329. URL https://doi.org/10.18653/v1/2020.acl-main.329. -
Kim et al. (2024a)
Seungduk Kim, Seungtaek Choi, and Myeongho Jeong.
Efficient and effective vocabulary expansion towards multilingual large language models.
*CoRR*, abs/2402.14714, 2024a. doi: 10.48550/ARXIV.2402.14714. URL https://doi.org/10.48550/arXiv.2402.14714. -
Kim et al. (2024b)
Seungduk Kim, Seungtaek Choi, and Myeongho Jeong.
Efficient and effective vocabulary expansion towards multilingual large language models.
*arXiv preprint arXiv:2402.14714*, 2024b. -
Kirk et al. (2024)
Hannah Rose Kirk, Alexander Whitefield, Paul Röttger, Andrew M. Bean, Katerina Margatina, Juan Ciro, Rafael Mosquera, Max Bartolo, Adina Williams, He He, Bertie Vidgen, and Scott A. Hale.
The PRISM alignment project: What participatory, representative and individualised human feedback reveals about the subjective and multicultural alignment of large language models.
*CoRR*, abs/2404.16019, 2024. doi: 10.48550/ARXIV.2404.16019. URL https://doi.org/10.48550/arXiv.2404.16019. -
Kirk et al. (2023)
Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu.
Understanding the effects of RLHF on LLM generalisation and diversity.
*CoRR*, abs/2310.06452, 2023. doi: 10.48550/ARXIV.2310.06452. URL https://doi.org/10.48550/arXiv.2310.06452. -
Kocetkov et al. (2022)
Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries.
The stack: 3 TB of permissively licensed source code.
*CoRR*, abs/2211.15533, 2022. doi: 10.48550/ARXIV.2211.15533. URL https://doi.org/10.48550/arXiv.2211.15533. -
Kojima et al. (2024a)
Takeshi Kojima, Itsuki Okimura, Yusuke Iwasawa, Hitomi Yanaka, and Yutaka Matsuo.
On the multilingual ability of decoder-based pre-trained language models: Finding and controlling language-specific neurons.
*CoRR*, abs/2404.02431, 2024a. doi: 10.48550/ARXIV.2404.02431. URL https://doi.org/10.48550/arXiv.2404.02431. -
Kojima et al. (2024b)
Takeshi Kojima, Itsuki Okimura, Yusuke Iwasawa, Hitomi Yanaka, and Yutaka Matsuo.
On the multilingual ability of decoder-based pre-trained language models: Finding and controlling language-specific neurons.
*arXiv preprint arXiv:2404.02431*, 2024b. -
Köpf et al. (2023)
Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick.
Openassistant conversations - democratizing large language model alignment.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.),
*Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/949f0f8f32267d297c2d4e3ee10a2e7e-Abstract-Datasets_and_Benchmarks.html. -
Kovac et al. (2023)
Grgur Kovac, Masataka Sawayama, Rémy Portelas, Cédric Colas, Peter Ford Dominey, and Pierre-Yves Oudeyer.
Large language models as superpositions of cultural perspectives.
*CoRR*, abs/2307.07870, 2023. doi: 10.48550/ARXIV.2307.07870. URL https://doi.org/10.48550/arXiv.2307.07870. -
Kraljevic et al. (2021)
Zeljko Kraljevic, Anthony Shek, Daniel Bean, Rebecca Bendayan, James T. Teo, and Richard J. B. Dobson.
Medgpt: Medical concept prediction from clinical narratives.
*CoRR*, abs/2107.03134, 2021. URL https://arxiv.org/abs/2107.03134. -
Ladhak et al. (2020)
Faisal Ladhak, Esin Durmus, Claire Cardie, and Kathleen R. McKeown.
Wikilingua: A new benchmark dataset for multilingual abstractive summarization.
In Trevor Cohn, Yulan He, and Yang Liu (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020*, volume EMNLP 2020 of*Findings of ACL*, pp. 4034–4048. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.FINDINGS-EMNLP.360. URL https://doi.org/10.18653/v1/2020.findings-emnlp.360. -
Lai et al. (2023a)
Viet Dac Lai, Nghia Trung Ngo, Amir Pouran Ben Veyseh, Hieu Man, Franck Dernoncourt, Trung Bui, and Thien Huu Nguyen.
Chatgpt beyond english: Towards a comprehensive evaluation of large language models in multilingual learning.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 13171–13189. Association for Computational Linguistics, 2023a. doi: 10.18653/V1/2023.FINDINGS-EMNLP.878. URL https://doi.org/10.18653/v1/2023.findings-emnlp.878. -
Lai et al. (2023b)
Viet Dac Lai, Chien Van Nguyen, Nghia Trung Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan A. Rossi, and Thien Huu Nguyen.
Okapi: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback.
In Yansong Feng and Els Lefever (eds.),
*Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023 - System Demonstrations, Singapore, December 6-10, 2023*, pp. 318–327. Association for Computational Linguistics, 2023b. doi: 10.18653/V1/2023.EMNLP-DEMO.28. URL https://doi.org/10.18653/v1/2023.emnlp-demo.28. -
Laurençon et al. (2023)
Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, Jörg Frohberg, Mario Sasko, Quentin Lhoest, Angelina McMillan-Major, Gérard Dupont, Stella Biderman, Anna Rogers, Loubna Ben Allal, Francesco De Toni, Giada Pistilli, Olivier Nguyen, Somaieh Nikpoor, Maraim Masoud, Pierre Colombo, Javier de la Rosa, Paulo Villegas, Tristan Thrush, Shayne Longpre, Sebastian Nagel, Leon Weber, Manuel Muñoz, Jian Zhu, Daniel van Strien, Zaid Alyafeai, Khalid Almubarak, Minh Chien Vu, Itziar Gonzalez-Dios, Aitor Soroa, Kyle Lo, Manan Dey, Pedro Ortiz Suarez, Aaron Gokaslan, Shamik Bose, David Ifeoluwa Adelani, Long Phan, Hieu Tran, Ian Yu, Suhas Pai, Jenny Chim, Violette Lepercq, Suzana Ilic, Margaret Mitchell, Sasha Luccioni, and Yacine Jernite.
The bigscience ROOTS corpus: A 1.6tb composite multilingual dataset.
*CoRR*, abs/2303.03915, 2023. doi: 10.48550/ARXIV.2303.03915. URL https://doi.org/10.48550/arXiv.2303.03915. - Le Scao et al. (2023) Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. 2023.
-
Lee et al. (2020)
Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang.
Biobert: a pre-trained biomedical language representation model for biomedical text mining.
*Bioinform.*, 36(4):1234–1240, 2020. doi: 10.1093/BIOINFORMATICS/BTZ682. URL https://doi.org/10.1093/bioinformatics/btz682. -
Lee et al. (2022)
Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini.
Deduplicating training data makes language models better.
In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.),
*Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 8424–8445, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.577. URL https://aclanthology.org/2022.acl-long.577. -
Lepikhin et al. (2020)
Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen.
Gshard: Scaling giant models with conditional computation and automatic sharding.
*arXiv preprint arXiv:2006.16668*, 2020. -
Lester et al. (2021)
Brian Lester, Rami Al-Rfou, and Noah Constant.
The power of scale for parameter-efficient prompt tuning.
In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.),
*Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021*, pp. 3045–3059. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EMNLP-MAIN.243. URL https://doi.org/10.18653/v1/2021.emnlp-main.243. -
Levy et al. (2022)
Sharon Levy, Emily Allaway, Melanie Subbiah, Lydia B. Chilton, Desmond Patton, Kathleen R. McKeown, and William Yang Wang.
Safetext: A benchmark for exploring physical safety in language models.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 2407–2421. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.154. URL https://doi.org/10.18653/v1/2022.emnlp-main.154. -
Lewis et al. (2019)
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer.
Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.
*arXiv preprint arXiv:1910.13461*, 2019. -
Lewis et al. (2020)
Patrick S. H. Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk.
MLQA: evaluating cross-lingual extractive question answering.
In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault (eds.),
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020*, pp. 7315–7330. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.ACL-MAIN.653. URL https://doi.org/10.18653/v1/2020.acl-main.653. -
Li et al. (2024a)
Cheng Li, Mengzhou Chen, Jindong Wang, Sunayana Sitaram, and Xing Xie.
Culturellm: Incorporating cultural differences into large language models.
*CoRR*, abs/2402.10946, 2024a. doi: 10.48550/ARXIV.2402.10946. URL https://doi.org/10.48550/arXiv.2402.10946. -
Li et al. (2024b)
Cheng Li, Damien Teney, Linyi Yang, Qingsong Wen, Xing Xie, and Jindong Wang.
Culturepark: Boosting cross-cultural understanding in large language models.
*CoRR*, abs/2405.15145, 2024b. doi: 10.48550/ARXIV.2405.15145. URL https://doi.org/10.48550/arXiv.2405.15145. -
Li et al. (2024c)
Chong Li, Shaonan Wang, Jiajun Zhang, and Chengqing Zong.
Improving in-context learning of multilingual generative language models with cross-lingual alignment.
In
*Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pp. 8051–8069, 2024c. -
Li et al. (2023a)
Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem.
CAMEL: communicative agents for "mind" exploration of large language model society.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.),
*Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023a. URL http://papers.nips.cc/paper_files/paper/2023/hash/a3621ee907def47c1b952ade25c67698-Abstract-Conference.html. -
Li et al. (2023b)
Haitao Li, Qingyao Ai, Jia Chen, Qian Dong, Yueyue Wu, Yiqun Liu, Chong Chen, and Qi Tian.
SAILER: structure-aware pre-trained language model for legal case retrieval.
In Hsin-Hsi Chen, Wei-Jou (Edward) Duh, Hen-Hsen Huang, Makoto P. Kato, Josiane Mothe, and Barbara Poblete (eds.),
*Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023*, pp. 1035–1044. ACM, 2023b. doi: 10.1145/3539618.3591761. URL https://doi.org/10.1145/3539618.3591761. -
Li et al. (2023c)
Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin.
Bactrian-x : A multilingual replicable instruction-following model with low-rank adaptation.
*CoRR*, abs/2305.15011, 2023c. doi: 10.48550/ARXIV.2305.15011. URL https://doi.org/10.48550/arXiv.2305.15011. -
Li et al. (2023d)
Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin.
Bactrian-x : A multilingual replicable instruction-following model with low-rank adaptation.
*CoRR*, abs/2305.15011, 2023d. doi: 10.48550/ARXIV.2305.15011. URL https://doi.org/10.48550/arXiv.2305.15011. -
Li et al. (2024d)
Huihan Li, Liwei Jiang, Jena D. Huang, Hyunwoo Kim, Sebastin Santy, Taylor Sorensen, Bill Yuchen Lin, Nouha Dziri, Xiang Ren, and Yejin Choi.
CULTURE-GEN: revealing global cultural perception in language models through natural language prompting.
*CoRR*, abs/2404.10199, 2024d. doi: 10.48550/ARXIV.2404.10199. URL https://doi.org/10.48550/arXiv.2404.10199. -
Li et al. (2023e)
Jiahuan Li, Hao Zhou, Shujian Huang, Shanbo Chen, and Jiajun Chen.
Eliciting the translation ability of large language models via multilingual finetuning with translation instructions.
*CoRR*, abs/2305.15083, 2023e. doi: 10.48550/ARXIV.2305.15083. URL https://doi.org/10.48550/arXiv.2305.15083. -
Li et al. (2024e)
Jie Li, Yi Liu, Chongyang Liu, Ling Shi, Xiaoning Ren, Yaowen Zheng, Yang Liu, and Yinxing Xue.
A cross-language investigation into jailbreak attacks in large language models.
*CoRR*, abs/2401.16765, 2024e. doi: 10.48550/ARXIV.2401.16765. URL https://doi.org/10.48550/arXiv.2401.16765. -
Li et al. (2023f)
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy V, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Moustafa-Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas
Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries.
Starcoder: may the source be with you!
*CoRR*, abs/2305.06161, 2023f. doi: 10.48550/ARXIV.2305.06161. URL https://doi.org/10.48550/arXiv.2305.06161. -
Li et al. (2023g)
Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis.
Self-alignment with instruction backtranslation.
*CoRR*, abs/2308.06259, 2023g. doi: 10.48550/ARXIV.2308.06259. URL https://doi.org/10.48550/arXiv.2308.06259. -
Li & Liang (2021)
Xiang Lisa Li and Percy Liang.
Prefix-tuning: Optimizing continuous prompts for generation.
In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.),
*Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021*, pp. 4582–4597. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.ACL-LONG.353. URL https://doi.org/10.18653/v1/2021.acl-long.353. -
Li et al. (2022)
Yujia Li, David H. Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals.
Competition-level code generation with alphacode.
*CoRR*, abs/2203.07814, 2022. doi: 10.48550/ARXIV.2203.07814. URL https://doi.org/10.48550/arXiv.2203.07814. -
Li et al. (2023h)
Yunxiang Li, Zihan Li, Kai Zhang, Ruilong Dan, and You Zhang.
Chatdoctor: A medical chat model fine-tuned on llama model using medical domain knowledge.
*CoRR*, abs/2303.14070, 2023h. doi: 10.48550/ARXIV.2303.14070. URL https://doi.org/10.48550/arXiv.2303.14070. -
Li et al. (2024f)
Zihao Li, Shaoxiong Ji, Timothee Mickus, Vincent Segonne, and Jörg Tiedemann.
A comparison of language modeling and translation as multilingual pretraining objectives.
In
*Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pp. 15882–15894, 2024f. -
Liang et al. (2020)
Yaobo Liang, Nan Duan, Yeyun Gong, Ning Wu, Fenfei Guo, Weizhen Qi, Ming Gong, Linjun Shou, Daxin Jiang, Guihong Cao, Xiaodong Fan, Ruofei Zhang, Rahul Agrawal, Edward Cui, Sining Wei, Taroon Bharti, Ying Qiao, Jiun-Hung Chen, Winnie Wu, Shuguang Liu, Fan Yang, Daniel Campos, Rangan Majumder, and Ming Zhou.
XGLUE: A new benchmark dataset for cross-lingual pre-training, understanding and generation.
In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.),
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pp. 6008–6018. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.EMNLP-MAIN.484. URL https://doi.org/10.18653/v1/2020.emnlp-main.484. -
Liao et al. (2024)
Yusheng Liao, Shuyang Jiang, Yu Wang, and Yanfeng Wang.
MING-MOE: enhancing medical multi-task learning in large language models with sparse mixture of low-rank adapter experts.
*CoRR*, abs/2404.09027, 2024. doi: 10.48550/ARXIV.2404.09027. URL https://doi.org/10.48550/arXiv.2404.09027. -
Lieber et al. (2024)
Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, et al.
Jamba: A hybrid transformer-mamba language model.
*arXiv preprint arXiv:2403.19887*, 2024. -
Lifelo et al. (2024)
Zita Lifelo, Huansheng Ning, and Sahraoui Dhelim.
Adapting mental health prediction tasks for cross-lingual learning via meta-training and in-context learning with large language model.
*arXiv preprint arXiv:2404.09045*, 2024. -
Lin et al. (2021)
Bill Yuchen Lin, Seyeon Lee, Xiaoyang Qiao, and Xiang Ren.
Common sense beyond english: Evaluating and improving multilingual language models for commonsense reasoning.
In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.),
*Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021*, pp. 1274–1287. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.ACL-LONG.102. URL https://doi.org/10.18653/v1/2021.acl-long.102. -
Lin et al. (2024)
Peiqin Lin, Shaoxiong Ji, Jörg Tiedemann, André F. T. Martins, and Hinrich Schütze.
Mala-500: Massive language adaptation of large language models.
*CoRR*, abs/2401.13303, 2024. doi: 10.48550/ARXIV.2401.13303. URL https://doi.org/10.48550/arXiv.2401.13303. -
Lin et al. (2022)
Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona T. Diab, Veselin Stoyanov, and Xian Li.
Few-shot learning with multilingual generative language models.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 9019–9052. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.616. URL https://doi.org/10.18653/v1/2022.emnlp-main.616. -
Liu et al. (2023)
Chen Cecilia Liu, Fajri Koto, Timothy Baldwin, and Iryna Gurevych.
Are multilingual llms culturally-diverse reasoners? an investigation into multicultural proverbs and sayings.
*CoRR*, abs/2309.08591, 2023. doi: 10.48550/ARXIV.2309.08591. URL https://doi.org/10.48550/arXiv.2309.08591. -
Liu et al. (2021)
Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott.
Visually grounded reasoning across languages and cultures.
In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.),
*Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021*, pp. 10467–10485. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EMNLP-MAIN.818. URL https://doi.org/10.18653/v1/2021.emnlp-main.818. -
Liu et al. (2022)
Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel.
Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning.
In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.),
*Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html. -
Liu et al. (2024a)
Weize Liu, Yinlong Xu, Hongxia Xu, Jintai Chen, Xuming Hu, and Jian Wu.
Unraveling babel: Exploring multilingual activation patterns within large language models.
*CoRR*, abs/2402.16367, 2024a. doi: 10.48550/ARXIV.2402.16367. URL https://doi.org/10.48550/arXiv.2402.16367. -
Liu et al. (2024b)
Weize Liu, Yinlong Xu, Hongxia Xu, Jintai Chen, Xuming Hu, and Jian Wu.
Unraveling babel: Exploring multilingual activation patterns within large language models.
*arXiv preprint arXiv:2402.16367*, 2024b. -
Liu et al. (2020)
Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer.
Multilingual denoising pre-training for neural machine translation.
*Trans. Assoc. Comput. Linguistics*, 8:726–742, 2020. doi: 10.1162/TACL\_A\_00343. URL https://doi.org/10.1162/tacl_a_00343. -
Longpre et al. (2023)
Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al.
The flan collection: Designing data and methods for effective instruction tuning.
*arXiv preprint arXiv:2301.13688*, 2023. -
Lozhkov et al. (2024)
Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian J. McAuley, Han Hu, Torsten Scholak, Sébastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, and et al.
Starcoder 2 and the stack v2: The next generation.
*CoRR*, abs/2402.19173, 2024. doi: 10.48550/ARXIV.2402.19173. URL https://doi.org/10.48550/arXiv.2402.19173. -
Luo et al. (2023a)
Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang.
Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct.
*CoRR*, abs/2308.09583, 2023a. doi: 10.48550/ARXIV.2308.09583. URL https://doi.org/10.48550/arXiv.2308.09583. -
Luo et al. (2023b)
Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang.
Wizardcoder: Empowering code large language models with evol-instruct.
*CoRR*, abs/2306.08568, 2023b. doi: 10.48550/ARXIV.2306.08568. URL https://doi.org/10.48550/arXiv.2306.08568. -
Ma et al. (2022)
Weicheng Ma, Samiha Datta, Lili Wang, and Soroush Vosoughi.
Encbp: A new benchmark dataset for finer-grained cultural background prediction in english.
In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.),
*Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022*, pp. 2811–2823. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.FINDINGS-ACL.221. URL https://doi.org/10.18653/v1/2022.findings-acl.221. -
Malmasi et al. (2022)
Shervin Malmasi, Anjie Fang, Besnik Fetahu, Sudipta Kar, and Oleg Rokhlenko.
Semeval-2022 task 11: Multilingual complex named entity recognition (multiconer).
In Guy Emerson, Natalie Schluter, Gabriel Stanovsky, Ritesh Kumar, Alexis Palmer, Nathan Schneider, Siddharth Singh, and Shyam Ratan (eds.),
*Proceedings of the 16th International Workshop on Semantic Evaluation, SemEval@NAACL 2022, Seattle, Washington, United States, July 14-15, 2022*, pp. 1412–1437. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.SEMEVAL-1.196. URL https://doi.org/10.18653/v1/2022.semeval-1.196. -
Mao & Yu (2024)
Zhuoyuan Mao and Yen Yu.
Tuning llms with contrastive alignment instructions for machine translation in unseen, low-resource languages.
*CoRR*, abs/2401.05811, 2024. doi: 10.48550/ARXIV.2401.05811. URL https://doi.org/10.48550/arXiv.2401.05811. -
Masoud et al. (2023)
Reem I. Masoud, Ziquan Liu, Martin Ferianc, Philip C. Treleaven, and Miguel Rodrigues.
Cultural alignment in large language models: An explanatory analysis based on hofstede’s cultural dimensions.
*CoRR*, abs/2309.12342, 2023. doi: 10.48550/ARXIV.2309.12342. URL https://doi.org/10.48550/arXiv.2309.12342. -
Mehta et al. (2024)
Sachin Mehta, Mohammad Hossein Sekhavat, Qingqing Cao, Maxwell Horton, Yanzi Jin, Chenfan Sun, Iman Mirzadeh, Mahyar Najibi, Dmitry Belenko, Peter Zatloukal, et al.
Openelm: An efficient language model family with open-source training and inference framework.
*arXiv preprint arXiv:2404.14619*, 2024. -
Meng et al. (2024)
Yu Meng, Mengzhou Xia, and Danqi Chen.
Simpo: Simple preference optimization with a reference-free reward.
*CoRR*, abs/2405.14734, 2024. doi: 10.48550/ARXIV.2405.14734. URL https://doi.org/10.48550/arXiv.2405.14734. -
Mesnard et al. (2024)
Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Léonard Hussenot, Aakanksha Chowdhery, Adam Roberts, Aditya Barua, Alex Botev, Alex Castro-Ros, Ambrose Slone, Amélie Héliou, Andrea Tacchetti, Anna Bulanova, Antonia Paterson, Beth Tsai, Bobak Shahriari, Charline Le Lan, Christopher A. Choquette-Choo, Clément Crepy, Daniel Cer, Daphne Ippolito, David Reid, Elena Buchatskaya, Eric Ni, Eric Noland, Geng Yan, George Tucker, George-Cristian Muraru, Grigory Rozhdestvenskiy, Henryk Michalewski, Ian Tenney, Ivan Grishchenko, Jacob Austin, James Keeling, Jane Labanowski, Jean-Baptiste Lespiau, Jeff Stanway, Jenny Brennan, Jeremy Chen, Johan Ferret, Justin Chiu, and et al.
Gemma: Open models based on gemini research and technology.
*CoRR*, abs/2403.08295, 2024. doi: 10.48550/ARXIV.2403.08295. URL https://doi.org/10.48550/arXiv.2403.08295. -
Mikhailov et al. (2021)
Vladislav Mikhailov, Oleg Serikov, and Ekaterina Artemova.
Morph call: Probing morphosyntactic content of multilingual transformers.
*CoRR*, abs/2104.12847, 2021. URL https://arxiv.org/abs/2104.12847. -
Mitra et al. (2024)
Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah.
Orca-math: Unlocking the potential of slms in grade school math.
*CoRR*, abs/2402.14830, 2024. doi: 10.48550/ARXIV.2402.14830. URL https://doi.org/10.48550/arXiv.2402.14830. -
Moradshahi et al. (2023)
Mehrad Moradshahi, Tianhao Shen, Kalika Bali, Monojit Choudhury, Gaël de Chalendar, Anmol Goel, Sungkyun Kim, Prashant Kodali, Ponnurangam Kumaraguru, Nasredine Semmar, Sina J. Semnani, Jiwon Seo, Vivek Seshadri, Manish Shrivastava, Michael Sun, Aditya Yadavalli, Chaobin You, Deyi Xiong, and Monica S. Lam.
X-risawoz: High-quality end-to-end multilingual dialogue datasets and few-shot agents.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 2773–2794. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-ACL.174. URL https://doi.org/10.18653/v1/2023.findings-acl.174. -
Mu et al. (2024a)
Yongyu Mu, Peinan Feng, Zhiquan Cao, Yuzhang Wu, Bei Li, Chenglong Wang, Tong Xiao, Kai Song, Tongran Liu, Chunliang Zhang, et al.
Large language models are parallel multilingual learners.
*arXiv preprint arXiv:2403.09073*, 2024a. -
Mu et al. (2024b)
Yongyu Mu, Peinan Feng, Zhiquan Cao, Yuzhang Wu, Bei Li, Chenglong Wang, Tong Xiao, Kai Song, Tongran Liu, Chunliang Zhang, et al.
Revealing the parallel multilingual learning within large language models.
In
*Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pp. 6976–6997, 2024b. -
Muennighoff et al. (2023a)
Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M. Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel.
Crosslingual generalization through multitask finetuning.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 15991–16111. Association for Computational Linguistics, 2023a. doi: 10.18653/V1/2023.ACL-LONG.891. URL https://doi.org/10.18653/v1/2023.acl-long.891. -
Muennighoff et al. (2023b)
Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M. Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel.
Crosslingual generalization through multitask finetuning.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 15991–16111. Association for Computational Linguistics, 2023b. doi: 10.18653/V1/2023.ACL-LONG.891. URL https://doi.org/10.18653/v1/2023.acl-long.891. -
Mukherjee et al. (2024)
Anjishnu Mukherjee, Aylin Caliskan, Ziwei Zhu, and Antonios Anastasopoulos.
Global gallery: The fine art of painting culture portraits through multilingual instruction tuning.
*CoRR*, 2024. -
Mukherjee et al. (2023)
Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah.
Orca: Progressive learning from complex explanation traces of GPT-4.
*CoRR*, abs/2306.02707, 2023. doi: 10.48550/ARXIV.2306.02707. URL https://doi.org/10.48550/arXiv.2306.02707. -
Nakano et al. (2021)
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman.
Webgpt: Browser-assisted question-answering with human feedback.
*CoRR*, abs/2112.09332, 2021. URL https://arxiv.org/abs/2112.09332. -
Naous et al. (2023)
Tarek Naous, Michael J. Ryan, and Wei Xu.
Having beer after prayer? measuring cultural bias in large language models.
*CoRR*, abs/2305.14456, 2023. doi: 10.48550/ARXIV.2305.14456. URL https://doi.org/10.48550/arXiv.2305.14456. -
Narayan et al. (2018)
Shashi Narayan, Shay B. Cohen, and Mirella Lapata.
Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization.
In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.),
*Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018*, pp. 1797–1807. Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1206. URL https://doi.org/10.18653/v1/d18-1206. -
Nezhad & Agrawal (2024)
Sina Bagheri Nezhad and Ameeta Agrawal.
What drives performance in multilingual language models?
*arXiv preprint arXiv:2404.19159*, 2024. -
Nguyen (2023)
Ha-Thanh Nguyen.
A brief report on lawgpt 1.0: A virtual legal assistant based on GPT-3.
*CoRR*, abs/2302.05729, 2023. doi: 10.48550/ARXIV.2302.05729. URL https://doi.org/10.48550/arXiv.2302.05729. -
Nguyen et al. (2024)
Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, Hieu Man, Nghia Trung Ngo, Franck Dernoncourt, Ryan A. Rossi, and Thien Huu Nguyen.
Culturax: A cleaned, enormous, and multilingual dataset for large language models in 167 languages.
In Nicoletta Calzolari, Min-Yen Kan, Véronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (eds.),
*Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy*, pp. 4226–4237. ELRA and ICCL, 2024. URL https://aclanthology.org/2024.lrec-main.377. -
Nguyen et al. (2023a)
Xuan-Phi Nguyen, Wenxuan Zhang, Xin Li, Mahani Aljunied, Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang, Chaoqun Liu, Hang Zhang, and Lidong Bing.
Seallms - large language models for southeast asia.
*CoRR*, abs/2312.00738, 2023a. doi: 10.48550/ARXIV.2312.00738. URL https://doi.org/10.48550/arXiv.2312.00738. -
Nguyen et al. (2023b)
Xuan-Phi Nguyen, Wenxuan Zhang, Xin Li, Mahani Aljunied, Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang, Chaoqun Liu, Hang Zhang, and Lidong Bing.
Seallms - large language models for southeast asia.
*CoRR*, abs/2312.00738, 2023b. doi: 10.48550/ARXIV.2312.00738. URL https://doi.org/10.48550/arXiv.2312.00738. -
Nijkamp et al. (2023)
Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong.
Codegen: An open large language model for code with multi-turn program synthesis.
In
*The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. URL https://openreview.net/pdf?id=iaYcJKpY2B_. - Nivre et al. (2018) Joakim Nivre, Mitchell Abrams, Željko Agic, Lars Ahrenberg, Lene Antonsen, Maria Jesus Aranzabe, Gashaw Arutie, Masayuki Asahara, Luma Ateyah, and Mohammed Attia et al. Universal dependencies 2.2. 2018.
-
Ogundepo et al. (2023)
Odunayo Ogundepo, Tajuddeen R. Gwadabe, Clara E. Rivera, Jonathan H. Clark, Sebastian Ruder, David Ifeoluwa Adelani, Bonaventure F. P. Dossou, Abdou Aziz Diop, Claytone Sikasote, Gilles Hacheme, Happy Buzaaba, Ignatius Ezeani, Rooweither Mabuya, Salomey Osei, Chris Emezue, Albert Njoroge Kahira, Shamsuddeen Hassan Muhammad, Akintunde Oladipo, Abraham Toluwase Owodunni, Atnafu Lambebo Tonja, Iyanuoluwa Shode, Akari Asai, Tunde Oluwaseyi Ajayi, Clemencia Siro, Steven Arthur, Mofetoluwa Adeyemi, Orevaoghene Ahia, Aremu Anuoluwapo, Oyinkansola Awosan, Chiamaka Chukwuneke, Bernard Opoku, Awokoya Ayodele, Verrah Otiende, Christine Mwase, Boyd Sinkala, Andre Niyongabo Rubungo, Daniel A. Ajisafe, Emeka Felix Onwuegbuzia, Habib Mbow, Emile Niyomutabazi, Eunice Mukonde, Falalu Ibrahim Lawan, Ibrahim Said Ahmad, Jesujoba O. Alabi, Martin Namukombo, Chinedu Emmanuel Mbonu, Mofya Phiri, Neo Putini, Ndumiso Mngoma, Priscilla A. Amuok, Ruqayya Nasir Iro, and Sonia Adhiambo.
Afriqa: Cross-lingual open-retrieval question answering for african languages.
*CoRR*, abs/2305.06897, 2023. doi: 10.48550/ARXIV.2305.06897. URL https://doi.org/10.48550/arXiv.2305.06897. - Ortiz Su’arez et al. (2019) Pedro Javier Ortiz Su’arez, Benoit Sagot, and Laurent Romary. Asynchronous pipelines for processing huge corpora on medium to low resource infrastructures. Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-7) 2019. Cardiff, 22nd July 2019, pp. 9 – 16, Mannheim, 2019. Leibniz-Institut f"ur Deutsche Sprache. doi: 10.14618/ids-pub-9021. URL http://nbn-resolving.de/urn:nbn:de:bsz:mh39-90215.
-
Ortiz Su’arez et al. (2020)
Pedro Javier Ortiz Su’arez, Laurent Romary, and Benoit Sagot.
A monolingual approach to contextualized word embeddings for mid-resource languages.
In
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 1703–1714, Online, July 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.acl-main.156. -
Ouyang et al. (2022a)
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
Training language models to follow instructions with human feedback.
*Advances in neural information processing systems*, 35:27730–27744, 2022a. -
Ouyang et al. (2022b)
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe.
Training language models to follow instructions with human feedback.
In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.),
*Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022*, 2022b. URL http://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html. -
Pagnoni et al. (2021)
Artidoro Pagnoni, Vidhisha Balachandran, and Yulia Tsvetkov.
Understanding factuality in abstractive summarization with FRANK: A benchmark for factuality metrics.
In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.),
*Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021*, pp. 4812–4829. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.NAACL-MAIN.383. URL https://doi.org/10.18653/v1/2021.naacl-main.383. -
Palta & Rudinger (2023)
Shramay Palta and Rachel Rudinger.
FORK: A bite-sized test set for probing culinary cultural biases in commonsense reasoning models.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 9952–9962. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-ACL.631. URL https://doi.org/10.18653/v1/2023.findings-acl.631. -
Pan et al. (2017)
Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji.
Cross-lingual name tagging and linking for 282 languages.
In Regina Barzilay and Min-Yen Kan (eds.),
*Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers*, pp. 1946–1958. Association for Computational Linguistics, 2017. doi: 10.18653/V1/P17-1178. URL https://doi.org/10.18653/v1/P17-1178. -
Park et al. (2024)
Dojun Park, Jiwoo Lee, Seohyun Park, Hyeyun Jeong, Youngeun Koo, Soonha Hwang, Seonwoo Park, and Sungeun Lee.
Multiprageval: Multilingual pragmatic evaluation of large language models.
*arXiv preprint arXiv:2406.07736*, 2024. -
Peng et al. (2023)
Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al.
Rwkv: Reinventing rnns for the transformer era.
*arXiv preprint arXiv:2305.13048*, 2023. -
Peng & Søgaard (2024)
Qiwei Peng and Anders Søgaard.
Concept space alignment in multilingual llms.
*arXiv preprint arXiv:2410.01079*, 2024. -
Petrov et al. (2023)
Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi.
Language model tokenizers introduce unfairness between languages.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.),
*Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html. -
Pfeiffer et al. (2020)
Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, and Sebastian Ruder.
MAD-X: an adapter-based framework for multi-task cross-lingual transfer.
In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.),
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pp. 7654–7673. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.EMNLP-MAIN.617. URL https://doi.org/10.18653/v1/2020.emnlp-main.617. -
Philippy et al. (2023)
Fred Philippy, Siwen Guo, and Shohreh Haddadan.
Towards a common understanding of contributing factors for cross-lingual transfer in multilingual language models: A review.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 5877–5891. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.ACL-LONG.323. URL https://doi.org/10.18653/v1/2023.acl-long.323. -
Pistilli et al. (2024)
Giada Pistilli, Alina Leidinger, Yacine Jernite, Atoosa Kasirzadeh, Alexandra Sasha Luccioni, and Margaret Mitchell.
Civics: Building a dataset for examining culturally-informed values in large language models.
*CoRR*, abs/2405.13974, 2024. doi: 10.48550/ARXIV.2405.13974. URL https://doi.org/10.48550/arXiv.2405.13974. -
Ponti et al. (2020)
Edoardo Maria Ponti, Goran Glavas, Olga Majewska, Qianchu Liu, Ivan Vulic, and Anna Korhonen.
XCOPA: A multilingual dataset for causal commonsense reasoning.
In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.),
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020*, pp. 2362–2376. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.EMNLP-MAIN.185. URL https://doi.org/10.18653/v1/2020.emnlp-main.185. -
Press et al. (2021)
Ofir Press, Noah A Smith, and Mike Lewis.
Train short, test long: Attention with linear biases enables input length extrapolation.
*arXiv preprint arXiv:2108.12409*, 2021. -
Qi et al. (2023)
Jirui Qi, Raquel Fernández, and Arianna Bisazza.
Cross-lingual consistency of factual knowledge in multilingual language models.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pp. 10650–10666. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EMNLP-MAIN.658. URL https://doi.org/10.18653/v1/2023.emnlp-main.658. -
Qin et al. (2024)
Libo Qin, Qiguang Chen, Yuhang Zhou, Zhi Chen, Yinghui Li, Lizi Liao, Min Li, Wanxiang Che, and Philip S Yu.
Multilingual large language model: A survey of resources, taxonomy and frontiers.
*arXiv preprint arXiv:2404.04925*, 2024. -
Qiu et al. (2024)
Pengcheng Qiu, Chaoyi Wu, Xiaoman Zhang, Weixiong Lin, Haicheng Wang, Ya Zhang, Yanfeng Wang, and Weidi Xie.
Towards building multilingual language model for medicine.
*Nature Communications*, 15(1):8384, 2024. -
Rae et al. (2021)
Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, H. Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy,
Chris Jones, James Bradbury, Matthew J. Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving.
Scaling language models: Methods, analysis & insights from training gopher.
*CoRR*, abs/2112.11446, 2021. URL https://arxiv.org/abs/2112.11446. -
Rafailov et al. (2023)
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn.
Direct preference optimization: Your language model is secretly a reward model.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.),
*Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html. -
Raffel et al. (2020a)
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.
Exploring the limits of transfer learning with a unified text-to-text transformer.
*J. Mach. Learn. Res.*, 21:140:1–140:67, 2020a. URL http://jmlr.org/papers/v21/20-074.html. -
Raffel et al. (2020b)
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.
Exploring the limits of transfer learning with a unified text-to-text transformer.
*Journal of machine learning research*, 21(140):1–67, 2020b. -
Rajaee & Pilehvar (2022)
Sara Rajaee and Mohammad Taher Pilehvar.
An isotropy analysis in the multilingual BERT embedding space.
In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.),
*Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022*, pp. 1309–1316. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.FINDINGS-ACL.103. URL https://doi.org/10.18653/v1/2022.findings-acl.103. -
Ramezani & Xu (2023)
Aida Ramezani and Yang Xu.
Knowledge of cultural moral norms in large language models.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 428–446. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.ACL-LONG.26. URL https://doi.org/10.18653/v1/2023.acl-long.26. -
Ranaldi et al. (2023)
Leonardo Ranaldi, Giulia Pucci, and André Freitas.
Empowering cross-lingual abilities of instruction-tuned large language models by translation-following demonstrations.
*CoRR*, abs/2308.14186, 2023. doi: 10.48550/ARXIV.2308.14186. URL https://doi.org/10.48550/arXiv.2308.14186. -
Rao et al. (2023)
Abhinav Rao, Aditi Khandelwal, Kumar Tanmay, Utkarsh Agarwal, and Monojit Choudhury.
Ethical reasoning over moral alignment: A case and framework for in-context ethical policies in llms.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 13370–13388. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-EMNLP.892. URL https://doi.org/10.18653/v1/2023.findings-emnlp.892. -
Rao et al. (2024)
Abhinav Rao, Akhila Yerukola, Vishwa Shah, Katharina Reinecke, and Maarten Sap.
NORMAD: A benchmark for measuring the cultural adaptability of large language models.
*CoRR*, abs/2404.12464, 2024. doi: 10.48550/ARXIV.2404.12464. URL https://doi.org/10.48550/arXiv.2404.12464. -
Razumovskaia et al. (2024)
Evgeniia Razumovskaia, Ivan Vulić, and Anna Korhonen.
Analyzing and adapting large language models for few-shot multilingual nlu: Are we there yet?
*arXiv preprint arXiv:2403.01929*, 2024. -
Riemer et al. (2019)
Matthew Riemer, Ignacio Cases, Robert Ajemian, Miao Liu, Irina Rish, Yuhai Tu, and Gerald Tesauro.
Learning to learn without forgetting by maximizing transfer and minimizing interference.
In
*7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net, 2019. URL https://openreview.net/forum?id=B1gTShAct7. -
Rozière et al. (2023)
Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve.
Code llama: Open foundation models for code.
*CoRR*, abs/2308.12950, 2023. doi: 10.48550/ARXIV.2308.12950. URL https://doi.org/10.48550/arXiv.2308.12950. -
Rust et al. (2021)
Phillip Rust, Jonas Pfeiffer, Ivan Vulic, Sebastian Ruder, and Iryna Gurevych.
How good is your tokenizer? on the monolingual performance of multilingual language models.
In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.),
*Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021*, pp. 3118–3135. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.ACL-LONG.243. URL https://doi.org/10.18653/v1/2021.acl-long.243. -
Sanh et al. (2022a)
Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Févry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush.
Multitask prompted training enables zero-shot task generalization.
In
*The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net, 2022a. URL https://openreview.net/forum?id=9Vrb9D0WI4. -
Sanh et al. (2022b)
Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Févry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush.
Multitask prompted training enables zero-shot task generalization.
In
*The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net, 2022b. URL https://openreview.net/forum?id=9Vrb9D0WI4. -
Sarfraz et al. (2023)
Fahad Sarfraz, Elahe Arani, and Bahram Zonooz.
Error sensitivity modulation based experience replay: Mitigating abrupt representation drift in continual learning.
In
*The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. URL https://openreview.net/pdf?id=zlbci7019Z3. -
Scao et al. (2022)
Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, and et al.
BLOOM: A 176b-parameter open-access multilingual language model.
*CoRR*, abs/2211.05100, 2022. doi: 10.48550/ARXIV.2211.05100. URL https://doi.org/10.48550/arXiv.2211.05100. -
Scarlatos & Lan (2023)
Alexander Scarlatos and Andrew Lan.
Tree-based representation and generation of natural and mathematical language.
In
*Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 3714–3730, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.acl-long.205. -
Schulman et al. (2017)
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
Proximal policy optimization algorithms.
*CoRR*, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707.06347. -
Seganti et al. (2021)
Alessandro Seganti, Klaudia Firlag, Helena Skowronska, Michal Satlawa, and Piotr Andruszkiewicz.
Multilingual entity and relation extraction dataset and model.
In Paola Merlo, Jörg Tiedemann, and Reut Tsarfaty (eds.),
*Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021*, pp. 1946–1955. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EACL-MAIN.166. URL https://doi.org/10.18653/v1/2021.eacl-main.166. -
Shaham et al. (2024a)
Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, and Matan Eyal.
Multilingual instruction tuning with just a pinch of multilinguality.
*arXiv preprint arXiv:2401.01854*, 2024a. -
Shaham et al. (2024b)
Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, and Matan Eyal.
Multilingual instruction tuning with just a pinch of multilinguality.
*CoRR*, abs/2401.01854, 2024b. doi: 10.48550/ARXIV.2401.01854. URL https://doi.org/10.48550/arXiv.2401.01854. -
Shao et al. (2024)
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo.
Deepseekmath: Pushing the limits of mathematical reasoning in open language models.
*CoRR*, abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi.org/10.48550/arXiv.2402.03300. -
Shazeer et al. (2017)
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean.
Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.
*arXiv preprint arXiv:1701.06538*, 2017. -
She et al. (2024)
Shuaijie She, Shujian Huang, Wei Zou, Wenhao Zhu, Xiang Liu, Xiang Geng, and Jiajun Chen.
MAPO: advancing multilingual reasoning through multilingual alignment-as-preference optimization.
*CoRR*, abs/2401.06838, 2024. doi: 10.48550/ARXIV.2401.06838. URL https://doi.org/10.48550/arXiv.2401.06838. -
Shen et al. (2024a)
Lingfeng Shen, Weiting Tan, Sihao Chen, Yunmo Chen, Jingyu Zhang, Haoran Xu, Boyuan Zheng, Philipp Koehn, and Daniel Khashabi.
The language barrier: Dissecting safety challenges of llms in multilingual contexts.
*arXiv preprint arXiv:2401.13136*, 2024a. -
Shen et al. (2024b)
Lingfeng Shen, Weiting Tan, Sihao Chen, Yunmo Chen, Jingyu Zhang, Haoran Xu, Boyuan Zheng, Philipp Koehn, and Daniel Khashabi.
The language barrier: Dissecting safety challenges of llms in multilingual contexts.
*CoRR*, abs/2401.13136, 2024b. doi: 10.48550/ARXIV.2401.13136. URL https://doi.org/10.48550/arXiv.2401.13136. -
Shen et al. (2024c)
Siqi Shen, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, Soujanya Poria, and Rada Mihalcea.
Understanding the capabilities and limitations of large language models for cultural commonsense.
*CoRR*, abs/2405.04655, 2024c. doi: 10.48550/ARXIV.2405.04655. URL https://doi.org/10.48550/arXiv.2405.04655. -
Shen et al. (2023)
Zhiqiang Shen, Tianhua Tao, Liqun Ma, Willie Neiswanger, Zhengzhong Liu, Hongyi Wang, Bowen Tan, Joel Hestness, Natalia Vassilieva, Daria Soboleva, et al.
Slimpajama-dc: Understanding data combinations for llm training.
*arXiv preprint arXiv:2309.10818*, 2023. -
Shi et al. (2024a)
Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, and Hao Wang.
Continual learning of large language models: A comprehensive survey.
*CoRR*, abs/2404.16789, 2024a. doi: 10.48550/ARXIV.2404.16789. URL https://doi.org/10.48550/arXiv.2404.16789. -
Shi et al. (2024b)
Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, and Hao Wang.
Continual learning of large language models: A comprehensive survey.
*CoRR*, abs/2404.16789, 2024b. doi: 10.48550/ARXIV.2404.16789. URL https://doi.org/10.48550/arXiv.2404.16789. -
Shi et al. (2024c)
Weiyan Shi, Ryan Li, Yutong Zhang, Caleb Ziems, Chunhua yu, Raya Horesh, Rogério Abreu de Paula, and Diyi Yang.
Culturebank: An online community-driven knowledge base towards culturally aware language technologies.
*CoRR*, abs/2404.15238, 2024c. doi: 10.48550/ARXIV.2404.15238. URL https://doi.org/10.48550/arXiv.2404.15238. - Shi et al. (2024d) Zhengyan Shi, Adam X. Yang, Bin Wu, Laurence Aitchison, Emine Yilmaz, and Aldo Lipani. Instruction tuning with loss over instructions, 2024d.
-
Shwartz (2022)
Vered Shwartz.
Good night at 4 pm?! time expressions in different cultures.
In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.),
*Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022*, pp. 2842–2853. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.FINDINGS-ACL.224. URL https://doi.org/10.18653/v1/2022.findings-acl.224. -
Si et al. (2024)
Nianwen Si, Hao Zhang, and Weiqiang Zhang.
Mpn: Leveraging multilingual patch neuron for cross-lingual model editing.
*arXiv preprint arXiv:2401.03190*, 2024. -
Singh et al. (2024)
Shivalika Singh, Freddie Vargus, Daniel D’souza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura O’Mahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Souza Moura, Dominik Krzeminski, Hakimeh Fadaei, Irem Ergün, Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Minh Chien Vu, Sebastian Ruder, Surya Guthikonda, Emad A. Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, Ahmet Üstün, Marzieh Fadaee, and Sara Hooker.
Aya dataset: An open-access collection for multilingual instruction tuning.
*CoRR*, abs/2402.06619, 2024. doi: 10.48550/ARXIV.2402.06619. URL https://doi.org/10.48550/arXiv.2402.06619. -
Singhal et al. (2023)
Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Le Hou, Kevin Clark, Stephen Pfohl, Heather Cole-Lewis, Darlene Neal, Mike Schaekermann, Amy Wang, Mohamed Amin, Sami Lachgar, Philip Andrew Mansfield, Sushant Prakash, Bradley Green, Ewa Dominowska, Blaise Agüera y Arcas, Nenad Tomasev, Yun Liu, Renee Wong, Christopher Semturs, S. Sara Mahdavi, Joelle K. Barral, Dale R. Webster, Gregory S. Corrado, Yossi Matias, Shekoofeh Azizi, Alan Karthikesalingam, and Vivek Natarajan.
Towards expert-level medical question answering with large language models.
*CoRR*, abs/2305.09617, 2023. doi: 10.48550/ARXIV.2305.09617. URL https://doi.org/10.48550/arXiv.2305.09617. -
Song et al. (2024)
Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, and Houfeng Wang.
Preference ranking optimization for human alignment.
In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (eds.),
*Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada*, pp. 18990–18998. AAAI Press, 2024. doi: 10.1609/AAAI.V38I17.29865. URL https://doi.org/10.1609/aaai.v38i17.29865. -
Sorensen et al. (2024)
Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell L. Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, Tim Althoff, and Yejin Choi.
A roadmap to pluralistic alignment.
*CoRR*, abs/2402.05070, 2024. doi: 10.48550/ARXIV.2402.05070. URL https://doi.org/10.48550/arXiv.2402.05070. -
Stanczak et al. (2022)
Karolina Stanczak, Edoardo M. Ponti, Lucas Torroba Hennigen, Ryan Cotterell, and Isabelle Augenstein.
Same neurons, different languages: Probing morphosyntax in multilingual pre-trained models.
In Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz (eds.),
*Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022*, pp. 1589–1598. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.NAACL-MAIN.114. URL https://doi.org/10.18653/v1/2022.naacl-main.114. -
Stanovsky et al. (2019)
Gabriel Stanovsky, Noah A. Smith, and Luke Zettlemoyer.
Evaluating gender bias in machine translation.
In Anna Korhonen, David R. Traum, and Lluís Màrquez (eds.),
*Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers*, pp. 1679–1684. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1164. URL https://doi.org/10.18653/v1/p19-1164. -
Starace et al. (2023)
Giulio Starace, Konstantinos Papakostas, Rochelle Choenni, Apostolos Panagiotopoulos, Matteo Rosati, Alina Leidinger, and Ekaterina Shutova.
Probing llms for joint encoding of linguistic categories.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 7158–7179. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-EMNLP.476. URL https://doi.org/10.18653/v1/2023.findings-emnlp.476. -
Stiennon et al. (2020)
Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano.
Learning to summarize with human feedback.
In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.),
*Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual*, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html. -
Su et al. (2024)
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu.
Roformer: Enhanced transformer with rotary position embedding.
*Neurocomputing*, 568:127063, 2024. -
Sun et al. (2023)
Hao Sun, Zhexin Zhang, Jiawen Deng, Jiale Cheng, and Minlie Huang.
Safety assessment of chinese large language models.
*CoRR*, abs/2304.10436, 2023. doi: 10.48550/ARXIV.2304.10436. URL https://doi.org/10.48550/arXiv.2304.10436. -
Sun et al. (2024)
Haoran Sun, Renren Jin, Shaoyang Xu, Leiyu Pan, Menglong Cui, Jiangcun Dui, Yikun Lei, Lei Yang, Ling Shi, Juesi Xiao, et al.
Fuxitranyu: A multilingual large language model trained with balanced data.
*arXiv preprint arXiv:2408.06273*, 2024. -
Sun et al. (2021)
Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, et al.
Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation.
*arXiv preprint arXiv:2107.02137*, 2021. - Survey (2022) World Values Survey. World values survey. 2022. URL https://www.worldvaluessurvey.org/wvs.jsp.
-
Talmor et al. (2019)
Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant.
Commonsenseqa: A question answering challenge targeting commonsense knowledge.
In Jill Burstein, Christy Doran, and Thamar Solorio (eds.),
*Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)*, pp. 4149–4158. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1421. URL https://doi.org/10.18653/v1/n19-1421. -
Tang et al. (2024)
Tianyi Tang, Wenyang Luo, Haoyang Huang, Dongdong Zhang, Xiaolei Wang, Xin Zhao, Furu Wei, and Ji-Rong Wen.
Language-specific neurons: The key to multilingual capabilities in large language models.
*CoRR*, abs/2402.16438, 2024. doi: 10.48550/ARXIV.2402.16438. URL https://doi.org/10.48550/arXiv.2402.16438. - Taori et al. (2023) Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. 2023. URL https://github.com/tatsu-lab/stanford_alpaca.
-
Tars et al. (2022)
Maali Tars, Andre Tättar, and Mark Fishel.
Cross-lingual transfer from large multilingual translation models to unseen under-resourced languages.
*Balt. J. Mod. Comput.*, 10(3), 2022. doi: 10.22364/BJMC.2022.10.3.16. URL https://doi.org/10.22364/bjmc.2022.10.3.16. -
Tay et al. (2022)
Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Garcia, Jason Wei, Xuezhi Wang, Hyung Won Chung, Siamak Shakeri, Dara Bahri, Tal Schuster, et al.
Ul2: Unifying language learning paradigms.
*arXiv preprint arXiv:2205.05131*, 2022. -
Team et al. (2024)
Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al.
Gemma: Open models based on gemini research and technology.
*arXiv preprint arXiv:2403.08295*, 2024. - Team (2023) InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities, 2023.
- Team (2024a) Qwen Team. Qwen1.5-moe: Matching 7b model performance with 1/3 activated parameters", February 2024a. URL https://qwenlm.github.io/blog/qwen-moe/.
- Team (2024b) The Mosaic Research Team. Dbrx: A new state-of-the-art open llm, March 2024b. URL https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm.
-
Thapliyal et al. (2022)
Ashish V. Thapliyal, Jordi Pont-Tuset, Xi Chen, and Radu Soricut.
Crossmodal-3600: A massively multilingual multimodal evaluation dataset.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 715–729. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.45. URL https://doi.org/10.18653/v1/2022.emnlp-main.45. -
Tiyajamorn et al. (2021)
Nattapong Tiyajamorn, Tomoyuki Kajiwara, Yuki Arase, and Makoto Onizuka.
Language-agnostic representation from multilingual sentence encoders for cross-lingual similarity estimation.
In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.),
*Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021*, pp. 7764–7774. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EMNLP-MAIN.612. URL https://doi.org/10.18653/v1/2021.emnlp-main.612. - Tokpanov et al. (2024) Yury Tokpanov, Beren Millidge, Paolo Glorioso, Jonathan Pilault, Adam Ibrahim, James Whittington, and Quentin Anthony. Zyda: A 1.3t dataset for open language modeling, 2024.
-
Touvron et al. (2023a)
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample.
Llama: Open and efficient foundation language models.
*CoRR*, abs/2302.13971, 2023a. doi: 10.48550/ARXIV.2302.13971. URL https://doi.org/10.48550/arXiv.2302.13971. -
Touvron et al. (2023b)
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.
Llama: Open and efficient foundation language models.
*arXiv preprint arXiv:2302.13971*, 2023b. -
Touvron et al. (2023c)
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov,
and Thomas Scialom.
Llama 2: Open foundation and fine-tuned chat models.
*CoRR*, abs/2307.09288, 2023c. doi: 10.48550/ARXIV.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288. -
Touvron et al. (2023d)
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
Llama 2: Open foundation and fine-tuned chat models.
*arXiv preprint arXiv:2307.09288*, 2023d. -
Upadhayay & Behzadan (2023)
Bibek Upadhayay and Vahid Behzadan.
Taco: Enhancing cross-lingual transfer for low-resource languages in llms through translation-assisted chain-of-thought processes.
*CoRR*, abs/2311.10797, 2023. doi: 10.48550/ARXIV.2311.10797. URL https://doi.org/10.48550/arXiv.2311.10797. -
Üstün et al. (2024)
Ahmet Üstün, Viraat Aryabumi, Zheng Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker.
Aya model: An instruction finetuned open-access multilingual language model.
*CoRR*, abs/2402.07827, 2024. doi: 10.48550/ARXIV.2402.07827. URL https://doi.org/10.48550/arXiv.2402.07827. -
Vashishtha et al. (2023)
Aniket Vashishtha, Kabir Ahuja, and Sunayana Sitaram.
On evaluating and mitigating gender biases in multilingual settings.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Findings of the Association for Computational Linguistics: ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 307–318. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS-ACL.21. URL https://doi.org/10.18653/v1/2023.findings-acl.21. -
Vaswani et al. (2017)
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.
Attention is all you need.
*Advances in neural information processing systems*, 30, 2017. -
Vilares et al. (2016)
David Vilares, Miguel A. Alonso, and Carlos Gómez-Rodríguez.
EN-ES-CS: an english-spanish code-switching twitter corpus for multilingual sentiment analysis.
In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Sara Goggi, Marko Grobelnik, Bente Maegaard, Joseph Mariani, Hélène Mazo, Asunción Moreno, Jan Odijk, and Stelios Piperidis (eds.),
*Proceedings of the Tenth International Conference on Language Resources and Evaluation LREC 2016, Portorož, Slovenia, May 23-28, 2016*. European Language Resources Association (ELRA), 2016. URL http://www.lrec-conf.org/proceedings/lrec2016/summaries/43.html. -
Vulic et al. (2023)
Ivan Vulic, Goran Glavas, Fangyu Liu, Nigel Collier, Edoardo Maria Ponti, and Anna Korhonen.
Probing cross-lingual lexical knowledge from multilingual sentence encoders.
In Andreas Vlachos and Isabelle Augenstein (eds.),
*Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2023, Dubrovnik, Croatia, May 2-6, 2023*, pp. 2081–2097. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.EACL-MAIN.153. URL https://doi.org/10.18653/v1/2023.eacl-main.153. -
Wang et al. (2024a)
Bin Wang, Geyu Lin, Zhengyuan Liu, Chengwei Wei, and Nancy F. Chen.
Craft: Extracting and tuning cultural instructions from the wild.
*CoRR*, abs/2405.03138, 2024a. doi: 10.48550/ARXIV.2405.03138. URL https://doi.org/10.48550/arXiv.2405.03138. -
Wang et al. (2023a)
Guangyu Wang, Guoxing Yang, Zongxin Du, Longjun Fan, and Xiaohu Li.
Clinicalgpt: Large language models finetuned with diverse medical data and comprehensive evaluation.
*CoRR*, abs/2306.09968, 2023a. doi: 10.48550/ARXIV.2306.09968. URL https://doi.org/10.48550/arXiv.2306.09968. -
Wang et al. (2024b)
Hetong Wang, Pasquale Minervini, and Edoardo M Ponti.
Probing the emergence of cross-lingual alignment during llm training.
*arXiv preprint arXiv:2406.13229*, 2024b. -
Wang et al. (2023b)
Rongsheng Wang, Yaofei Duan, Chan-Tong Lam, Jiexi Chen, Jiangsheng Xu, Haoming Chen, Xiaohong Liu, Patrick Cheong-Iao Pang, and Tao Tan.
Ivygpt: Interactive chinese pathway language model in medical domain.
In Lu Fang, Jian Pei, Guangtao Zhai, and Ruiping Wang (eds.),
*Artificial Intelligence - Third CAAI International Conference, CICAI 2023, Fuzhou, China, July 22-23, 2023, Revised Selected Papers, Part II*, volume 14474 of*Lecture Notes in Computer Science*, pp. 378–382. Springer, 2023b. doi: 10.1007/978-981-99-9119-8\_34. URL https://doi.org/10.1007/978-981-99-9119-8_34. -
Wang et al. (2023c)
Wenxuan Wang, Wenxiang Jiao, Jingyuan Huang, Ruyi Dai, Jen-tse Huang, Zhaopeng Tu, and Michael R. Lyu.
Not all countries celebrate thanksgiving: On the cultural dominance in large language models.
*CoRR*, abs/2310.12481, 2023c. doi: 10.48550/ARXIV.2310.12481. URL https://doi.org/10.48550/arXiv.2310.12481. -
Wang et al. (2023d)
Wenxuan Wang, Zhaopeng Tu, Chang Chen, Youliang Yuan, Jen-tse Huang, Wenxiang Jiao, and Michael R. Lyu.
All languages matter: On the multilingual safety of large language models.
*CoRR*, abs/2310.00905, 2023d. doi: 10.48550/ARXIV.2310.00905. URL https://doi.org/10.48550/arXiv.2310.00905. -
Wang et al. (2023e)
Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang.
Orthogonal subspace learning for language model continual learning.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 10658–10671. Association for Computational Linguistics, 2023e. doi: 10.18653/V1/2023.FINDINGS-EMNLP.715. URL https://doi.org/10.18653/v1/2023.findings-emnlp.715. -
Wang et al. (2022)
Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen.
Super-naturalinstructions: Generalization via declarative instructions on 1600+ NLP tasks.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 5085–5109. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.340. URL https://doi.org/10.18653/v1/2022.emnlp-main.340. -
Wang et al. (2023f)
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi.
Self-instruct: Aligning language models with self-generated instructions.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 13484–13508. Association for Computational Linguistics, 2023f. doi: 10.18653/V1/2023.ACL-LONG.754. URL https://doi.org/10.18653/v1/2023.acl-long.754. -
Wang et al. (2021)
Yue Wang, Weishi Wang, Shafiq R. Joty, and Steven C. H. Hoi.
Codet5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation.
In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.),
*Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021*, pp. 8696–8708. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EMNLP-MAIN.685. URL https://doi.org/10.18653/v1/2021.emnlp-main.685. -
Wang et al. (2023g)
Yue Wang, Hung Le, Akhilesh Gotmare, Nghi D. Q. Bui, Junnan Li, and Steven C. H. Hoi.
Codet5+: Open code large language models for code understanding and generation.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pp. 1069–1088. Association for Computational Linguistics, 2023g. doi: 10.18653/V1/2023.EMNLP-MAIN.68. URL https://doi.org/10.18653/v1/2023.emnlp-main.68. -
Weber et al. (2024)
Alexander Arno Weber, Klaudia Thellmann, Jan Ebert, Nicolas Flores-Herr, Jens Lehmann, Michael Fromm, and Mehdi Ali.
Investigating multilingual instruction-tuning: Do polyglot models demand for multilingual instructions?
*CoRR*, abs/2402.13703, 2024. doi: 10.48550/ARXIV.2402.13703. URL https://doi.org/10.48550/arXiv.2402.13703. -
Webster et al. (2020)
Kellie Webster, Xuezhi Wang, Ian Tenney, Alex Beutel, Emily Pitler, Ellie Pavlick, Jilin Chen, and Slav Petrov.
Measuring and reducing gendered correlations in pre-trained models.
*CoRR*, abs/2010.06032, 2020. URL https://arxiv.org/abs/2010.06032. -
(323)
Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le.
Finetuned language models are zero-shot learners.
In
*International Conference on Learning Representations*. -
Wei et al. (2022)
Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le.
Finetuned language models are zero-shot learners.
In
*The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022*. OpenReview.net, 2022. URL https://openreview.net/forum?id=gEZrGCozdqR. -
Wei et al. (2023)
Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, Tianxiang Hu, Shangjie Li, Binyuan Hui, Bowen Yu, Dayiheng Liu, Baosong Yang, Fei Huang, and Jun Xie.
Polylm: An open source polyglot large language model.
*CoRR*, abs/2307.06018, 2023. doi: 10.48550/ARXIV.2307.06018. URL https://doi.org/10.48550/arXiv.2307.06018. -
Wendler et al. (2024)
Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West.
Do llamas work in english? on the latent language of multilingual transformers.
*CoRR*, abs/2402.10588, 2024. doi: 10.48550/ARXIV.2402.10588. URL https://doi.org/10.48550/arXiv.2402.10588. -
Wenzek et al. (2020)
Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave.
Ccnet: Extracting high quality monolingual datasets from web crawl data.
In Nicoletta Calzolari, Frédéric Béchet, Philippe Blache, Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, Asunción Moreno, Jan Odijk, and Stelios Piperidis (eds.),
*Proceedings of The 12th Language Resources and Evaluation Conference, LREC 2020, Marseille, France, May 11-16, 2020*, pp. 4003–4012. European Language Resources Association, 2020. URL https://aclanthology.org/2020.lrec-1.494/. -
Wu et al. (2024a)
Chengyue Wu, Yukang Gan, Yixiao Ge, Zeyu Lu, Jiahao Wang, Ye Feng, Ping Luo, and Ying Shan.
Llama pro: Progressive llama with block expansion.
*CoRR*, abs/2401.02415, 2024a. doi: 10.48550/ARXIV.2401.02415. URL https://doi.org/10.48550/arXiv.2401.02415. - Wu et al. (2024b) Junkang Wu, Yuexiang Xie, Zhengyi Yang, Jiancan Wu, Jinyang Gao, Bolin Ding, Xiang Wang, and Xiangnan He. -dpo: Direct preference optimization with dynamic , 2024b. URL https://arxiv.org/abs/2407.08639.
-
Wu et al. (2024c)
Qiyu Wu, Masaaki Nagata, Zhongtao Miao, and Yoshimasa Tsuruoka.
Word alignment as preference for machine translation.
*CoRR*, abs/2405.09223, 2024c. doi: 10.48550/ARXIV.2405.09223. URL https://doi.org/10.48550/arXiv.2405.09223. -
Wu et al. (2024d)
Zhaofeng Wu, Ananth Balashankar, Yoon Kim, Jacob Eisenstein, and Ahmad Beirami.
Reuse your rewards: Reward model transfer for zero-shot cross-lingual alignment.
*CoRR*, abs/2404.12318, 2024d. doi: 10.48550/ARXIV.2404.12318. URL https://doi.org/10.48550/arXiv.2404.12318. -
Xiao et al. (2021)
Chaojun Xiao, Xueyu Hu, Zhiyuan Liu, Cunchao Tu, and Maosong Sun.
Lawformer: A pre-trained language model for chinese legal long documents.
*AI Open*, 2:79–84, 2021. doi: 10.1016/J.AIOPEN.2021.06.003. URL https://doi.org/10.1016/j.aiopen.2021.06.003. -
Xie et al. (2024)
Rui Xie, Zhengran Zeng, Zhuohao Yu, Chang Gao, Shikun Zhang, and Wei Ye.
Codeshell technical report.
*CoRR*, abs/2403.15747, 2024. doi: 10.48550/ARXIV.2403.15747. URL https://doi.org/10.48550/arXiv.2403.15747. -
Xie et al. (2022)
Zhihui Xie, Handong Zhao, Tong Yu, and Shuai Li.
Discovering low-rank subspaces for language-agnostic multilingual representations.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 5617–5633. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.379. URL https://doi.org/10.18653/v1/2022.emnlp-main.379. -
Xiong et al. (2023)
Honglin Xiong, Sheng Wang, Yitao Zhu, Zihao Zhao, Yuxiao Liu, Linlin Huang, Qian Wang, and Dinggang Shen.
Doctorglm: Fine-tuning your chinese doctor is not a herculean task.
*CoRR*, abs/2304.01097, 2023. doi: 10.48550/ARXIV.2304.01097. URL https://doi.org/10.48550/arXiv.2304.01097. - Xiong et al. (2024) Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint, 2024. URL https://arxiv.org/abs/2312.11456.
-
Xu et al. (2023a)
Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang.
Wizardlm: Empowering large language models to follow complex instructions.
*CoRR*, abs/2304.12244, 2023a. doi: 10.48550/ARXIV.2304.12244. URL https://doi.org/10.48550/arXiv.2304.12244. -
Xu et al. (2022)
Frank F. Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn.
A systematic evaluation of large language models of code.
In Swarat Chaudhuri and Charles Sutton (eds.),
*MAPS@PLDI 2022: 6th ACM SIGPLAN International Symposium on Machine Programming, San Diego, CA, USA, 13 June 2022*, pp. 1–10. ACM, 2022. doi: 10.1145/3520312.3534862. URL https://doi.org/10.1145/3520312.3534862. -
Xu et al. (2023b)
Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla.
A paradigm shift in machine translation: Boosting translation performance of large language models.
*CoRR*, abs/2309.11674, 2023b. doi: 10.48550/ARXIV.2309.11674. URL https://doi.org/10.48550/arXiv.2309.11674. -
Xu et al. (2024a)
Haoran Xu, Amr Sharaf, Yunmo Chen, Weiting Tan, Lingfeng Shen, Benjamin Van Durme, Kenton Murray, and Young Jin Kim.
Contrastive preference optimization: Pushing the boundaries of LLM performance in machine translation.
*CoRR*, abs/2401.08417, 2024a. doi: 10.48550/ARXIV.2401.08417. URL https://doi.org/10.48550/arXiv.2401.08417. -
Xu et al. (2023c)
Ningyu Xu, Qi Zhang, Jingting Ye, Menghan Zhang, and Xuanjing Huang.
Are structural concepts universal in transformer language models? towards interpretable cross-lingual generalization.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 13951–13976. Association for Computational Linguistics, 2023c. doi: 10.18653/V1/2023.FINDINGS-EMNLP.931. URL https://doi.org/10.18653/v1/2023.findings-emnlp.931. -
Xu et al. (2024b)
Nuo Xu, Jun Zhao, Can Zu, Sixian Li, Lu Chen, Zhihao Zhang, Rui Zheng, Shihan Dou, Wenjuan Qin, Tao Gui, Qi Zhang, and Xuanjing Huang.
Advancing translation preference modeling with RLHF: A step towards cost-effective solution.
*CoRR*, abs/2402.11525, 2024b. doi: 10.48550/ARXIV.2402.11525. URL https://doi.org/10.48550/arXiv.2402.11525. -
Xu et al. (2023d)
Shaoyang Xu, Junzhuo Li, and Deyi Xiong.
Language representation projection: Can we transfer factual knowledge across languages in multilingual language models?
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pp. 3692–3702. Association for Computational Linguistics, 2023d. doi: 10.18653/V1/2023.EMNLP-MAIN.226. URL https://doi.org/10.18653/v1/2023.emnlp-main.226. -
Xu et al. (2023e)
Shaoyang Xu, Junzhuo Li, and Deyi Xiong.
Language representation projection: Can we transfer factual knowledge across languages in multilingual language models?
*arXiv preprint arXiv:2311.03788*, 2023e. -
Xu et al. (2024c)
Shaoyang Xu, Yongqi Leng, Linhao Yu, and Deyi Xiong.
Self-pluralising culture alignment for large language models.
*CoRR*, abs/2410.12971, 2024c. URL https://arxiv.org/abs/2410.12971. -
Xu et al. (2024d)
Yifan Xu, Xiao Liu, Xinghan Liu, Zhenyu Hou, Yueyan Li, Xiaohan Zhang, Zihan Wang, Aohan Zeng, Zhengxiao Du, Wenyi Zhao, Jie Tang, and Yuxiao Dong.
Chatglm-math: Improving math problem-solving in large language models with a self-critique pipeline.
*CoRR*, abs/2404.02893, 2024d. doi: 10.48550/ARXIV.2404.02893. URL https://doi.org/10.48550/arXiv.2404.02893. -
Xu et al. (2024e)
Yuemei Xu, Ling Hu, Jiayi Zhao, Zihan Qiu, Yuqi Ye, and Hanwen Gu.
A survey on multilingual large language models: Corpora, alignment, and bias.
*arXiv preprint arXiv:2404.00929*, 2024e. -
Xu et al. (2023f)
Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, and Qi Tian.
Qa-lora: Quantization-aware low-rank adaptation of large language models.
*CoRR*, abs/2309.14717, 2023f. doi: 10.48550/ARXIV.2309.14717. URL https://doi.org/10.48550/arXiv.2309.14717. -
Xue et al. (2024)
Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You.
Openmoe: An early effort on open mixture-of-experts language models.
*arXiv preprint arXiv:2402.01739*, 2024. -
Xue et al. (2020)
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel.
mt5: A massively multilingual pre-trained text-to-text transformer.
*arXiv preprint arXiv:2010.11934*, 2020. -
Xue et al. (2021)
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel.
mt5: A massively multilingual pre-trained text-to-text transformer.
In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.),
*Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021*, pp. 483–498. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.NAACL-MAIN.41. URL https://doi.org/10.18653/v1/2021.naacl-main.41. -
Xue et al. (2022)
Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel.
ByT5: Towards a token-free future with pre-trained byte-to-byte models.
*Transactions of the Association for Computational Linguistics*, 10:291–306, 2022. doi: 10.1162/tacl_a_00461. URL https://aclanthology.org/2022.tacl-1.17. -
Yang et al. (2023a)
Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, Hang Xu, Haoze Sun, Hongda Zhang, Hui Liu, Jiaming Ji, Jian Xie, Juntao Dai, Kun Fang, Lei Su, Liang Song, Lifeng Liu, Liyun Ru, Luyao Ma, Mang Wang, Mickel Liu, MingAn Lin, Nuolan Nie, Peidong Guo, Ruiyang Sun, Tao Zhang, Tianpeng Li, Tianyu Li, Wei Cheng, Weipeng Chen, Xiangrong Zeng, Xiaochuan Wang, Xiaoxi Chen, Xin Men, Xin Yu, Xuehai Pan, Yanjun Shen, Yiding Wang, Yiyu Li, Youxin Jiang, Yuchen Gao, Yupeng Zhang, Zenan Zhou, and Zhiying Wu.
Baichuan 2: Open large-scale language models.
*CoRR*, abs/2309.10305, 2023a. doi: 10.48550/ARXIV.2309.10305. URL https://doi.org/10.48550/arXiv.2309.10305. -
Yang et al. (2023b)
Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al.
Baichuan 2: Open large-scale language models.
*arXiv preprint arXiv:2309.10305*, 2023b. -
Yang et al. (2024a)
Jian Yang, Hongcheng Guo, Yuwei Yin, Jiaqi Bai, Bing Wang, Jiaheng Liu, Xinnian Liang, Linzheng Chai, Liqun Yang, and Zhoujun Li.
m3p: Towards multimodal multilingual translation with multimodal prompt.
In Nicoletta Calzolari, Min-Yen Kan, Véronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (eds.),
*Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy*, pp. 10858–10871. ELRA and ICCL, 2024a. URL https://aclanthology.org/2024.lrec-main.948. -
Yang et al. (2024b)
Shu Yang, Muhammad Asif Ali, Cheng-Long Wang, Lijie Hu, and Di Wang.
Moral: Moe augmented lora for llms’ lifelong learning.
*CoRR*, abs/2402.11260, 2024b. doi: 10.48550/ARXIV.2402.11260. URL https://doi.org/10.48550/arXiv.2402.11260. -
Yang et al. (2023c)
Wen Yang, Chong Li, Jiajun Zhang, and Chengqing Zong.
Bigtrans: Augmenting large language models with multilingual translation capability over 100 languages.
*CoRR*, abs/2305.18098, 2023c. doi: 10.48550/ARXIV.2305.18098. URL https://doi.org/10.48550/arXiv.2305.18098. -
Yang et al. (2019)
Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge.
PAWS-X: A cross-lingual adversarial dataset for paraphrase identification.
In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.),
*Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019*, pp. 3685–3690. Association for Computational Linguistics, 2019. doi: 10.18653/V1/D19-1382. URL https://doi.org/10.18653/v1/D19-1382. -
Yao et al. (2024)
Binwei Yao, Ming Jiang, Diyi Yang, and Junjie Hu.
Benchmarking llm-based machine translation on cultural awareness.
*CoRR*, abs/2305.14328, 2024. doi: 10.48550/ARXIV.2305.14328. URL https://doi.org/10.48550/arXiv.2305.14328. -
Yin et al. (2022)
Da Yin, Hritik Bansal, Masoud Monajatipoor, Liunian Harold Li, and Kai-Wei Chang.
Geomlama: Geo-diverse commonsense probing on multilingual pre-trained language models.
In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.),
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, pp. 2039–2055. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.EMNLP-MAIN.132. URL https://doi.org/10.18653/v1/2022.emnlp-main.132. -
Yong et al. (2023a)
Zheng Xin Yong, Cristina Menghini, and Stephen H. Bach.
Low-resource languages jailbreak GPT-4.
*CoRR*, abs/2310.02446, 2023a. doi: 10.48550/ARXIV.2310.02446. URL https://doi.org/10.48550/arXiv.2310.02446. -
Yong et al. (2023b)
Zheng Xin Yong, Hailey Schoelkopf, Niklas Muennighoff, Alham Fikri Aji, David Ifeoluwa Adelani, Khalid Almubarak, M. Saiful Bari, Lintang Sutawika, Jungo Kasai, Ahmed Baruwa, Genta Indra Winata, Stella Biderman, Edward Raff, Dragomir Radev, and Vassilina Nikoulina.
BLOOM+1: adding language support to BLOOM for zero-shot prompting.
In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (eds.),
*Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, pp. 11682–11703. Association for Computational Linguistics, 2023b. doi: 10.18653/V1/2023.ACL-LONG.653. URL https://doi.org/10.18653/v1/2023.acl-long.653. -
Yoon et al. (2024)
Dongkeun Yoon, Joel Jang, Sungdong Kim, Seungone Kim, Sheikh Shafayat, and Minjoon Seo.
Langbridge: Multilingual reasoning without multilingual supervision.
*CoRR*, abs/2401.10695, 2024. doi: 10.48550/ARXIV.2401.10695. URL https://doi.org/10.48550/arXiv.2401.10695. -
Young et al. (2024)
Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al.
Yi: Open foundation models by 01. ai.
*arXiv preprint arXiv:2403.04652*, 2024. -
Yu et al. (2023)
Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu.
Metamath: Bootstrap your own mathematical questions for large language models.
*CoRR*, abs/2309.12284, 2023. doi: 10.48550/ARXIV.2309.12284. URL https://doi.org/10.48550/arXiv.2309.12284. -
Yuan et al. (2023a)
Fei Yuan, Shuai Yuan, Zhiyong Wu, and Lei Li.
How multilingual is multilingual llm?
*CoRR*, abs/2311.09071, 2023a. doi: 10.48550/ARXIV.2311.09071. URL https://doi.org/10.48550/arXiv.2311.09071. -
Yuan et al. (2023b)
Zheng Yuan, Hongyi Yuan, Chuanqi Tan, Wei Wang, Songfang Huang, and Fei Huang.
RRHF: rank responses to align language models with human feedback without tears.
*CoRR*, abs/2304.05302, 2023b. doi: 10.48550/ARXIV.2304.05302. URL https://doi.org/10.48550/arXiv.2304.05302. -
Yue et al. (2023a)
Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, and Zhongyu Wei.
Disc-lawllm: Fine-tuning large language models for intelligent legal services.
*CoRR*, abs/2309.11325, 2023a. doi: 10.48550/ARXIV.2309.11325. URL https://doi.org/10.48550/arXiv.2309.11325. -
Yue et al. (2023b)
Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen.
Mammoth: Building math generalist models through hybrid instruction tuning.
*CoRR*, abs/2309.05653, 2023b. doi: 10.48550/ARXIV.2309.05653. URL https://doi.org/10.48550/arXiv.2309.05653. -
Zeng et al. (2024)
Jiali Zeng, Fandong Meng, Yongjing Yin, and Jie Zhou.
Teaching large language models to translate with comparison.
In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (eds.),
*Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada*, pp. 19488–19496. AAAI Press, 2024. doi: 10.1609/AAAI.V38I17.29920. URL https://doi.org/10.1609/aaai.v38i17.29920. -
Zeng et al. (2021)
Wei Zeng, Xiaozhe Ren, Teng Su, Hui Wang, Yi Liao, Zhiwei Wang, Xin Jiang, ZhenZhang Yang, Kaisheng Wang, Xiaoda Zhang, Chen Li, Ziyan Gong, Yifan Yao, Xinjing Huang, Jun Wang, Jianfeng Yu, Qi Guo, Yue Yu, Yan Zhang, Jin Wang, Hengtao Tao, Dasen Yan, Zexuan Yi, Fang Peng, Fangqing Jiang, Han Zhang, Lingfeng Deng, Yehong Zhang, Zhe Lin, Chao Zhang, Shaojie Zhang, Mingyue Guo, Shanzhi Gu, Gaojun Fan, Yaowei Wang, Xuefeng Jin, Qun Liu, and Yonghong Tian.
Pangu-: Large-scale autoregressive pretrained chinese language models with auto-parallel computation.
*CoRR*, abs/2104.12369, 2021. URL https://arxiv.org/abs/2104.12369. -
Zhang & Sennrich (2019)
Biao Zhang and Rico Sennrich.
Root mean square layer normalization.
*Advances in Neural Information Processing Systems*, 32, 2019. -
Zhang et al. (2023a)
Chen Zhang, Luis F. D’Haro, Chengguang Tang, Ke Shi, Guohua Tang, and Haizhou Li.
xdial-eval: A multilingual open-domain dialogue evaluation benchmark.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 5579–5601. Association for Computational Linguistics, 2023a. doi: 10.18653/V1/2023.FINDINGS-EMNLP.371. URL https://doi.org/10.18653/v1/2023.findings-emnlp.371. -
Zhang et al. (2024a)
Hongbin Zhang, Kehai Chen, Xuefeng Bai, Yang Xiang, and Min Zhang.
Paying more attention to source context: Mitigating unfaithful translations from large language model.
*arXiv preprint arXiv:2406.07036*, 2024a. -
Zhang et al. (2023b)
Hongbo Zhang, Junying Chen, Feng Jiang, Fei Yu, Zhihong Chen, Guiming Chen, Jianquan Li, Xiangbo Wu, Zhiyi Zhang, Qingying Xiao, Xiang Wan, Benyou Wang, and Haizhou Li.
Huatuogpt, towards taming language model to be a doctor.
In Houda Bouamor, Juan Pino, and Kalika Bali (eds.),
*Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023*, pp. 10859–10885. Association for Computational Linguistics, 2023b. doi: 10.18653/V1/2023.FINDINGS-EMNLP.725. URL https://doi.org/10.18653/v1/2023.findings-emnlp.725. -
Zhang et al. (2023c)
Shaolei Zhang, Qingkai Fang, Zhuocheng Zhang, Zhengrui Ma, Yan Zhou, Langlin Huang, Mengyu Bu, Shangtong Gui, Yunji Chen, Xilin Chen, and Yang Feng.
Bayling: Bridging cross-lingual alignment and instruction following through interactive translation for large language models.
*CoRR*, abs/2306.10968, 2023c. doi: 10.48550/ARXIV.2306.10968. URL https://doi.org/10.48550/arXiv.2306.10968. -
Zhang et al. (2023d)
Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, and Guoyin Wang.
Instruction tuning for large language models: A survey.
*CoRR*, abs/2308.10792, 2023d. doi: 10.48550/ARXIV.2308.10792. URL https://doi.org/10.48550/arXiv.2308.10792. -
Zhang et al. (2024b)
Shimao Zhang, Changjiang Gao, Wenhao Zhu, Jiajun Chen, Xin Huang, Xue Han, Junlan Feng, Chao Deng, and Shujian Huang.
Getting more from less: Large language models are good spontaneous multilingual learners.
In
*Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pp. 8037–8051, 2024b. -
Zhang et al. (2022)
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer.
OPT: open pre-trained transformer language models.
*CoRR*, abs/2205.01068, 2022. doi: 10.48550/ARXIV.2205.01068. URL https://doi.org/10.48550/arXiv.2205.01068. -
Zhang et al. (2023e)
Wenxuan Zhang, Mahani Aljunied, Chang Gao, Yew Ken Chia, and Lidong Bing.
M3exam: A multilingual, multimodal, multilevel benchmark for examining large language models.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.),
*Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023e. URL http://papers.nips.cc/paper_files/paper/2023/hash/117c5c8622b0d539f74f6d1fb082a2e9-Abstract-Datasets_and_Benchmarks.html. -
Zhang et al. (2024c)
Yuanchi Zhang, Yile Wang, Zijun Liu, Shuo Wang, Xiaolong Wang, Peng Li, Maosong Sun, and Yang Liu.
Enhancing multilingual capabilities of large language models through self-distillation from resource-rich languages.
*CoRR*, abs/2402.12204, 2024c. doi: 10.48550/ARXIV.2402.12204. URL https://doi.org/10.48550/arXiv.2402.12204. -
Zhang et al. (2024d)
Yuanchi Zhang, Yile Wang, Zijun Liu, Shuo Wang, Xiaolong Wang, Peng Li, Maosong Sun, and Yang Liu.
Enhancing multilingual capabilities of large language models through self-distillation from resource-rich languages.
*arXiv preprint arXiv:2402.12204*, 2024d. -
Zhang et al. (2023f)
Zhihan Zhang, Dong-Ho Lee, Yuwei Fang, Wenhao Yu, Mengzhao Jia, Meng Jiang, and Francesco Barbieri.
PLUG: leveraging pivot language in cross-lingual instruction tuning.
*CoRR*, abs/2311.08711, 2023f. doi: 10.48550/ARXIV.2311.08711. URL https://doi.org/10.48550/arXiv.2311.08711. -
Zhang et al. (2024e)
Zhihao Zhang, Jun Zhao, Qi Zhang, Tao Gui, and Xuanjing Huang.
Unveiling linguistic regions in large language models.
*CoRR*, abs/2402.14700, 2024e. doi: 10.48550/ARXIV.2402.14700. URL https://doi.org/10.48550/arXiv.2402.14700. -
Zhao et al. (2024a)
Jun Zhao, Zhihao Zhang, Luhui Gao, Qi Zhang, Tao Gui, and Xuanjing Huang.
Llama beyond english: An empirical study on language capability transfer.
*CoRR*, abs/2401.01055, 2024a. doi: 10.48550/ARXIV.2401.01055. URL https://doi.org/10.48550/arXiv.2401.01055. -
Zhao et al. (2024b)
Wenlong Zhao, Debanjan Mondal, Niket Tandon, Danica Dillion, Kurt Gray, and Yuling Gu.
Worldvaluesbench: A large-scale benchmark dataset for multi-cultural value awareness of language models.
In Nicoletta Calzolari, Min-Yen Kan, Véronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (eds.),
*Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, LREC/COLING 2024, 20-25 May, 2024, Torino, Italy*, pp. 17696–17706. ELRA and ICCL, 2024b. URL https://aclanthology.org/2024.lrec-main.1539. -
Zhao et al. (2023)
Yao Zhao, Rishabh Joshi, Tianqi Liu, Misha Khalman, Mohammad Saleh, and Peter J. Liu.
Slic-hf: Sequence likelihood calibration with human feedback.
*CoRR*, abs/2305.10425, 2023. doi: 10.48550/ARXIV.2305.10425. URL https://doi.org/10.48550/arXiv.2305.10425. -
Zhao et al. (2024c)
Yiran Zhao, Wenxuan Zhang, Guizhen Chen, Kenji Kawaguchi, and Lidong Bing.
How do large language models handle multilingualism?
*CoRR*, abs/2402.18815, 2024c. doi: 10.48550/ARXIV.2402.18815. URL https://doi.org/10.48550/arXiv.2402.18815. -
Zhao et al. (2024d)
Yiran Zhao, Wenxuan Zhang, Guizhen Chen, Kenji Kawaguchi, and Lidong Bing.
How do large language models handle multilingualism?
*arXiv preprint arXiv:2402.18815*, 2024d. -
Zhao et al. (2024e)
Yiran Zhao, Wenxuan Zhang, Huiming Wang, Kenji Kawaguchi, and Lidong Bing.
Adamergex: Cross-lingual transfer with large language models via adaptive adapter merging.
*arXiv preprint arXiv:2402.18913*, 2024e. -
Zheng et al. (2023a)
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang.
Lmsys-chat-1m: A large-scale real-world LLM conversation dataset.
*CoRR*, abs/2309.11998, 2023a. doi: 10.48550/ARXIV.2309.11998. URL https://doi.org/10.48550/arXiv.2309.11998. - Zheng et al. (2023b) Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023b.
-
Zheng et al. (2023c)
Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Lei Shen, Zihan Wang, Andi Wang, Yang Li, Teng Su, Zhilin Yang, and Jie Tang.
Codegeex: A pre-trained model for code generation with multilingual benchmarking on humaneval-x.
In Ambuj K. Singh, Yizhou Sun, Leman Akoglu, Dimitrios Gunopulos, Xifeng Yan, Ravi Kumar, Fatma Ozcan, and Jieping Ye (eds.),
*Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2023, Long Beach, CA, USA, August 6-10, 2023*, pp. 5673–5684. ACM, 2023c. doi: 10.1145/3580305.3599790. URL https://doi.org/10.1145/3580305.3599790. -
Zheng et al. (2023d)
Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang, Lei Shen, Andi Wang, Yang Li, Teng Su, Zhilin Yang, and Jie Tang.
Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x.
*CoRR*, abs/2303.17568, 2023d. doi: 10.48550/ARXIV.2303.17568. URL https://doi.org/10.48550/arXiv.2303.17568. -
Zhou et al. (2023a)
Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy.
LIMA: less is more for alignment.
In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.),
*Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023a. URL http://papers.nips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html. -
Zhou & Cao (2021)
Fan Zhou and Chengtai Cao.
Overcoming catastrophic forgetting in graph neural networks with experience replay.
In
*Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021*, pp. 4714–4722. AAAI Press, 2021. doi: 10.1609/AAAI.V35I5.16602. URL https://doi.org/10.1609/aaai.v35i5.16602. -
Zhou et al. (2024)
Li Zhou, Taelin Karidi, Nicolas Garneau, Yong Cao, Wanlong Liu, Wenyu Chen, and Daniel Hershcovich.
Does mapo tofu contain coffee? probing llms for food-related cultural knowledge.
*CoRR*, abs/2404.06833, 2024. doi: 10.48550/ARXIV.2404.06833. URL https://doi.org/10.48550/arXiv.2404.06833. -
Zhou et al. (2023b)
Zhihan Zhou, Yanrong Ji, Weijian Li, Pratik Dutta, Ramana V. Davuluri, and Han Liu.
DNABERT-2: efficient foundation model and benchmark for multi-species genome.
*CoRR*, abs/2306.15006, 2023b. doi: 10.48550/ARXIV.2306.15006. URL https://doi.org/10.48550/arXiv.2306.15006. -
Zhu et al. (2024a)
Dawei Zhu, Pinzhen Chen, Miaoran Zhang, Barry Haddow, Xiaoyu Shen, and Dietrich Klakow.
Fine-tuning large language models to translate: Will a touch of noisy data in misaligned languages suffice?
*CoRR*, abs/2404.14122, 2024a. doi: 10.48550/ARXIV.2404.14122. URL https://doi.org/10.48550/arXiv.2404.14122. -
Zhu et al. (2024b)
Shaolin Zhu, Menglong Cui, and Deyi Xiong.
Towards robust in-context learning for machine translation with large language models.
In
*Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)*, pp. 16619–16629, 2024b. -
Zhu et al. (2024c)
Shaolin Zhu, Leiyu Pan, Bo Li, and Deyi Xiong.
Landermt: Dectecting and routing language-aware neurons for selectively finetuning llms to machine translation.
In
*Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 12135–12148, 2024c. -
Zhu et al. (2024d)
Shaolin Zhu, Leiyu Pan, and Deyi Xiong.
Feds-icl: Enhancing translation ability and efficiency of large language model by optimizing demonstration selection.
*Information Processing & Management*, 61(5):103825, 2024d. -
Zhu et al. (2023a)
Wenhao Zhu, Hongyi Liu, Qingxiu Dong, Jingjing Xu, Lingpeng Kong, Jiajun Chen, Lei Li, and Shujian Huang.
Multilingual machine translation with large language models: Empirical results and analysis.
*CoRR*, abs/2304.04675, 2023a. doi: 10.48550/ARXIV.2304.04675. URL https://doi.org/10.48550/arXiv.2304.04675. -
Zhu et al. (2023b)
Wenhao Zhu, Yunzhe Lv, Qingxiu Dong, Fei Yuan, Jingjing Xu, Shujian Huang, Lingpeng Kong, Jiajun Chen, and Lei Li.
Extrapolating large language models to non-english by aligning languages.
*CoRR*, abs/2308.04948, 2023b. doi: 10.48550/ARXIV.2308.04948. URL https://doi.org/10.48550/arXiv.2308.04948. -
Zhu et al. (2024e)
Wenhao Zhu, Shujian Huang, Fei Yuan, Chen Cheng, Jiajun Chen, and Alexandra Birch.
The power of question translation training in multilingual reasoning: Broadened scope and deepened insights.
*CoRR*, abs/2405.01345, 2024e. URL https://doi.org/10.48550/arXiv.2405.01345. -
Zhu et al. (2024f)
Wenhao Zhu, Shujian Huang, Fei Yuan, Shuaijie She, Jiajun Chen, and Alexandra Birch.
Question translation training for better multilingual reasoning.
*CoRR*, abs/2401.07817, 2024f. doi: 10.48550/ARXIV.2401.07817. URL https://doi.org/10.48550/arXiv.2401.07817. -
Ziegler et al. (2019)
Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul F. Christiano, and Geoffrey Irving.
Fine-tuning language models from human preferences.
*CoRR*, abs/1909.08593, 2019. URL http://arxiv.org/abs/1909.08593. -
Ziemski et al. (2016)
Michal Ziemski, Marcin Junczys-Dowmunt, and Bruno Pouliquen.
The united nations parallel corpus v1.0.
In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Sara Goggi, Marko Grobelnik, Bente Maegaard, Joseph Mariani, Hélène Mazo, Asunción Moreno, Jan Odijk, and Stelios Piperidis (eds.),
*Proceedings of the Tenth International Conference on Language Resources and Evaluation LREC 2016, Portorož, Slovenia, May 23-28, 2016*. European Language Resources Association (ELRA), 2016. URL http://www.lrec-conf.org/proceedings/lrec2016/summaries/1195.html. -
Zou et al. (2023)
Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson.
Universal and transferable adversarial attacks on aligned language models.
*CoRR*, abs/2307.15043, 2023. doi: 10.48550/ARXIV.2307.15043. URL https://doi.org/10.48550/arXiv.2307.15043.