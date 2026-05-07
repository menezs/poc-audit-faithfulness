UTF8gbsn

# A Survey on Multilingual Large Language Models: Corpora, Alignment, and Bias

###### Abstract

Based on the foundation of Large Language Models (LLMs), Multilingual Large Language Models (MLLMs) have been developed to address the challenges of multilingual natural language processing tasks, hoping to achieve knowledge transfer from high-resource to low-resource languages. However, significant limitations and challenges still exist, such as language imbalance, multilingual alignment, and inherent bias. In this paper, we aim to provide a comprehensive analysis of MLLMs, delving deeply into discussions surrounding these critical issues. First of all, we start by presenting an overview of MLLMs, covering their evolution, key techniques, and multilingual capacities. Secondly, we explore widely utilized multilingual corpora for MLLMs’ training and multilingual datasets oriented for downstream tasks that are crucial for enhancing the cross-lingual capability of MLLMs. Thirdly, we survey the existing studies on multilingual representations and investigate whether the current MLLMs can learn a universal language representation. Fourthly, we discuss bias on MLLMs including its category and evaluation metrics, and summarize the existing debiasing techniques. Finally, we discuss existing challenges and point out promising research directions. By demonstrating these aspects, this paper aims to facilitate a deeper understanding of MLLMs and their potentiality in various domains.

###### Index Terms:

Large Language Model, Multilingual Large Language Model, Corpora, Alignment, Bias, Survey.## I Introduction

The emergence of Large Language Models (LLMs) has brought about a paradigm shift and revolution in the field of Natural Language Processing (NLP). This innovative approach trains a transformer-based model [1] on extensive volumes of data and then leverages fine-tuning or prompt learning to facilitate the model’s adaption to a wide variety of tasks. Based on the foundation of LLMs, large-scale Multilingual Large Language Models (MLLMs), such as mBERT [2], XLM [3], mT5 [4], BLOOM [5] and LLaMA [6], have been developed to tackle multilingual NLP tasks. MLLMs are pre-trained on a concatenation of texts in multiple languages with the hope that low-resource languages may benefit from high-resource languages due to linguistic similarities and shared representations inherent within language pairs.

Compared to LLMs, MLLMs require larger multilingual corpora that cover more languages and diverse downstream tasks to ensure applicability and fairness across different languages. MLLMs are trained to understand and capture the structures and patterns of multiple languages. For instance, pre-trained on data from 104 languages, BLOOM supports 46 languages and 13 programming languages, covering the eight most widely spoken languages in the world [5]. Numerous MLLMs have been proposed in the past 5 years and the emergence of ChatGPT-like LLMs has accelerated the development of MLLMs. These MLLMs differ in the architecture (e.g., number of layers, parameters, etc), data used for pre-training (Wikipedia, Common Crawl, etc) and the number of languages involved (ranging from 12 to 110). However, it is uncertain how much cross-lingual transfer learning capability MLLMs have to support unseen languages or low-resource languages during pre-training. As a result, we first start by providing an overview of MLLMs in section II, which contains evolution, key techniques, and a detailed analysis of the multilingual capacities of MLLMs.

Despite the success of MLLMs, existing MLLMs still face numerous issues and challenges, which can be summarized as three aspects: corpora, alignment, and bias. As shown in figure 1, as input of MLLMs, corpora heavily influence the ability of MLLMs. On the one hand, unbalanced corpora and training lead to misalignment of MLLMs among different languages; On the other hand, inherent bias within corpora induces MLLMs to produce biased output. Therefore, our following discussion is presented around these three aspects.

Firstly, MLLMs heavily rely on multilingual corpora to enhance their performance. For example, among the training corpus of ChatGPT, the English corpus accounts for 92.099%, and the Chinese only accounts for 0.16%, so its dialogues in the English context are much higher than those in other languages in terms of quality and speed. However, the size of available corpus resources for different languages varies greatly, and most of the existing labelled datasets are mainly focused on a few languages, limited effectiveness exists in zero-shot cross-lingual transfer from high-resource languages to languages that are unseen during training. Furthermore, MLLMs suffer from what Conneau et al. [7] call the curse of multilinguality: more languages lead to better cross-lingual performance on low-resource languages up until a point, after which the overall performance of MLLMs on monolingual and cross-lingual benchmarks will decrease. In conclusion, the scale, quality, and diversity of corpora have a significant impact on performance of MLLMs. Therefore, In section III, we present a survey of the multilingual corpora that representative MLLMs trained on, offering insights into their language distribution, data source and language coverage.

Secondly, the success of MLLMs is their ability to achieve multilingual representation alignment from multiple languages. Aligning the representation of diverse languages acts as an integral part of NLP’s multilingual tasks and applications[8] and under-representation on low-resource languages leads to MLLMs’ poor performance on these languages. Inspired by the impressive performance of monolingual representation models like Word2vec [9] and GloVe [10], recent research has made great progress in multilingual representation. In section IV, we review previous research on multilingual word embeddings and classify them into three categories: static multilingual representation, contextual multilingual representation and combined multilingual representation. We also analyze the impact of various factors on multilingual alignment performance, including initial alignment solution, linearity of mapping function, language typological distance, and pre-training data and settings of MLLMs.

Thirdly, MLLMs are prone to produce harmful outcomes and social bias [11] in part due to bias is naturally present in cross-cultural datasets and the design of MLLMs’ modeling processes [12]. Previous studies have explored bias in various NLP tasks and demographic groups. However, these studies are specific to English-based models [13, 14] and cannot be generalized to other languages. What are the types of bias in existing MLLMs? What are the main de-biasing techniques available for MLLMs? Does the removal of these biases affect the performance of the large language models? What are the existing bias evaluation datasets for MLLMs? These are very worthwhile research questions. This survey tries to answer these questions and offers valuable insights for bias on MLLMs.

Summarizing the above discussion, the contributions of this survey are as follows:

-
•
We present an overview of MLLMs and analyze the the language imbalance challenge within MLLMs, their capacity to support low-resource languages and their potential for cross-lingual transfer learning.

-
•
We provide an overview of the datasets and corpora utilized by existing MLLMs to offer a comprehensive insight into the language distribution within these training datasets.

-
•
We survey the existing studies on multilingual representations and explore whether the current MLLMs can learn a universal language representation.

-
•
Our survey delves into bias within MLLMs, seeking to address essential questions such as identifying the types of bias present in current MLLMs, exploring prominent de-biasing techniques, summarizing available bias evaluation datasets for MLLMs.


## II Overview of MLLMs

This section provides a brief overview of MLLMs, tracing their evolution from monolingual large language models (LLMs) to multilingual LLMs. It then illustrates the key techniques that contribute most to the success of MLLMs, as well as the multilingual capacities of MLLMs.

### II-A Evolution of MLLMs

#### II-A1 Monolingual Evolution

The development of monolingual LLMs has made great progress in understanding and generating human languages. These models employ a transformer-based architecture to pre-train a large corpus of texts, followed by fine-tuning or prompt learning to enhance models’ performance on specific tasks or languages.

The representative monolingual LLMs are the BERT series and GPT series. With the success of BERT [2], BERT-variants models have been developed for specific languages, such as FlauBERT for French [15], BERTje for Dutch [16], AraBERT for Arabic [17]. The GPT series, evolving from GPT-1 to GPT-2, GPT-3, and beyond [18, 19, 20, 21, 22], has experienced growth in parameters and training corpora, e.g., parameters ranging from hundreds of millions (GPT-1) [18] to 1.5 trillion (GPT-3) [20]. This progression empowers the models with improved sophisticated language understanding and generation capabilities. T5 model [23] introduces a unified framework that converts a variety of tasks into a text-to-text format by prepending a unique prefix to the input for each task. BART [24] is a sequence-to-sequence model, not an auto-regressive model like GPT-2 or an auto-encoder model like BERT, and thus it is particularly effective on comprehension tasks like summarization.

Monolingual LLMs have seen advancements in transformer-based architecture and pre-training strategies but are language-dependent. Training such language-specific LLMs is only feasible for a few languages with necessary corpora.

#### II-A2 Multilingual Evolution

Multilingual LLMs build upon the foundation of monolingual LLMs to learn universal language patterns from extensive unlabeled data across multiple languages. MLLMs have advanced to overcome linguistic limitations and enhance low-resource languages by leveraging shared vocabulary and genetic relatedness from high-resource languages [25].

Following the success of monolingual BERT, multilingual BERT (mBERT) [2] was the first released MLLM by following the training procedure of BERT but on multilingual Wikipedia text corpora including 104 languages. Other MLLMs such as XLM-R [7], mBART[26] and mT5 [4] follow the step of mBERT, further explore the capacities and limitations of MLLMs across languages. Studies reveal that MLLMs work surprisingly well on cross-lingual tasks even without direct cross-lingual supervision like parallel or comparable data [27, 28].

The further development trend has led to a tremendous increase in model parameters and data scale, resulting in enhancements that promote multilingual capabilities. For example, PaLM with 540 billion (540B) parameters yields impressive capabilities on the multilingual benchmarks by training on a mixture of multilingual versions of Wikipedia and dialogue data, including 124 languages [29]. With the early successful attempts, the autoregressive language modeling and prompt learning paradigm represented by GPT series have received much attention and follow-up from major companies and universities. Thus, more MLLMs (e.g., InstructGPT [21], LaMDA [30], OPT [31], BLOOM [5], LLaMA [6]) have been proposed to achieve breakthrough performance in a range of multi-step reasoning tasks over multiple languages. In addition to the GPT series and its derivative models, numerous other models have been proposed to boost the development of LLMs. Examples include GLM [32, 33], Vicuna [34], Gemini [35], and several others.

The development of MLLMs has been guided by several tendencies: (1) parameter growth; (2) linguistic diversity; (3) multimodal unification. Regarding (1) MLLMs have been expanded to hundreds of billions of parameters or even trillions. Increasing the size of parameters brings clear benefits, such as alleviating hallucination phenomena heavily present in minor-parameter (e.g., 7B, 13B) models. However, there’s a limit to the amount of text data available online, and obtaining high-quality data is becoming increasingly challenging, which might slow down the parameter growth of MLLMs. Regarding (2), most high-resource languages belong to similar language families, thus sharing numerous linguistic features. Disregarding this diversity inevitably leads to poor generalizability and language-specific biases [36]. Recent work in MLLMs has focused on addressing this issue since low-resource and unseen languages still account for a large proportion of the world’s languages. Regarding (3), multimodal MLLMs are a growing focus of research, realizing a variety of specific real-world needs by unifying diverse types of modality (i.e., text, image, and speech). Additionally, current research aims to extend MLLMs to accommodate more modalities like web pages, heat maps, graphs, and tables, thereby increasing the model’s generality and applicability [37].

Table I summarizes some representative MLLMs in recent years, divided into two categories: monolingual and multilingual, showing the evolution of MLLMs from multiple perspectives in chronological order of release.

| Model | Release Time | Publishing Authority | Params | Context Length | Pre-training File Size | Architecture | Base Model | Pre-training Function | Publicly Available | Modal |
| Monolingual | ||||||||||
| GPT-1 [18] | Jun-18 | OpenAI | 117M | 2K | - | Decoder-only | GPT | LM | Open | Text |
| BERT [2] | Oct-18 | 340M | 2K | 1.3GB | Encoder-only | - | Seq2Seq MLM | Open | Text | |
| GPT-2 [19] | Feb-19 | OpenAI | 1.5B | 2K | 40GB | Decoder-only | GPT | LM | Open | Text |
| T5 [23] | Oct-19 | 11B | 2K | 21GB | En-decoder | - | Seq2Seq MLM | Open | Text | |
| BART [24] | Oct-19 | 400M | 2K | 20GB | En-decoder | - | DAE | Open | Text | |
| GPT-3 [20] | May-20 | OpenAI | 175B | 2K | 570GB | Decoder-only | GPT | LM | Closed | Text |
| Gopher [38] | Dec-21 | DeepMind | 280B | 2K | - | Decoder-only | - | LM | Open | Text |
| Multilingual | ||||||||||
| mBERT [2] | Jul-19 | 172M | 2K | - | Encoder-only | BERT | MLM | Open | Text | |
| XLM-R [7] | Nov-19 | 550M | 2K | - | Encoder-only | - | TLM | Open | Text | |
| mBART [26] | Jan-20 | 680M | 2K | - | En-decoder | BART | DAE | Open | Text | |
| mT5 [4] | Oct-20 | 13B | 2K | - | En-decoder | T5 | Seq2Seq MLM | Open | Text | |
| PanGu-α [39] | Apr-21 | Huawei | 200B | 2K | 100GB | Decoder-only | - | LM | Open | Text |
| LaMDA [30] | Jan-22 | 137B | 32K | - | Decoder-only | - | LM | Open | Text | |
| PaLM [29] | Apr-22 | 540B | 2K | - | Decoder-only | - | LM | Closed | Text | |
| OPT [31] | May-22 | Meta | 175B | 2K | 85GB | Decoder-only | - | LM | Open | Text |
| BLOOM [5] | Jul-22 | BigScience | 176B | 2K | 350GB | Decoder-only | - | LM | Open | Text |
| GLM-130B [33] | Aug-22 | ZHIPU | 130B | 2K | - | En-decoder | GLM | ABI | Closed | Text |
| FLAN-T5 [40] | Oct-22 | 11B | 2K | 17.3MB | En-decoder | T5 | LM | Open | Text | |
| GPT-3.5 [20] | Nov-22 | OpenAI | 175B | 2K | - | Decoder-only | GPT | LM | Closed | Text |
| ChatGPT [41] | Nov-22 | OpenAI | 175B | 2K | - | Decoder-only | GPT-3.5 | LM | Open | Text |
| LLaMA [6] | Feb-23 | Meta | 65B | 4K | 120GB | Decoder-only | - | LM | Open | Text |
| ChatGLM [38] | Mar-23 | ZHIPU | 130B | 2K | 8GB | En-decoder | GLM | ABI | Open | Text |
| PaLM-E [42] | Mar-23 | 562B | 2K | - | Decoder-only | PaLM | LM | Open | Text, Image | |
| Alpaca [43] | Mar-23 | StandFord | 7B | 2K | 14GB | Decoder-only | LLaMA | LM | Open | Text |
| GPT-4 [22] | Mar-23 | OpenAI | - | 8K | - | Decoder-only | GPT | LM | Closed | Text, Image |
| PanGu-Σ [44] | Mar-23 | Huawei | 1085B | - | - | Decoder-only | PanGu-α | LM | Closed | Text |
| Pythia [45] | Apr-23 | EleutherAI | 12B | 2K | 24GB | Decoder-only | - | LM | Open | Text |
| PaLM 2 [46] | May-23 | 340B | 2K | - | Decoder-only | PaLM | LM | Closed | Text | |
| ChatGLM2 [32] | Jun-23 | ZHIPU | 12B | 4K | - | En-decoder | GLM | ABI | Closed | Text |
| Vicuna [34] | Jun-23 | LMSYS | 33B | 2K | 65GB | Decoder-only | LLaMA | LM | Open | Text |
| LLaMA 2 [47] | Jul-23 | Meta | 70B | 4K | 129GB | Decoder-only | LLaMA | LM | Open | Text |
| Bard [48] | Jul-23 | - | - | - | Decoder-only | LaMDA | LM | Open | Text, Image | |
| Baichuan [49] | Jul-23 | BAICHUAN | 13B | 4K | 26.6GB | Decoder-only | - | LM | Open | Text |
| GPT-4V [22] | Sep-23 | OpenAI | - | 32K | - | Decoder-only | GPT-4 | LM | Closed | Text, Image |
| Baichuan2 [50] | Sep-23 | BAICHUAN | 13B | 4K | 27.8GB | Decoder-only | Baichuan | LM | Open | Text |
| ChatGLM3 [33] | Oct-23 | ZHIPU | 6B | 32K | 12GB | En-decoder | GLM | ABI | Open | Text |
| GPT-4 Turbo [22] | Nov-23 | OpenAI | - | 128K | - | Decoder-only | GPT-4 | LM | Closed | Text, Image, Speech |
| Geimini-ultra [35] | Dec-23 | DeepMind | 300B | 32K | - | Decoder-only | - | LM | Closed | Text, Image |
| Gemini-pro [35] | Dec-23 | DeepMind | 100B | 32K | - | Decoder-only | - | LM | Closed | Text, Image |
| Phi-2 [51] | Dec-23 | Microsoft | 2.7B | 2K | 5.4GB | Decoder-only | - | LM | Open | Text |
| GLM-4 [52] | Jan-24 | ZHIPU | - | 128K | - | En-decoder | GLM | ABI | Closed | Text, Image |

### II-B Key Techniques of MLLMs

Transformer architecture, pre-training technique, and reinforcement learning with human feedback are the key techniques for MLLMs. In this section, we present the key idea of these techniques.

#### II-B1 Transformer Architecture

Transformer architecture, first introduced in 2017, has become the foundation of MLLMs owing to its suitability for parallel computing and flexibility for diverse model design. Transformer architecture consists of two main modules, an Encoder, and a Decoder, along with a self-attention mechanism within the modules. The encoder, using stacked multi-head self-attention layers, encodes the input sequence and generates latent representations. In contrast, the decoder employs cross-attention to utilize the encoder’s latent representations, attending to them while autoregressively generating the target sequence [53].

MLLMs can be categorized into three groups based on the underlying transformer structure:

-
•
Encoder-only (e.g., GLM ): MLLMs with encoder-only architecture can effectively handle long-range dependencies within the input sequences, making them well-suited for the analysis and classification of textual content, including tasks like sentiment analysis and named entity recognition.

-
•
Decoder-only (e.g., PaLM): MLLMs with decoder-only architecture are mainly designed to generate sequences of language texts. They predict the next token based on contextual information from the current and preceding steps.

-
•
Encoder-decoder hybrid (e.g., mT5): MLLMs with encoder-decoder architecture enable themself to process sequential data and generate accurate and coherent outputs that excel in tasks such as text generation, and summarization.


#### II-B2 Pre-training Technique

Pre-training technique aims to learn universal language representations from billion-scale unlabeled corpus (e.g., Wikipedia, Webpages, News, etc.) and then initializes the parameters of the Transformer-based MLLMs. This approach reduces the reliance on massive parallel corpus, helping MLLMs generate similar representations in a common vector space for similar sentences and words (or words in similar context) across languages[54].

The benefits of pre-training technique can be attributed to two key factors: Paradigm and Task. Pre-training paradigms have been proposed to capture linguistic patterns in the training data and adapt MLLMs to downstream tasks, including “pre-training + fine-tuning” and “pre-training + prompting”. The former representative models are BERT [2], GPT-2 [19], while the latter presentative models like GPT-3 [20]. Pre-training tasks improve the ability of MLLMs to encode and generate coherent multilingual text.

In learning the universal representation of language, pre-training tasks play a crucial role and the widely used pre-training tasks include probabilistic language modeling (LM), masked language modeling (MLM), next sentence prediction (NSP), and Denoising autoencoder (DAE). Probabilistic LM is a fundamental task in NLP, estimating the probability distribution of sequences of words in a language. In practice, LM typically involves auto-regressive LM or unidirectional LM. MLM has emerged as a novel pre-training task to overcome the drawback of the standard unidirectional LM. By masking certain tokens in a sequence and predicting them based on context, MLM encourages models to learn bidirectional representations, capturing dependencies from both left and right contexts. Punctuations are the natural separators of text data. So, it is reasonable to construct pre-training methods by utilizing them. NSP is just a great example of this. NSP encourages the model to understand the contextual coherence and relationships between sentences. DAE takes a partially corrupted input and aims to recover the original undistorted input. Specific to language, a sequence-to-sequence model, such as the standard Transformer, is used to reconstruct the original text. Eq.1 summarize loss function of these pre-training tasks, where denotes a sequence of tokens [55].

| (1) | ||||

#### II-B3 Reinforcement Learning with Human Feedback

MLLMs may generate inaccurate or harmful outputs due to their probabilistic statistical text generation mechanism [56]. Reinforcement Learning from Human Feedback (RLHF) and its variants [57, 58, 59, 60, 61] has been proposed to fix this by optimizing MLLMs with human feedback, aligning them better with human values in three fundamental dimensions: helpfulness, honesty, and harmlessness [62].

Essentially, RLHF consists of three core steps [63, 64]:

-
1.
Pre-training a language Model: To pre-train MLLMs, extensive prompts, and multilingual datasets are utilized as examples, teaching the model how to respond appropriately in a specific context.

-
2.
Training a Reward Model: Prompts serve as input to MLLMs, where pairs of {prompt, response} are manually scored by human evaluators to align with human preferences. The rankings of {sample, reward} pairs are normalized into a scalar reward signal for training the Reward Model (RM).



The loss function is defined as follows, where is the prompt, and denote the better and worse model responses respectively, and is the output of the RM.(2) - 3.

### II-C Multilingual Capacities of MLLMs

Pre-training MLLMs on extensive multilingual data enhances their multilingual capacities and cross-lingual transfer learning (CLTL) from one language to another. However, MLLMs still face challenges in training with multilingual corpora and their exact CLTL capabilities remain unknown. This section focuses on these two concerns.

#### II-C1 Challenges brought by Multilingual Corpora

Three challenges arise from multilingual corpora training. Firstly, while MLLMs outperform monolingual LLMs in downstream tasks for high-resource languages, their performance on low-resource languages remains unsatisfactory due to limited annotated data. Secondly, the “curse of multilinguality” phenomenon in MLLMs worsens this situation. Supporting more languages can lead to a significant performance decline in performance for low-resource languages, making them victims of this curse [67]. Thirdly, the distribution of languages in the pre-training corpora is highly skewed towards English, further complicating efforts to address the “curse of multilinguality” phenomenon.

To mitigate these challenges, two approaches have been proposed. One involves fine-tuning existing MLLMs to suit the linguistic features of low-resource languages [68]. However, this method is constrained by the demand for extensive specific-task annotated training data [69]. Alternatively, another approach is to pre-train monolingual LLMs on low-resource languages [70]. This method allows models to learn from diverse sources and contexts within the target language without requiring costly annotated data. Therefore, MLLMs trained by this approach exhibit superior performance on low-resource languages compared to the aforementioned fine-tuning approach. For example, Torge et al. [71] pre-trained monolingual RoBERTa models for Czech, Polish, and a bilingual model for Czech-Polish, which demonstrated superior performance to the current state-of-the-art multilingual model, XLM-R, across various downstream tasks. Recently, there has been a growing interest in developing low-resource language models to meet the demands of morphologically rich, low-resource languages. Examples include language-specific BERT models like FlauBERT for French [15], BERTje for Dutch [16], FinBERT for Finnish [72], among others [54].

The main reason for MLLMs’ poor performance on low-resource languages is the skewed distribution of languages in the pre-training data. Therefore, techniques have been proposed to address this issue. Data sampling techniques like exponential weighted smoothing [7] help prevent the under-representation of low-resource languages, while vocabulary augmentation approaches [73] enrich the model’s vocabulary by inducing new tokens of unseen languages during training. Moreover, research also attempted to tackle the language imbalance. Choenni et al. discovered that languages influence each other during the pre-training phrase and MLLMs benefit from reinforcement or complementary learning [74]. Wang et al. emphasized the significance of imbalanced learning algorithms in Vision-Language models (VLMs) [75]. For example, CLIP model demonstrated an improvement from 5% to 69% on iNaturalist dataset by adopting imbalanced methods. Jiang et al. proposed a data augmentation pipeline to address imbalance in social media, effectively handling multiclass problems[76].

#### II-C2 Cross-lingual Transfer Learning brought by Multilingual Corpora

MLLMs can facilitate CLTL from one language to another. This naturally raises the question of how much CLTL capability that MLLMs possess to support these unseen languages or low-resource languages during pre-training.

Research has been dedicated to exploring the cross-lingual transferability of MLLMs through zero-shot learning. Lin et al. [77] trained 4 multilingual generative language models and examined their zero-shot and in-context few-shot learning capabilities in a wide range of tasks. They found that these models can achieve cross-lingual few-shot learning in non-English languages without requiring source-to-target language translation. Tian et al. [78] found that MLLMs exhibit strong rumour detection performance in zero-shot cross-lingual transfer learning. What’s more, MLLMs showed surprisingly strong multilingual reasoning abilities even in under-represented languages such as Bengali and Swahili [79].

To further improve the transfer learning performance of MLLMs on the unseen or low-resource languages, as these languages still account for a significant portion of the world’s languages, MLLMs are pre-trained to learn languages from the same linguistic family or branch [80, 81, 82]. MLLMs trained on a small amount of data from genetically related languages could achieve performance comparable to the ones trained on large but unrelated data [80]. MLLMs trained on only low-resource languages with small datasets but similar languages sometimes achieved better performance than models trained on large datasets with high-resource languages [81]. For example, the AfriBERTa model [82], pre-trained on less than 1 GB of text data from 11 African languages, most of which belong to the Bantu branch of the Niger-Congo language family, demonstrated the effectiveness of scratching solely on low-resource languages without any high-resource transfer learning.

A prominent future concern will be how to improve the CLTL capacities of MLLMs. Pikuliak et al. conducted a survey on existing cross-lingual transfer paradigms of MLLMs [83] and Philippy et al. investigated various factors that impacted cross-lingual transfer performance, including linguistic similarity, lexical overlap, model architecture, pre-training setting, and pre-training corpus size [84]. Specifically, this avenue of research seeks to investigate how and why MLLMs possess different CLTL abilities on various languages. This pursuit holds the potential to leverage CLTL capacities to mitigate the dependence on annotated data and maintain or even enhance the performance of MLLMs in well-trained or unseen languages.

## III Multilingual Corpora and Datasets

In this section, we delve into the widely utilized multilingual corpora for MLLMs’ associated training corpora and multilingual datasets oriented for downstream tasks. Tabel II summarizes the multilingual corpora that representative MLLMs trained on, offering insights into their language distribution, data source and language coverage.

MLLMs have a more extensive language coverage in their training data compared to LLMs. A significant portion of these training data originates from multilingual repositories like Common Crawl, WikiPedia and Web documents, encompassing a broad range of languages. These multilingual repositories are crucial for enhancing the cross-lingual capability of MLLMs. In this section, we discuss training data’s language composition from both a general perspective and a language family perspective.

| Model | Language | Language proportion | Source |
| mBERT [2] | 104 languages | Unknown | Wikipedia |
| XLM-R [7] | 100 languages | English (12.56%); Russian (11.61%); Others (63.89%) Indonesian (6.19%); Vietnamese (5.73%) | Generated using the open source; CC-Net repository |
| mT5 [4] | 101 languages | English (5.67%); Russian (3.71%); Spanish (3.09 %); German (3.05%); Others (84.48%) | Common Crawl |
| GPT-3 [20] | 95 languages | English (92.7%); French (1.8%); German (1.5%); Others (5.9%) | Common Crawl; Wikipedia; Books1; Books2; WebText2 |
| Gopher [38] | 51 languages | Over 99% English | MassiveWeb (48%); C4 (10%); News (10%); Books (27%); GitHub (3%);Wikipedia (2%) |
| LaMDA [30] | Unknown | Over 90% English | Public dialog data and other public web documents |
| InstructGPT [21] | Unknown | Over 96% English | Text prompts written by labelers or from the OpenAI API |
| PaLM [29] | Over 100 languages | English (77.98%); German (3.50%); French (3.25%); Spanish (2.11%); Others (13.15%) | Social media conversations (50%); Filtered webpages (27%); Books (13%); GitHub (5%); Wikipedia (4%); News (1%) |
| BLOOM [5] | 46 languages | English (30.03%); Simplified Chinese (16.16%); French (12.9%); Spanish (10.85%); Portuguese (4.91%); Arabic (4.6%); Others (20.55%) | Web Crawl(38%); BigScience Catalogue Data(62%) |
| LLaMA[6] | Over 20 languages | Over 67% English | Common Crawl (67.0%); C4 (15.0%); Github (4.5%);Wikipedia (4.5%); Books (4.5%); ArXiv (2.5%); StackExchange (2.0%) |
| Vicuna [34] | Unknown | Unknown | User-shared conversations from ShareGPT.com |
| Falcon [85] | Over 100 languages | Excluding English: Russian (13.19%); German (10.81%); Spanish (9.45%); Others (66.55%) | Common Crawl |
| PaLM 2 [46] | Over 100 languages | Excluding English: Spanish (11.51%); Chinese (10.19%); Russian (8.73%); Others (69.57%) | Web documents; books; code; mathematics; conversational data |
| LLaMA 2 [47] | Over 100 languages | English (89.70%); Unknown (8.38%); German (0.17%); France (0.16%); Others (1.59%) | Publicly available sources excludes Meta user data |
| GLM-130B [33] | English; Chinese | English(48%); Chinese(52%) | Pile English; Chinese Wudao-Corpora; Chinese corpora crawled from the web |

### III-A Multilingual Corpora in MLLMs

First, we analyze the linguistic composition of MLLMs’ training data, investigating the total number of languages and different language proportions within each training corpora. Analysis reveals that most MLLMs are trained on corpora where English is the predominant language. Notably, several MLLMs, including GPT-3 [20], Gopher [38], LaMDA [30] and InstructGPT [21], are trained on corpora where English comprises over 90%. The overwhelming English texts in corpora lead to MLLMs’ English-centric ability. To alleviate this issue, some MLLMs are trained on corpora with more balanced language distribution. For example, the training data of BLOOM [5] covers 46 languages and English only comprises less than half. GLM-130B [33] makes great efforts to balance its training data, achieving a close 1:1 ratio between English and Chinese training data. Compared to its base model PaLM [29], PaLM 2 [46] includes a higher percentage of non-English data, further enhancing its multilingual capabilities. We present the percentages of its non-English language in the web documents sub of its pre-training corpus in Table II, as the language distribution for the English language was not published.

Second, we explore the language composition of MLLMs’ training data from a language family perspective. Languages within the same language family share similar characteristics and MLLMs have better transfer performance on languages belonging to the same language family [54]. Thus, the proposition of language families in MLLMs’ training data can help us better understand the multilingual capabilities of MLLMs. What’s more, we can also leverage language families to observe the linguistic composition of the MLLMs’ training data. Since English is predominant in most MLLMs’ corpora, considering it in language family analysis would heavily favor the Indo-European language family to which English belongs. To gain a more detailed understanding of the language family proposition in MLLMs’ corpora, we exclude English and focus on the top 20 prominent non-English languages of the training data and their corresponding language families. The distribution of language families of each MLLM is shown in Fig. 3.

Notably, French, German, Chinese, and Spanish emerge as the most prevalent languages in the training data. For example, French constitutes 1.8% of the training corpora for GPT-3[20] and 12.9% of the training corpora for PaLM[29]. French, German, and Spanish all belong to the Indo-European language family, which demonstrates that the Indo-European language family holds a prominent position in MLLMs’ corpora, both in terms of quantity and language diversity. An exception to this is Chinese, which belongs to the Sino-Tibetan language family while maintaining a significant presence in the training corpora. But in terms of language diversity, the Sino-Tibetan language family in the training corpora, mainly consisting of the Chinese language, is much less diverse compared to the Indo-European language family. Besides Indo-European and Sino-Tibetan language families, some other language families are found in most MLLMs’ training corpora as well. Similar to Sino-Tibetan, they mainly contain only one language in training corpora. For example, Austronesian mainly includes Indonesian, Japonic mainly includes Japanese, and Koreanic mainly includes Korean. Apart from the lack of diversity within the same language family, there is also a lack of diversity across different language families in MLLMs’ training corpora. For example, despite Niger-Congo and Trans-New Guinea being among the largest language families in the world, they are notably absent from the top 20 languages in the training data.

Through the above analysis of multilingual training corpora in MLLMs, we have derived the following key insights: MLLMs broaden language coverage beyond LLMs, yet English remains dominant in their training corpora. From a language family perspective, Indo-European languages occupy a prominent place in terms of both quantity and linguistic variety. Further work should consider a more comprehensive inclusion of language families and prioritize language diversity within the same language family when training MLLMs.

### III-B Multilingual Datasets for Downstream Tasks

| Task | Dataset | Language | Size | Source |
| Multilingual NER | Masakha NER2.0 [86] | 20 African languages | 4.8K to 11K sentences per language | News articles |
| Multilingual NER | MultiCo NER [87] | 11 languages | 26M tokens | Wikipedia; ORCAS dataset MS-MARCO QnA corpus; |
| Multilingual SA | XED[88] | 32 languages | More than 950 lines per language | OPUS |
| Multilingual SA | NollySenti[89] | 5 languages | 1K to 1.5K reviews per language | Movie reviews |
| Multilingual SA | NaijaSenti[90] | 5 languages | 30K tweets | |
| Cross-Lingual IR | AfriCLIR Matrix [91] | 15 African languages | 6M English queries and 23M relevance judgments | Wikipedia |
| Cross-Lingual IR | CLIRMatrix [92] | 8 languages | 49M unique queries and 34B (query, document, label) triplets | Wikipedia |
| Multilingual TC | Taxi1500 [93] | Over 1500 languages | About 1K verses per language | Parallel translations of the Bible |
| Multilingual TC | MARC [94] | 6 languages | 210K reviews per language | Amazon reviews |
| Multilingual Versatile | MUSE [95] | 110 language pairs | About 6.5K word pairs for each language pair | Self-created |
| Multilingual Versatile | Wikipedia Monolingual Copora[96] | 30 languages | 10B tokens | Wikipedia |
| Multilingual Versatile | Multilingual Open Text [97] | 44 languages | Over 2.8M news articles and an additional 1M short snippets | VOA News |

Multilingual datasets play a crucial role in fine-tuning MLLMs to be adaptive across various NLP tasks. Table III summarizes some representative multilingual datasets, including Multilingual Named Entity Recognition (Multilingual NER), Multilingual Sentiment Analysis (Multilingual SA), Cross-Lingual Information Retrieval (Cross-Lingual IR), and Multilingual Text Classification (Multilingual TC).

Multilingual NER. Named Entity Recognition tasks locate and classify named entities from unstructured natural language. These tasks utilize datasets from sources like News and Wikipedia, which provide rich contextual information across a wide range of real-world entities. Efforts have been made to expand the training data for low-resource languages. A notable example is Masakha NER2.0 [86], the largest human-annotated Africa-centric dataset, deriving its data from African local news.

Multilingual SA. Sentiment analysis tasks, which focus on the sentiment orientation of data, often utilize datasets extracted from comments or reviews found on reviews platforms such as Amazon and IMDb, as well as social media platform like Facebook and Twitter. The sentiment analysis dataset XED [88] is sourced from OPUS [98], a parallel corpus extracted from movie subtitles. In terms of language diversity, while XED [88] primarily focuses on English and Finnish, NollySenti [89] and NaijaSenti [90] are sentiment analysis datasets specifically designed for African languages such as Hausa, Igbo, Nigerian, Pidgin and Yoruba.

Cross-Lingual IR. Cross-Lingual Information Retrieval tasks ask queries in one language and retrieve documents in one or more other languages. These tasks utilize datasets that include documents containing hyperlinks to parallel documents in different languages. Therefore, many datasets such as AfriCLIR Matrix [91] and CLIR Matrix [92] are sourced from multilingual encyclopedia (e.g.,Wikipedia). CLIR Matrix [92] is the current largest and most comprehensive CLIR dataset. It includes Arabic, German, English, Spanish, French, Japanese, Russian, and Chinese, covering mainly the common languages of all continents except Africa. Thus, AfriCLIR Matrix [91] was developed to address the absence of African languages.

Multilingual TC. Text Classification tasks have diversified applications on news classification, sentiment classification and so on. These tasks utilize diverse datasets tailored for specific applications. For example, Multilingual Amazon Reviews Corpus (MARC) [94]，which includes product category and star rating, can be used for both product classification or sentiment classification. Taxi1500 [93], covering more than 1500 languages, relies solely on the parallel translation of the Bible as its data source, limiting its domain to only religious-related text classification. However, as Bible is the most translated book, its parallel translation is good data source to enhance language diversity in datasets.

Multilingual Versatile. Besides the above mentioned multilingual datasets, Wikipedia Monolingual Corpora [96], MUSE [95] and Multilingual Open Text (MOT) [97] are widely used multilingual datasets for general NLP tasks. Wikipedia Monolingual Corpora [96] covers 30 languages. Each language has its own XML file containing the full monolingual Wikipedia contents with annotations like article and paragraph boundaries, number of links referring to each article, cross-language links and more. MUSE [95] provides state-of-the-art multilingual word embeddings aligned in a single vector space for 30 languages and 110 large-scale ground-truth bilingual dictionaries. Multilingual Open Text (MOT) [97] comprises news articles and short snippets (photo captions, video descriptions and etc.) from Voice of America (VOA) news websites. It was designed to supply high-quality unlabeled texts for lower-resource languages like Albanian, Amharic and Persian. It contains complete collection of VOA’s documents which can be further annotated for various NLP works (e.g., document classification, syntactic or semantic parsing).

## IV Multilingual Representation Alignment

The success of MLLMs is their ability to achieve multilingual representation alignment from multiple languages. Table IV summarizes some multilingual alignment performance of MLLMs on 10 languages and three cross-lingual tasks: bilingual lexicon induction (BLI), cross-lingual classification (XNLI), and machine translation (MT). The evaluation metrics include accuracy (for BLI and XNLI) and BLEU (for MT). The performance of MLLMs on multilingual alignment varies across languages, with better performance observed for English and its closely related languages.

Aligning the representation of diverse languages acts as an integral part of NLP’s multilingual tasks and appplications[8]. Inspired by the impressive performance of monolingual representation models like Word2vec [9] and GloVe [10], recent research has made great progress in multilingual representation. Fig. 4 summarizes the evolution of multilingual representation from static approaches to more dynamic ones like contextual and combined multilingual representations. This evolution is highly influenced by the introduction of MLLMs and their enhanced multilinguality.

Static multilingual representations are attained through learning a mapping matrix to align two monolingual embedding spaces, while contextual ones can be achieved by both mapping and joint approaches, with the latter being supported by MLLMs. To achieve even better alignment, combined methods were proposed to take advantage of both static and contextual information. Details of the three paradigms will be explained below. Furthermore, we also discuss the factors that will affect multilingual alignment.

| Model | Task | Evaluation Metric | ES | DE | FR | RU | AR | ZH | BG | TR | HI | Avg. |
| fastText [101] | BLI | Accuracy | 72.00 | 67.17 | - | 56.42 | 47.43 | 33.39 | 45.69 | 48.92 | 28.19 | 49.90 |
| BLOOM-7B [5] | BLI | Accuracy | 52.50 | 38.34 | - | 26.06 | 32.67 | 34.35 | 16.75 | 30.82 | 28.30 | 32.47 |
| LLaMA-13B [6] | BLI | Accuracy | 60.58 | 57.80 | - | 64.44 | 22.13 | 32.28 | 56.86 | 44.90 | 30.68 | 46.21 |
| GPT-3.5 [20] | BLI | Accuracy | 68.17 | 63.07 | - | 74.15 | 65.94 | 65.12 | 67.51 | 54.49 | 56.11 | 64.32 |
| BLI | Accuracy | 63.31 | 56.60 | - | 55.27 | 42.02 | 41.29 | 46.70 | 44.78 | 35.82 | 48.23 | |
| ✓ | ✓ | - | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ||||
| mBERT [2] | XNLI | Accuracy | 68.0 | 70.0 | 64.3 | 73.4 | 67.8 | 60.9 | 73.5 | 58.9 | 57.2 | 66.00 |
| mT5-270M [4] | XNLI | Accuracy | 78.6 | 77.4 | 73.3 | 79.1 | 77.1 | 72.8 | 80.3 | 70.8 | 68.3 | 75.30 |
| XLM-R-270M [7] | XNLI | Accuracy | 80.7 | 78.7 | 79.7 | 78.1 | 73.8 | 76.7 | 79.6 | 74.2 | 72.4 | 77.10 |
| mT5-10.7B [4] | XNLI | Accuracy | 87.7 | 87.3 | 84.5 | 86.9 | 85.1 | 83.8 | 87.8 | 83.2 | 79.8 | 85.12 |
| XLM-R-10.7B [7] | XNLI | Accuracy | 87.3 | 87.0 | 86.2 | 82.5 | 82.5 | 82.6 | 85.7 | 82.0 | 79.8 | 83.96 |
| XNLI | Accuracy | 80.46 | 80.08 | 77.68 | 80.0 | 77.26 | 75.36 | 81.38 | 73.82 | 71.50 | 77.50 | |
| ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ||||
| XGLM-7.5B [102] | MT | BLEU | 27.98 | 34.03 | 36.81 | 27.83 | 26.06 | 6.06 | 34.48 | 23.91 | 26.99 | 27.13 |
| OPT-175B [31] | MT | BLEU | 30.81 | 39.15 | 43.02 | 18.80 | 1.03 | 12.36 | 11.48 | 24.39 | 1.17 | 20.25 |
| Falcon-7B [85] | MT | BLEU | 30.13 | 34.60 | 41.62 | 14.26 | 1.81 | 22.78 | 8.07 | 10.05 | 1.26 | 18.29 |
| LLaMA2-7B [47] | MT | BLEU | 33.09 | 41.94 | 44.11 | 33.44 | 22.35 | 26.26 | 38.18 | 21.75 | 21.04 | 31.35 |
| ChatGPT [41] | MT | BLEU | 33.48 | 43.56 | 46.13 | 38.04 | 38.94 | 30.05 | 41.65 | 38.14 | 38.15 | 38.68 |
| GPT-4 [22] | MT | BLEU | 33.76 | 47.04 | 48.81 | 38.75 | 43.29 | 32.83 | 44.97 | 43.43 | 45.88 | 42.09 |
| MT | BLEU | 31.54 | 40.05 | 43.42 | 28.52 | 22.25 | 21.72 | 29.81 | 26.95 | 22.42 | 29.62 | |
| ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |

### IV-A Static multilingual representation

Based on whether parallel corpora are used or not, static alignment approaches can be categorized into three groups: supervised, semi-supervised, and unsupervised approaches. Recently, unsupervised approaches, such as MUSE [95] and VecMap [103], have gained much more attention.

Let and represent monolingual word embeddings from two languages, respectively. Static alignment approaches can be roughly divided into two steps: initially, the introduction of an initial mapping by aligning the source and target language distributions; subsequently, a pseudo-supervised refinement based on the initial solution, where the transformation matrix is constrainted to be orthogonal, i.e., .

| (3) |

Orthogonal constraint serves as a means to ensure monolingual invariance but is not held for all languages, particularly for the semantically distant languages [104]. Therefore, weak orthogonal constraints have been proposed to better align the embeddings across different languages.

Generally, linear projection only learns one global transformation matrix to project the entire embedding space of the source to that of the target. However, the global transformation matrix does not consistently perform optimally across all subspaces [105]. To address this issue, specific mappings for different subspaces were proposed [106].

Static multilingual representations have exhibited promising performance but there is still ample room for improvement on low-resource languages and distant language pairs. Besides, the polysemy problem in static multilingual representation has not been well addressed and needs further exploration.

### IV-B Contextual multilingual representation

Contextual representation is introduced to address the polysemy challenge faced in static representation. ELMo [107] and BERT [2] stand out as the highly representative models for contextual monolingual representation. Contextual multilingual word representation can be derived from these models. The existing approaches to contextual multilingual representation can be categorized into two groups: mapping approach and joint approach.

Mapping approaches use pre-trained contextual monolingual embeddings from various languages as input and project them into a shared semantic space [108]. However, two challenges remain. Firstly, the computation cost for pre-trained monolingual embeddings rises exponentially with the number of languages. Secondly, in contextual approaches, alignment is more challenging compared to static ones. Simply calculating a mapping alone is no longer sufficient to generate robust alignments [8].

In comparison, joint approaches supported by MLLMs belong to an end-to-end process, which no longer requires pre-trained monolingual representations but instead depends on unlabeled multilingual corpora. Tokenization is a critical technique in the end-to-end process, segmenting raw data from various languages into sequences of tokens for subsequent processing by MLLMs. Transformer-based MLLMs commonly employ subword-level tokenizers, such as Byte-Pair Encoding (BPE) [109] and WordPiece [110], to address out-of-vocabulary (OOV) issue. What’s more, variants of BPE have been proposed to improve the tokenization of multilingual corpora and alleviate lexical overlap between languages.

In summary, contextual multilingual representation contains richer in-context information than the static multilingual approaches and thus shows greater potential for multilinguality. However, there is still a range of multilingual NLP tasks that contextual multilingual approaches underperformed than static ones, demonstrating several challenges remain:

-
1.
It comes with higher computational costs and is far more resource-intensive during both training and inference.

- 2.
-
3.
The alignment between low-resource languages and distant language pairs has not been well investigated.


### IV-C Combined multilingual language representation

Combined multilingual representation has been proposed to take advantage of both static and contextual paradigms. The existing combined multilingual approaches can be divided into two paradigms: (1) From Static to Contextual (S2C), leveraging static information to induce better contextual multilingual alignment [113]; (2) From Contextual to Static (C2S), leveraging contextual information to induce better static multilingual alignment [114, 115].

S2C achieves higher-quality contextual representation by integrating extra static instruction, while C2S achieves higher-quality static representation by integrating extra contextual information. Although S2C makes contextual approaches easier to interpret, accurate extraction of contextual representations from MLLMs is still in challenge.

Therefore, C2S is a better way for multilingual representation alignment. Existing C2S can be divided into two steps: 1) roughly achieving static multilingual representations like introduced in section IV-A; 2) fine-tuning static multilingual representations by leveraging contextual representations. Zheng et al. [114] proposed a spring network to use the contextual representations to pull the static word embeddings to better positions in the unified space for easy alignment. Li et al. [115] fine-tune pre-trained multilingual LMs to extract more useful representations and then combine static and extracted contextual embeddings to achieve high-quality cross-lingual word embeddings.

### IV-D Factors That Affect Alignments

Based on the aforementioned discussion, we delve into the impact of various factors on multilingual alignment performance and investigate which factors have a more significant impact.

Initial Solution. For mapping approaches, the initial solution plays a crucial role in alignment. Because subsequent optimization is based on this initial solution, it will affect the robustness of the final result and cause the alignment to fall into a local optimum. Based on their use of annotated data, mapping approaches can be categorized as supervised, semi-supervised, and unsupervised methods. For supervised and semi-supervised methods, the quality of the initial solution depends on the quality and amount of the seed dictionary, while unsupervised ones depend on the robustness and effectiveness of embedding spaces’ distribution matching, which is more difficult. GAN-based adversarial training [95], optimal transport solution [116], auto-encoder [117] and graph alignment [118] were utilized to better match distribution and find a better initial solution in a fully unsupervised way.

Linearity of mapping. Mapping functions are always constrained to be orthogonal during training out of the “approximate isomorphism assumption”, which fails especially when the two languages are far apart semantically. To address this issue, Mohiuddin et al. [119] and Glavaš and Vulic [120] used a non-linear Mapping function. Marchisio et al. [121] considered relative isomorphism during the process of pre-training monolingual embedding, which can address the misalignment from the root.

Typological distance. More typologically distant language pairs tend to be less well-aligned than more similar ones [122]. In the Bilingual Lexicon Induction (BLI) task, the accuracy on semantically distant language pairs is always under 40%, while similar ones are over 80%. To alleviate this problem, auxiliary languages have been proposed as a medium to bridge the gap between semantically distant language pairs [3, 123]. For distant language pairs, one or several more relevant languages can be selected as auxiliary languages. Transferring the additional information provided by the auxiliary Languages monolingual embedding or corpora can improve the alignment between distant language pairs.

Pre-Training Data and Settings. Pre-training data and settings are found to be correlated with the cross-lingual transfer ability. The size and quality of data are crucial factors for enhanced cross-lingual transfer capabilities in MLLMs. The relative balance and diversity in the pre-training data and the larger data size will improve the efficiency and effectiveness of MLLMs [7]. The settings of pre-training are also important to the cross-lingual performance of MLLMs. The parameters scale [8], pre-training learning objective [124] and window size of input of MLLMs [125] have proved to be influential to cross-lingual transfer ability.

## V Bias on Multi-lingual Language Models

Bias acquired by MLLMs, such as gender bias, race bias, and language bias, pose significant challenges to the fairness of MLLMs, which severely restricts the deployment of MLLMs in real-world applications. Existing literature on bias mainly focuses on various stereotypical biases in English [13, 14], which limits its generalizability to other languages. Additionally, prior research primarily focused on bias within individual languages or limited attributes in LLMs [126]. Bias in MLLMs has not been well investigated. This section gives a systematic and in-depth review of bias in MLLMs across various languages.

We aim to address the following questions. Why do MLLMs bias and what are the types of bias in existing MLLMs (Bias Category), how to evaluate bias in MLLMs (Bias Benchmark), what can be done to mitigate the bias, and whether debiasing techniques affect the performance of MLLMs (Debias Technique).

### V-A Bias Category

Bias in MLLMs can arise from factors such as unmoderated training data [127], differences in model design [12] and the presence of biased multilingual word embedding representation [128]. Based on studies related to bias in MLLMs, we categorize these prevalent biases centered around specific languages, limited attributes, and related models into three types: language bias, demographic bias, and evaluation bias.

Language bias. Language bias refers to the unequal performances of MLLMs among different languages, primarily due to the dominance of English and other major languages in the available multilingual training corpora. Specifically speaking, MLLMs exhibit higher proficiency in these widely used languages and this further exacerbated the lack of support for low-resource languages or minority dialects [129]. Recent studies have brought attention to the unequal quality of multilingual representations, highlighting that pre-trained models like mBERT and CLIP do not equally learn high-quality representations for all languages, particularly for low-resource languages [130] [131]. When investigating knowledge in MLLMs, Kassner et al. [132] found mBERT exhibited language bias, wherein the choice of query language can impact the obtained results. To go a step further, studies in [133] [134] explored how MLLMs exhibited bias across languages and focused on bias in attributes like race, religion, nationality, and gender. They found that mBERT and XLM-R model did not consistently show low-level bias in certain languages [133]; mBERT, XLM-R, and mT5 exhibited varying degrees of fairness across languages and XLM-R exhibited higher and more consistent correlations across languages compared to mBERT and mT5 [134].

Demographic bias. Demographic bias refers to the MLLMs’ biased behavior towards specific gender, race, ethnicity, or other social groups, caused by the training data disproportionately emphasizing particular demographic groups [129]. Previous research has shown that both multilingual and monolingual LLMs suffer from demographic bias towards specific social groups [135] [136], while monolingual LLMs specific for low-resource languages exhibit less bias [70]. Touileb et al. [135] investigated demographic bias in Norwegian demographics, finding that both language-specific models like Norwegian pre-trained language models and MLLMs like XLM-R demonstrated a bias towards gender-balanced occupations. Likewise, research in [136] discovered that MLLMs like BLOOM and ChatGPT, along with monolingual LLMs trained exclusively on Arabic data, displayed cultural bias towards Western culture. This is evidenced by the fact that when processing and generating Arabic texts, Western-appropriate content is usually preferred over relevant Arabic content. Notably, LLMs for low-resource languages like Sudanese exhibited gender-neutral behavior without displaying distinct biases [70]. Additionally, bias against a particular cultural group is a common manifestation of demographic bias. Levy et al. [133] revealed that mBERT and XLM-R favored culturally dominant groups in each language. GPT-3 has been found to exhibit a stereotypical religious bias for associating Muslims with violence more often than other religious groups [137].

Evaluation bias. Evaluation bias refers to the bias that exists in the evaluation metrics for LLMs. Factors that can bias the metric calculation itself include noise in the evaluation dataset, models used in the metric calculation, and the configuration of the inference experiment [138]. Significantly, if bias against certain sensitive attributes, such as gender, occurs in the evaluation metrics, models that reinforce such bias are likely to be rewarded and favored [139]. For this reason, Sun et al. [140] conducted a systematic study of social biases in various PLMs-based metrics, such as BERTScore [141], BLEURT [142] and BARTScore [143]. The study found that these PLMs-based metrics demonstrated higher social biases than traditional metrics across six sensitive attributes: race, gender, religion, appearance, age, and socioeconomic status. Further analysis revealed that the choice of modeling paradigms [143] (matching, regression, or generation) in PLMs-based metrics has a greater impact on fairness than the choice of PLMs themselves. To assess the bias evaluation of LLMs, Koo et al. [144] proposed COBBLER, the COgnitive Bias Benchmark for evaluating the quality and reliability of LLMs as automatic evaluators. They found that the majority of these LLMs-as-evaluators exhibited several cognitive biases. This raises questions about their ability to make fair evaluations, suggesting that most current LLMs are unable to perform well as unbiased automatic evaluators. Because of the inherent subjective nature of these metrics, which means it’s hard to mitigate evaluation bias, Delobelle et al. [145] recommended avoiding embedding-based metrics and focusing on fairness assessments in downstream tasks to improve the evaluation of bias.

### V-B Bias Benchmark

This section focuses on the issue of bias evaluation in MLLMs. Extensive studies have developed varied datasets and approaches that serve as benchmarks for bias assessment. In this section, we provide a thorough review of these benchmarks. Table VI illustrates benchmarks commonly used for evaluating bias. Notably, these datasets primarily focus on bias attributes related to gender and occupation [146, 147, 148], predominantly available in English [149, 150, 151, 152]. Several datasets also encompass languages such as Spanish, German, and French [128] [134].

Based on the tasks and languages, benchmarks in Table VI can be categorized into three types: general benchmarks, task-specific benchmarks, and language-specific benchmarks.

General benchmarks mainly refer to evaluation benchmarks that have a wide range of applications and can be used for different tasks, including some major evaluation metrics and datasets. For example, Association Test (WEAT, SEAT, and CEAT) [153, 154, 155] are widely used to measure bias in word-, sentence- and contextualized-level embeddings; GLUE [150] is designed to measure the impact that the introduced debiasing techniques will have on downstream performance by evaluating the capabilities of the NLP model.

Task-specific benchmarks refer to benchmark datasets designed for a specific task or situation. For example, Winogender [146] and WinoBias [147] are applicable for the coreference resolution system; CrowS-Pairs [151] is designed for detecting bias against social groups, particularly in the United States.

Multilingual benchmarks refer to the benchmark datasets in multilingual contexts, including MIBs [128] and MozArt [134]. The lack of robust multilingual evaluation benchmarks poses significant barriers to assessing biases in multilingual contexts. Therefore, creating more multilingual evaluation datasets is an urgent problem to be solved. One potential solution is to translate existing bias benchmarks that mainly only cover English [156, 157]. Nevertheless, it is important to note that translated benchmarks may introduce additional biases due to translation errors and cultural differences. Thus, when designing a multilingual bias benchmark, it’s crucial to consider various cultural contexts and develop cultural-diverse datasets [12].

| Benchmark | Time | Benchmark/Metric | Type of bias | Goal |
| General Category | ||||
| WEAT [153] | 2017 | Evaluation metric | Gender | Measure bias in word embeddings |
| GLUE [150] | 2018 | English benchmark | Untargeted | Evaluate how debiasing techniques affect downstream task performance. |
| SEAT [154] | 2019 | Evaluation metric | Gender | Measure bias in sentence encoders. |
| CEAT [155] | 2020 | Evaluation metric | Untargeted | Measure bias in contextualized word embeddings. |
| InBias [128] | 2020 | Evaluation metric | Gender, Occupation | Quantify intrinsic bias in multilingual word embeddings. |
| ExBias [158] | 2020 | Evaluation metric | Gender, Occupation | Measure debiasing word embeddings by comparing their performance before and after debiasing. |
| StereoSet [14] | 2020 | English benchmark | Gender, Occupation, Race etc. | Evaluate the stereotypical biases of popular PLMs. |
| Task-specific Category | ||||
| Winogender [146] | 2018 | English benchmark | Gender, Occupation | Identify bias in in coreference resolution systems. |
| WinoBias [147] | 2018 | English benchmark | Gender, Occupation | Identify bias in coreference resolution systems. |
| EEC [149] | 2018 | English benchmark | Gender, Race | Measure bias of race and gender through differences in predicting sentiment intensity between sentences. |
| CrowS-Pairs [151] | 2018 | English benchmark | Race, Religion, Age etc. | Measure certain social bias in LLMs. |
| WinoMT [148] | 2019 | English benchmark | Gender | Investigate gender bias in machine translation systems. |
| BiosBias [152] | 2019 | English benchmark | Gender, Occupation | Evaluate bias in predicting individual occupation based on their short biography. |
| FairFace [159] | 2019 | Face Attribute benchmark | Gender, Race, Age | Evaluate how to mitigate bias in existing databases by collecting more diverse facial images. |
| Language-specific Category | ||||
| MIBs [128] | 2020 | English, Spanish, German, and French benchmark | Gender, Occupation | Conduct the intrinsic bias analysis. |
| MozArt [134] | 2022 | English, Spanish, German, and French benchmark | Gender, Language | Evaluate whether MLLMs are equally fair to demographic groups across languages. |

### V-C Debias Technique

Current debiasing techniques for MLLMs can be broadly categorized into model debiasing and data debiasing. model debiasing techniques rely on refining MLLMs’ inner settings like pre-training parameters, fine-tuning datasets, and representations, while data debiasing focuses on addressing bias within the input training data of MLLMs.

#### V-C1 Model Debiasing

The existing methods for debiasing models can be categorized into four lines: representation based methods, pre-training based methods, fine-tuning based methods, and prompt based methods.

Representation based methods. Representation, commonly employed to encode semantic information of texts, has the potential to encode unintended biases. For example, words associated with specific professions like “nurses” and “homemakers”, may cluster near feminine words, acting as a potential source of semantic bias for downstream models [153]. Representations based methods aim to mitigate bias at sentence-level [160] or word-level [161].

Sentence-level methods：Sent-Debias is introduced to debias sentence-level representations by estimating a linear subspace for a particular type of bias [160]. The debiasing process involves projecting onto the estimated bias subspace and subtracting the resulting projection from the original sentence representations.

Word-level methods：They focus on static [161] or contextual embedding representations [162]. For example, INLP [161] was proposed to remove bias like race, gender, and age in static word embeddings with iterative null-space projection-based debiasing method. Linguistic Identity Removal (LIR) [162] was proposed to addressed bias in multilingual contextual word embeddings. It utilized singular value decomposition and orthogonal projection to identify and remove linguistic information in multilingual semantic space.

Pre-training based methods. In this approach, debiasing occurs during the pre-training stage, where the parameters of LLMs are modified to align with fairness criteria such as SEAT [154]. Dropout as proposed in [163], is a bias mitigation technique using dropout regularization [164]. By adjusting dropout parameters in BERT and ALBERT for attention weights and hidden activations, along with performing an extra phrase of pre-training, gender bias within these models can be alleviated. However, this method cannot guarantee whether the bias associations may resurge when the debiased models are fine-tuned on downstream tasks [165].

Fine-tuning based methods. In this approach, debiasing occurs during the fine-tuning stage, which is independent of the model architecture or pre-training parameters, making it applicable across various downstream tasks. Leonardo et al. [166] proposed a debiasing approach for LLMs through fine-tuning using causal language modeling. They selectively froze a large number of parameters and trained the model using LoRA [167]. This technique yields robust debiased models that maintain high performance on downstream tasks.

However, fine-tuning the models on top of the pre-training stage carries the risk of inheriting biases, given that biases from the pre-trained stage tend to propagate to the fine-tuned models. Therefore, it is more beneficial to effectively manipulate the fine-tuned dataset to debias than to intervene in the pre-trained model itself [168]. In addition, fine-tuning all pre-trained parameters requires huge computing resources and time, and it is crucial to address how to debias effectively with a smaller set of parameters.

Prompt based methods. This approach mitigates biases in MLLMs without heavily relying on additional corpora for fine-tuning, as low-quality corpora may introduce new biases. Studies found that prompting can reduce bias in MLLMs but its success is largely dependent on the chosen prompt [169] [170]. Prompt based debiasing methods need to address two issues: how to measure biases carried by MLLMs and how to debias them.

For example, Guo et al. [169] proposed a framework named Auto-Debias, using cloze-style prompts to probe, identify and correct the biases in PLMs. This method first searches for the biased prompts probes the biased content with such prompts, and then corrects the model bias. Mattern et al. [170] explored GPT-3’s stereotypical associations with genders and jobs and proposed a framework to quantify and further reduce these biases using debiasing prompts. They also discussed prompt selection with varying degrees of abstraction and concluded that more concrete debiasing prompts exhibited a more pronounced effect. Dhingra et al. [171] demonstrated that employing a method involving chain-of-thought prompting through SHAP analysis can efficiently mitigate biases against queer people in the output of LLMs. Schick et al. [172] introduced a debiasing technique named Self-Debias which uses a model’s internal knowledge to discourage biased text generation. It starts by utilizing hand-crafted prompts to encourage the model to generate toxic text. Subsequently, a second continuation that is non-discriminatory can be produced from the model by scaling down the probabilities of tokens considered likely under the first toxic generation.

#### V-C2 Data Debiasing

Data debiasing aims to mitigate bias within input training corpora, helping MLLMs generate debiased content. Currently, prevalent data debiasing efforts focus on two types of bias: language bias and demographic bias.

Language bias mitigation. Language bias in MLLMs is caused by the imbalanced language proportion, acting as the dominance of English and other major languages in the available multilingual training corpora. Constructing more balanced corpora has proven to be an effective solution for mitigating language bias. For example, XNLI [173] was developed to support 15 languages on the evaluation of XLU, providing information-rich standard evaluation tasks for cross-language sentence understanding. In addition, the release of CulturaX [174], a multilingual dataset that includes 167 languages and a total of 63,000 tokens, addresses the lack of open-source and easy-to-use datasets for effectively training multilingual large models. Furthermore, The ROOTS dataset [175] was developed to cover 59 languages with a total size of 1.6TB.

However, building more balanced corpora also faces many challenges. First, manually collecting and annotating low-resource data requires high human costs. To prevent the introduction of additional bias, relatively professional data annotators are required and need to be trained in advance. Second, a large part of the low-resource corpora is of low quality. Kreutzer et al. [176] found a large part of the corpora contained less than 50 % of sentences of acceptable quality and discussed the potential risks of releasing low-quality data. In short, evaluating and improving the techniques to build high-quality multilingual corpora is essential for development of MLLMs.

Demographic bias mitigation. Demographic bias occurs when data overly emphasizes or represents a certain specific population. The commonly used method for mitigating demographic bias is counterfactual data augmentation. Based on identifying biased terms, it creates text that contradicts existing facts, reducing over-reliance on specific scenarios or groups and mitigating biases stemming from class imbalances within data. With the method, model’s reliance on false features can largely reduce and thus enhance the model’s robustness. Counterfactual augmented data is mainly achieved through two methods: manual generation and model generation, where both of them achieved comparable quality of generation [177]. Existing studies [178, 179, 180] have shown that counterfactual data augmentation is a simple and effective approach to mitigate bias in data.

Apart from its impressive performance in mitigating bias within datasets, counterfactual augmented data can also serve as an evaluation tool for detecting bias existing in MLLMs. Counterfactual data augmentation alters certain variables or features in the original data to highlight different data points. This method aids in understanding how changes in these variables affect the system’s output, uncovering potential biases or dependencies not readily apparent in the original dataset [181]. However, it also has limitations and drawbacks, such as possibly overlooking context information, causing the model to confuse key features [182], or preventing the model from learning robust features that have not been perturbed [183], and it may even exacerbate false correlations in the data.

As a conclusion, although some progress has been made in data debiasing, there are still some limitations and challenges. First, although language bias can be mitigated by constructing more balanced corpora, how to effectively collect and integrate high-quality low-resource language texts is still a problem [176]. Second, although counterfactual data augmentation alleviates demographic bias to some extent, it may affect other performance of the model. Therefore, how to make debiased data prevent model generating bias while maintaining model performance is a promising direction worth exploring in depth.

## VI Future Directions

This survey provides a holistic, systematic overview of the evolution of multilingual large language models. The MLLMs are still in a developing stage and thus there are still several challenges for future research, which we summarize below:

-
•
Performance on Low-resource Languages. MLLMs outperform monolingual LLMs in downstream tasks for high-resource languages, but their performance on low-resource languages remains unsatisfactory [184], which may be due to limited annotated data [185] for low-resource languages and low lexical overlap between high-resource and low-resource languages [186]. Specializing MLLMs based on language families can be an efficient way to more easily share information across languages [187]. In addition, how to find a more robust tokenizer for most languages is worth investigating as well.

-
•
Limited and Unbalanced Multilingual Corpora. The performance of MLLMs largely depends on the training data’s quality, size, and diversity [188]. However, there is only a limited amount of data available for most of the world’s languages. The overwhelming English texts in corpora lead to MLLMs’ English-centric ability. Even though for some high-resource languages where data is available, previous work has shown that some commonly used multilingual resources have severe quality issues [189]. How to collect much more high-quality, larger scale, and more diverse training data from various languages deserves further research.

-
•
Usage of Multimodal Data Sources. Leveraging information from multimodal data sources such as speech and images can alleviate high reliance on text data. Human cognition and perception capabilities rely on diverse information, and the usage of multimodal data can better align with human intentions. Supported by multimodal data equates to higher quality, more diverse training data. However, how to achieve universal representation accurately by modality alignment poses a new challenge, deserving further investigation.

-
•
Evaluation of Multilingual LLMs. The evaluation benchmarks for MLLMs are mainly based on the development of English task sets. However, these benchmarks are not fully applicable to other languages. Although some task sets can be translated into other languages, due to the differences between languages, the performance of the translated data set will be lower than the source language. Besides, current evaluation benchmarks are all task-centric, lacking a universal and flexible evaluation system. The topic of how to collect high-quality multilingual evaluation datasets and build a system to properly evaluate the true multilinguality of MLLMs is still undervalued.

-
•
Ethical Impact of Multilingual LLMs. Multilingual LLMs can inherit biases present in their training data, leading to ethical risk of generation. Due to the high proportion of Western language data in training data, the MLLMs are inclined to reflect Western-centric concepts [190]. How to mitigate biases and ensure fairness and culturally sensitive in text generation are key challenges for the further development of MLLMs.


## References

-
[1]
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”
*Advances in neural information processing systems*, vol. 30, 2017. -
[2]
J. D. M.-W. C. Kenton and L. K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in
*Proc. NAACL-HLT*, 2019, pp. 4171–4186. -
[3]
A. Conneau and G. Lample, “Cross-lingual language model pretraining,”
*Advances in neural information processing systems*, vol. 32, 2019. -
[4]
L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel, “mt5: A massively multilingual pre-trained text-to-text transformer,” in
*Proc. 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2021, pp. 483–498. -
[5]
T. L. Scao
*et al.*, “Bloom: A 176b-parameter open-access multilingual language model,”*arXiv preprint arXiv:2211.05100*, 2022. -
[6]
H. Touvron
*et al.*, “Llama: Open and efficient foundation language models,” 2023. -
[7]
A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, É. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, “Unsupervised cross-lingual representation learning at scale,” in
*Proc. 58th Annual Meeting of the Association for Computational Linguistics*, 2020, pp. 8440–8451. -
[8]
S. Cao, N. Kitaev, and D. Klein, “Multilingual alignment of contextual word representations,” in
*8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*, 2020. -
[9]
T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, “Distributed representations of words and phrases and their compositionality,”
*Advances in neural information processing systems*, vol. 26, 2013. -
[10]
J. Pennington, R. Socher, and C. D. Manning, “Glove: Global vectors for word representation,” in
*Proc. 2014 conference on empirical methods in natural language processing (EMNLP)*, 2014, pp. 1532–1543. -
[11]
E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell, “On the dangers of stochastic parrots: Can language models be too big?” in
*Proceedings of the 2021 ACM conference on fairness, accountability, and transparency*, 2021, pp. 610–623. -
[12]
Z. Talat
*et al.*, “You reap what you sow: On the challenges of bias evaluation under multilingual settings,” in*Proc. BigScience Episode# 5–Workshop on Challenges & Perspectives in Creating Large Language Models*, 2022, pp. 26–41. -
[13]
B. Hutchinson, V. Prabhakaran, E. Denton, K. Webster, Y. Zhong, and S. Denuyl, “Social biases in nlp models as barriers for persons with disabilities,” in
*Proc. 58th Annual Meeting of the Association for Computational Linguistics*, 2020, pp. 5491–5501. -
[14]
M. Nadeem, A. Bethke, and S. Reddy, “Stereoset: Measuring stereotypical bias in pretrained language models,” in
*Proc. 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, 2021, pp. 5356–5371. -
[15]
H. Le, L. Vial, J. Frej, V. Segonne, M. Coavoux, B. Lecouteux, A. Allauzen, B. Crabbé, L. Besacier, and D. Schwab, “Flaubert: Unsupervised language model pre-training for french,” in
*Proc. Twelfth Language Resources and Evaluation Conference*, 2020, pp. 2479–2490. -
[16]
W. de Vries, A. van Cranenburgh, A. Bisazza, T. Caselli, G. van Noord, and M. Nissim, “Bertje: A dutch BERT model,”
*CoRR*, vol. abs/1912.09582, 2019. -
[17]
W. Antoun, F. Baly, and H. Hajj, “Arabert: Transformer-based model for arabic language understanding,” in
*Proc. 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection*, 2020, pp. 9–15. -
[18]
A. Radford, K. Narasimhan, T. Salimans, I. Sutskever
*et al.*, “Improving language understanding by generative pre-training,” 2018. -
[19]
A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language models are unsupervised multitask learners,”
*OpenAI blog*, vol. 1, no. 8, p. 9, 2019. -
[20]
T. Brown
*et al.*, “Language models are few-shot learners,”*Advances in neural information processing systems*, vol. 33, pp. 1877–1901, 2020. -
[21]
L. Ouyang
*et al.*, “Training language models to follow instructions with human feedback,”*Advances in Neural Information Processing Systems*, vol. 35, pp. 27 730–27 744, Nov. 2022. -
[22]
J. Achiam
*et al.*, “Gpt-4 technical report,”*arXiv preprint arXiv:2303.08774*, 2023. -
[23]
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,”
*The Journal of Machine Learning Research*, vol. 21, no. 1, pp. 5485–5551, 2020. -
[24]
M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” in
*Proc. 58th Annual Meeting of the Association for Computational Linguistics*, 2020, pp. 7871–7880. -
[25]
T. Q. Nguyen and D. Chiang, “Transfer learning across low-resource, related languages for neural machine translation,” in
*Proc. Eighth International Joint Conference on Natural Language Processing (Volume 2: Short Papers)*, 2017, pp. 296–301. -
[26]
Y. Liu, J. Gu, N. Goyal, X. Li, S. Edunov, M. Ghazvininejad, M. Lewis, and L. Zettlemoyer, “Multilingual denoising pre-training for neural machine translation,”
*Transactions of the Association for Computational Linguistics*, vol. 8, pp. 726–742, 2020. -
[27]
T. Pires, E. Schlinger, and D. Garrette, “How multilingual is multilingual bert?” in
*Proc. 57th Annual Meeting of the Association for Computational Linguistics*, 2019, pp. 4996–5001. -
[28]
M. Artetxe, S. Ruder, and D. Yogatama, “On the cross-lingual transferability of monolingual representations,” in
*Proc. 58th Annual Meeting of the Association for Computational Linguistics*, 2020, pp. 4623–4637. -
[29]
A. Chowdhery
*et al.*, “Palm: Scaling language modeling with pathways,”*Journal of Machine Learning Research*, vol. 24, no. 240, pp. 1–113, 2023. -
[30]
R. Thoppilan
*et al.*, “Lamda: Language models for dialog applications,” 2022. -
[31]
S. Zhang
*et al.*, “Opt: Open pre-trained transformer language models,”*arXiv preprint arXiv:2205.01068*, 2022. -
[32]
Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang, “Glm: General language model pretraining with autoregressive blank infilling,” in
*Proc. 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2022, pp. 320–335. -
[33]
A. Zeng
*et al.*, “Glm-130b: An open bilingual pre-trained model,” in*The Eleventh International Conference on Learning Representations*, 2022. -
[34]
W.-L. Chiang
*et al.*, “Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality,” March 2023. [Online]. Available: https://lmsys.org/blog/2023-03-30-vicuna/ -
[35]
G. Team
*et al.*, “Gemini: a family of highly capable multimodal models,”*arXiv preprint arXiv:2312.11805*, 2023. -
[36]
P. Rust, J. Pfeiffer, I. Vulić, S. Ruder, and I. Gurevych, “How good is your tokenizer? on the monolingual performance of multilingual language models,” in
*Proc. 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, Online, Aug. 2021, pp. 3118–3135. -
[37]
D. Zhang, Y. Yu, C. Li, J. Dong, D. Su, C. Chu, and D. Yu, “Mm-llms: Recent advances in multimodal large language models,”
*arXiv preprint arXiv:2401.13601*, 2024. -
[38]
J. W. Rae
*et al.*, “Scaling language models: Methods, analysis & insights from training gopher,” 2022. -
[39]
W. Zeng
*et al.*, “Pangu-: Large-scale autoregressive pretrained chinese language models with auto-parallel computation,”*arXiv preprint arXiv:2104.12369*, 2021. -
[40]
H. W. Chung
*et al.*, “Scaling instruction-finetuned language models,”*arXiv preprint arXiv:2210.11416*, 2022. - [41] OpenAI. (2022). [Online]. Available: https://openai.com/blog/chatgpt
-
[42]
D. Driess
*et al.*, “Palm-e: An embodied multimodal language model,”*arXiv preprint arXiv:2303.03378*, 2023. - [43] R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto, “Stanford alpaca: An instruction-following llama model,” 2023.
-
[44]
X. Ren
*et al.*, “Pangu-Sigma: Towards trillion parameter language model with sparse heterogeneous computing,”*arXiv preprint arXiv:2303.10845*, 2023. -
[45]
S. Biderman
*et al.*, “Pythia: A suite for analyzing large language models across training and scaling,” in*International Conference on Machine Learning*. PMLR, 2023, pp. 2397–2430. -
[46]
R. Anil
*et al.*, “Palm 2 technical report,” Google, Tech. Rep., 2023. -
[47]
H. Touvron
*et al.*, “Llama 2: Open foundation and fine-tuned chat models,”*arXiv preprint arXiv:2307.09288*, 2023. - [48] GOOGLE, “An overview of bard: an early experiment with generative ai,” https://ai.google/static/documents/google-about-bard.pdf, 2023.
- [49] BAICHUAN,, “Blog: Baichuan-7b,” https://github.com/baichuan-inc/Baichuan-7B, 2023.
-
[50]
A. Yang
*et al.*, “Baichuan 2: Open large-scale language models,”*arXiv preprint arXiv:2309.10305*, 2023. - [51] MICROSOFT, “Phi-2: The surprising power of small language models,” https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/, 2023.
- [52] ZHIPU,, “Zhipu ai devday glm-4,” https://zhipuai.cn/en/devday, 2024.
-
[53]
W. X. Zhao
*et al.*, “A survey of large language models,”*CoRR*, vol. abs/2303.18223, 2023. -
[54]
S. Doddapaneni, G. Ramesh, A. Kunchukuttan, P. Kumar, and M. M. Khapra, “A primer on pretrained multilingual language models,”
*CoRR*, vol. abs/2107.00676, 2021. -
[55]
X. Qiu, T. Sun, Y. Xu, Y. Shao, N. Dai, and X. Huang, “Pre-trained models for natural language processing: A survey,”
*Science China Technological Sciences*, vol. 63, no. 10, pp. 1872–1897, 2020. -
[56]
T. Shen, R. Jin, Y. Huang, C. Liu, W. Dong, Z. Guo, X. Wu, Y. Liu, and D. Xiong, “Large language model alignment: A survey,”
*arXiv preprint arXiv:2309.15025*, 2023. -
[57]
A. Glaese
*et al.*, “Improving alignment of dialogue agents via targeted human judgements,”*ArXiv*, vol. abs/2209.14375, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:252596089 -
[58]
Y. Bai
*et al.*, “Training a helpful and harmless assistant with reinforcement learning from human feedback,”*arXiv preprint arXiv:2204.05862*, 2022. -
[59]
R. Liu, G. Zhang, X. Feng, and S. Vosoughi, “Aligning generative language models with human values,” in
*Findings of the Association for Computational Linguistics: NAACL 2022*, 2022, pp. 241–252. -
[60]
A. Baheti, X. Lu, F. Brahman, R. L. Bras, M. Sap, and M. Riedl, “Improving language models with advantage-based offline policy gradients,”
*arXiv preprint arXiv:2305.14718*, 2023. -
[61]
D. Go, T. Korbak, G. Kruszewski, J. Rozen, N. Ryu, and M. Dymetman, “Aligning language models with preferences through f-divergence minimization,”
*arXiv preprint arXiv:2302.08215*, 2023. -
[62]
A. Askell
*et al.*, “A general language assistant as a laboratory for alignment,”*arXiv preprint arXiv:2112.00861*, 2021. -
[63]
N. Lambert, L. Castricato, L. von Werra, and A. Havrilla, “Illustrating reinforcement learning from human feedback (rlhf),”
*Hugging Face Blog*, 2022, https://huggingface.co/blog/rlhf. -
[64]
N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano, “Learning to summarize with human feedback,”
*Advances in Neural Information Processing Systems*, vol. 33, pp. 3008–3021, 2020. -
[65]
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,”
*arXiv preprint arXiv:1707.06347*, 2017. -
[66]
V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, “Asynchronous methods for deep reinforcement learning,” in
*International conference on machine learning*, New York, New York, USA, 20–22 Jun 2016, pp. 1928–1937. -
[67]
F. R. M., “Catastrophic forgetting in connectionist networks,”
*Trends in cognitive sciences*, vol. 3, pp. 128–135, 1999. -
[68]
M. A. Hedderich, L. Lange, H. Adel, J. Strötgen, and D. Klakow, “A survey on recent approaches for natural language processing in low-resource scenarios,” in
*Proc. 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2021, pp. 2545–2568. -
[69]
J. O. Alabi, D. I. Adelani, M. Mosbach, and D. Klakow, “Adapting pre-trained language models to african languages via multilingual adaptive fine-tuning,” in
*Proc. 29th International Conference on Computational Linguistics*, 2022, pp. 4336–4349. -
[70]
W. Wongso, H. Lucky, and D. Suhartono, “Pre-trained transformer-based language models for sundanese,”
*Journal of Big Data*, vol. 9, no. 1, p. 39, 2022. -
[71]
S. Torge, A. Politov, C. Lehmann, B. Saffar, and Z. Tao, “Named entity recognition for low-resource languages-profiting from language families,” in
*Proc. 9th Workshop on Slavic Natural Language Processing 2023 (SlavicNLP 2023)*, 2023, pp. 1–10. -
[72]
S. Rönnqvist, J. Kanerva, T. Salakoski, and F. Ginter, “Is multilingual bert fluent in language generation?” in
*Proc. First NLPL Workshop on Deep Learning for Natural Language Processing*, 2019, pp. 29–36. -
[73]
Z. Wang, K. Karthikeyan, S. Mayhew, and D. Roth, “Extending multilingual bert to low-resource languages,” in
*Findings of the Association for Computational Linguistics: EMNLP 2020*, 2020, pp. 2649–2656. -
[74]
R. Choenni, D. Garrette, and E. Shutova, “How do languages influence each other? studying cross-lingual data sharing during llm fine-tuning,”
*arXiv preprint arXiv:2305.13286*, 2023. -
[75]
Y. Wang, Z. Yu, J. Wang, Q. Heng, H. Chen, W. Ye, R. Xie, X. Xie, and S. Zhang, “Exploring vision-language models for imbalanced learning,”
*arXiv preprint arXiv:2304.01457*, 2023. -
[76]
Y. Jiang, R. Qiu, Y. Zhang, and P.-F. Zhang, “Balanced and explainable social media analysis for public health with large language models,” in
*Australasian Database Conference*, 2023, pp. 73–86. -
[77]
X. V. Lin
*et al.*, “Few-shot learning with multilingual generative language models,” in*Proc. 2022 Conference on Empirical Methods in Natural Language Processing*, 2022, pp. 9019–9052. -
[78]
L. Tian, X. Zhang, and J. H. Lau, “Rumour detection via zero-shot cross-lingual transfer learning,” in
*Machine Learning and Knowledge Discovery in Databases. Research Track: European Conference, ECML PKDD 2021, Bilbao, Spain, September 13–17, 2021, Proceedings, Part I 21*, 2021, pp. 603–618. -
[79]
F. Shi
*et al.*, “Language models are multilingual chain-of-thought reasoners,” in*The Eleventh International Conference on Learning Representations*, 2022. -
[80]
T. Ogunremi, D. Jurafsky, and C. D. Manning, “Mini but mighty: Efficient multilingual pretraining with linguistically-informed data selection,” in
*Findings of the Association for Computational Linguistics: EACL 2023*, 2023, pp. 1221–1236. -
[81]
K. Ogueji, Y. Zhu, and J. Lin, “Small data? no problem! exploring the viability of pretrained multilingual language models for low-resourced languages,” in
*Proc. 1st Workshop on Multilingual Representation Learning*, 2021, pp. 116–126. - [82] K. Ogueji, “Afriberta: Towards viable multilingual language models for low-resource languages,” Master’s thesis, University of Waterloo, 2022.
-
[83]
M. Pikuliak, M. Šimko, and M. Bieliková, “Cross-lingual learning for text processing: A survey,”
*Expert Systems with Applications*, vol. 165, pp. 113–165, 2021. -
[84]
F. Philippy, S. Guo, and S. Haddadan, “Towards a common understanding of contributing factors for cross-lingual transfer in multilingual language models: A review,” in
*Proc. 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023*, 2023, pp. 5877–5891. -
[85]
G. Penedo, Q. Malartic, D. Hesslow, R. Cojocaru, A. Cappelli, H. Alobeidli, B. Pannier, E. Almazrouei, and J. Launay, “The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only,”
*arXiv preprint arXiv:2306.01116*, 2023. -
[86]
D. I. Adelani
*et al.*, “Masakhaner 2.0: Africa-centric transfer learning for named entity recognition,” in*Proc. 2022 Conference on Empirical Methods in Natural Language Processing*, 2022, pp. 4488–4508. -
[87]
S. Malmasi, A. Fang, B. Fetahu, S. Kar, and O. Rokhlenko, “Multiconer: A large-scale multilingual dataset for complex named entity recognition,” in
*Proc. 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022*, 2022, pp. 3798–3809. -
[88]
E. Öhman, M. Pàmies, K. Kajava, and J. Tiedemann, “XED: A multilingual dataset for sentiment analysis and emotion detection,” in
*The 28th International Conference on Computational Linguistics (COLING 2020)*, 2020. - [89] I. Shode, D. I. Adelani, J. Peng, and A. Feldman, “Nollysenti: Leveraging transfer learning and machine translation for nigerian movie sentiment classification,” 2023.
-
[90]
S. H. Muhammad, D. I. Adelani, A. Anuoluwapo, and I. Abdulmumin, “Naijasenti: A nigerian twitter sentiment corpus for multilingual sentiment analysis,” in
*Proc. Thirteenth Language Resources and Evaluation Conference, LREC 2022, Marseille, France, 20-25 June 2022*, 2022, pp. 590–602. -
[91]
O. Ogundepo, X. Zhang, S. Sun, K. Duh, and J. Lin, “AfriCLIRMatrix: Enabling cross-lingual information retrieval for african languages,” in
*Proc. 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Dec. 2022. -
[92]
S. Sun and K. Duh, “CLIRMatrix: A massively large collection of bilingual and multilingual datasets for cross-lingual information retrieval,” in
*Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Online, Nov. 2020. - [93] C. Ma, A. ImaniGooghari, H. Ye, E. Asgari, and H. Schütze, “Taxi1500: A multilingual dataset for text classification in 1500 languages,” 2023.
-
[94]
P. Keung, Y. Lu, G. Szarvas, and N. A. Smith, “The multilingual Amazon reviews corpus,” in
*Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Online, Nov. 2020, pp. 4563–4568. -
[95]
G. Lample, A. Conneau, M. Ranzato, L. Denoyer, and H. Jégou, “Word translation without parallel data,” in
*6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings*, 2018. - [96] (2018) Wikipedia monolingual corpora. [Online]. Available: http://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/
-
[97]
C. Palen-Michel, J. Kim, and C. Lignos, “Multilingual open text release 1: Public domain news in 44 languages,” in
*Proc. Language Resources and Evaluation Conference*, Marseille, France, June 2022, pp. 2080–2089. -
[98]
P. Lison and J. Tiedemann, “OpenSubtitles2016: Extracting large parallel corpora from movie and TV subtitles,” in
*Proc. Tenth International Conference on Language Resources and Evaluation (LREC’16)*, Portorož, Slovenia, May 2016, pp. 923–929. -
[99]
W. Zhu, H. Liu, Q. Dong, J. Xu, L. Kong, J. Chen, L. Li, and S. Huang, “Multilingual machine translation with large language models: Empirical results and analysis,”
*arXiv preprint arXiv:2304.04675*, 2023. -
[100]
N. Goyal, J. Du, M. Ott, G. Anantharaman, and A. Conneau, “Larger-scale transformers for multilingual masked language modeling,”
*arXiv preprint arXiv:2105.00572*, 2021. -
[101]
P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, “Enriching word vectors with subword information,”
*Transactions of the association for computational linguistics*, vol. 5, pp. 135–146, 2017. -
[102]
X. V. Lin
*et al.*, “Few-shot learning with multilingual generative language models,” in*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, Y. Goldberg, Z. Kozareva, and Y. Zhang, Eds. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, Dec. 2022, pp. 9019–9052. [Online]. Available: https://aclanthology.org/2022.emnlp-main.616 -
[103]
M. Artetxe, G. Labaka, and E. Agirre, “A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings,” in
*Proc. 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2018, pp. 789–798. -
[104]
A. Søgaard, S. Ruder, and I. Vulić, “On the limitations of unsupervised bilingual dictionary induction,” in
*Proc. 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2018, pp. 778–788. -
[105]
N. Nakashole, “Norma: Neighborhood sensitive maps for multilingual word embeddings,” in
*Proc. 2018 Conference on Empirical Methods in Natural Language Processing*, 2018, pp. 512–522. -
[106]
H. Wang, J. Henderson, and P. Merlo, “Multi-adversarial learning for cross-lingual word embeddings,” in
*Proc. 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2021, pp. 463–472. -
[107]
J. Sarzynska-Wawer, A. Wawer, A. Pawlak, J. Szymanowska, I. Stefaniak, M. Jarkiewicz, and L. Okruszek, “Detecting formal thought disorder by deep contextualized word representations,”
*Psychiatry Research*, vol. 304, p. 114135, 2021. -
[108]
T. Schuster, O. Ram, R. Barzilay, and A. Globerson, “Cross-lingual alignment of contextual word embeddings, with applications to zero-shot dependency parsing,” in
*Proc. NAACL-HLT*, 2019, pp. 1599–1613. -
[109]
P. Gage, “A new algorithm for data compression,”
*C Users J.*, vol. 12, no. 2, p. 23–38, feb 1994. -
[110]
M. Schuster and K. Nakajima, “Japanese and korean voice search,” in
*2012 IEEE international conference on acoustics, speech and signal processing (ICASSP)*, 2012, pp. 5149–5152. -
[111]
I. Vulić, E. M. Ponti, R. Litschko, G. Glavaš, and A. Korhonen, “Probing pretrained language models for lexical semantics,” in
*Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Online, Nov. 2020, pp. 7222–7240. -
[112]
J. Zhang, B. Ji, N. Xiao, X. Duan, M. Zhang, Y. Shi, and W. Luo, “Combining static word embeddings and contextual representations for bilingual lexicon induction,” in
*Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, 2021, pp. 2943–2955. -
[113]
K. Hämmerl, J. Libovickỳ, and A. Fraser, “Combining static and contextualised multilingual embeddings,” in
*Findings of the Association for Computational Linguistics: ACL 2022*, 2022, pp. 2316–2329. -
[114]
J. Zheng, Y. Wang, G. Wang, J. Xia, Y. Huang, G. Zhao, Y. Zhang, and S. Li, “Using context-to-vector with graph retrofitting to improve word embeddings,” in
*Proc. 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2022, pp. 8154–8163. -
[115]
Y. Li, F. Liu, N. Collier, A. Korhonen, and I. Vulić, “Improving word translation via two-stage contrastive learning,” in
*Proc. 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, Dublin, Ireland, May 2022, pp. 4353–4374. -
[116]
D. Alvarez-Melis and T. Jaakkola, “Gromov-wasserstein alignment of word embedding spaces,” in
*Conference on Empirical Methods in Natural Language Processing*, 2018. -
[117]
S. Ren, S. Liu, M. Zhou, and S. Ma, “A graph-based coarse-to-fine method for unsupervised bilingual lexicon induction,” in
*Proc. 58th Annual Meeting of the Association for Computational Linguistics*, Online, Jul. 2020, pp. 3476–3485. -
[118]
T. Mohiuddin and S. Joty, “Revisiting adversarial autoencoder for unsupervised word translation with cycle consistency and improved training,” in
*Proc. 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, Minneapolis, Minnesota, Jun. 2019, pp. 3857–3867. -
[119]
T. Mohiuddin, M. S. Bari, and S. Joty, “LNMap: Departures from isomorphic assumption in bilingual lexicon induction through non-linear mapping in latent space,” in
*Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Online, Nov. 2020, pp. 2712–2723. -
[120]
G. Glavaš and I. Vulić, “Non-linear instance-based cross-lingual mapping for non-isomorphic embedding spaces,” in
*Proc. 58th Annual Meeting of the Association for Computational Linguistics*, Online, Jul. 2020, pp. 7548–7555. -
[121]
K. Marchisio, N. Verma, K. Duh, and P. Koehn, “Isovec: Controlling the relative isomorphism of word embedding spaces,” in
*Proc. 2022 Conference on Empirical Methods in Natural Language Processing*, 2022, pp. 6019–6033. -
[122]
J. Singh, B. McCann, R. Socher, and C. Xiong, “Bert is not an interlingua and the bias of tokenization,” in
*Proc. 2nd Workshop on Deep Learning Approaches for Low-Resource NLP (DeepLo 2019)*, 2019, pp. 47–55. -
[123]
H. Taitelbaum, G. Chechik, and J. Goldberger, “Multilingual word translation using auxiliary languages,” in
*Proc. 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, Hong Kong, China, Nov. 2019, pp. 1330–1335. -
[124]
K. Karthikeyan, Z. Wang, S. Mayhew, and D. Roth, “Cross-lingual ability of multilingual bert: An empirical study,” in
*International Conference on Learning Representations*, 2019. -
[125]
C.-L. Liu, T.-Y. Hsu, Y.-S. Chuang, and H.-Y. Lee, “A study of cross-lingual ability and language-specific information in multilingual bert,”
*arXiv preprint arXiv:2004.09205*, 2020. -
[126]
J. Ahn and A. Oh, “Mitigating language-dependent ethnic bias in BERT,” in
*Proc. 2021 Conference on Empirical Methods in Natural Language Processing*, Online and Punta Cana, Dominican Republic, 2021. -
[127]
N. Meade, E. Poole-Dayan, and S. Reddy, “An empirical survey of the effectiveness of debiasing techniques for pre-trained language models,” in
*Proc. 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022*, 2022, pp. 1878–1898. -
[128]
J. Zhao, S. Mukherjee, S. Hosseini, K. Chang, and A. H. Awadallah, “Gender bias in multilingual embeddings and cross-lingual transfer,” in
*Proc. 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020*, 2020, pp. 2896–2907. -
[129]
E. Ferrara, “Should chatgpt be biased? challenges and risks of bias in large language models,”
*CoRR*, vol. abs/2304.03738, 2023. -
[130]
S. Wu and M. Dredze, “Are all languages created equal in multilingual bert?” in
*Proc. 5th Workshop on Representation Learning for NLP, RepL4NLP@ACL 2020, Online, July 9, 2020*, 2020, pp. 120–130. -
[131]
J. Wang, Y. Liu, and X. E. Wang, “Assessing multilingual fairness in pre-trained multimodal representations,” in
*Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022*, 2022, pp. 2681–2695. -
[132]
N. Kassner, P. Dufter, and H. Schütze, “Multilingual lama: Investigating knowledge in multilingual pretrained language models,” in
*Proc. 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, 2021, pp. 3250–3258. -
[133]
S. Levy, N. A. John, L. Liu, Y. Vyas, J. Ma, Y. Fujinuma, M. Ballesteros, V. Castelli, and D. Roth, “Comparing biases and the impact of multilingual training across multiple languages,” in
*Proc. 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, 2023, pp. 10 260–10 280. -
[134]
L. C. Piqueras and A. Søgaard, “Are pretrained multilingual models equally fair across languages?” in
*Proc. 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022*, 2022, pp. 3597–3605. -
[135]
S. Touileb, L. Øvrelid, and E. Velldal, “Occupational biases in Norwegian and multilingual language models,” in
*Proc. 4th Workshop on Gender Bias in Natural Language Processing (GeBNLP)*, Seattle, Washington, Jul. 2022, pp. 200–211. -
[136]
T. Naous, M. J. Ryan, and W. Xu, “Having beer after prayer? measuring cultural bias in large language models,”
*CoRR*, vol. abs/2305.14456, 2023. -
[137]
A. Abid, M. Farooqi, and J. Zou, “Large language models associate muslims with violence,”
*Nature Machine Intelligence*, vol. 3, pp. 461–463, 06 2021. -
[138]
Y. T. Cao, Y. Pruksachatkun, K.-W. Chang, R. Gupta, V. Kumar, J. Dhamala, and A. Galstyan, “On the intrinsic and extrinsic fairness evaluation metrics for contextualized language representations,” in
*Proc. 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, Dublin, Ireland, May 2022, pp. 561–570. -
[139]
C. Leiter, P. Lertvittayakumjorn, M. Fomicheva, W. Zhao, Y. Gao, and S. Eger, “Towards explainable evaluation metrics for machine translation,”
*CoRR*, vol. abs/2306.13041, 2023. -
[140]
T. Sun, J. He, X. Qiu, and X. Huang, “Bertscore is unfair: On social bias in language model-based metrics for text generation,” in
*Proc. 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022*, 2022, pp. 3726–3739. -
[141]
T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “Bertscore: Evaluating text generation with BERT,” in
*8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*, 2020. -
[142]
T. Sellam, D. Das, and A. Parikh, “BLEURT: Learning robust metrics for text generation,” in
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, Online, Jul. 2020, pp. 7881–7892. -
[143]
W. Yuan, G. Neubig, and P. Liu, “Bartscore: Evaluating generated text as text generation,” in
*Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual*, 2021, pp. 27 263–27 277. -
[144]
R. Koo, M. Lee, V. Raheja, J. I. Park, Z. M. Kim, and D. Kang, “Benchmarking cognitive biases in large language models as evaluators,”
*CoRR*, vol. abs/2309.17012, 2023. -
[145]
P. Delobelle, E. Tokpo, T. Calders, and B. Berendt, “Measuring fairness with biased rulers: A comparative study on bias metrics for pre-trained language models,” in
*Proc. 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, Seattle, United States, Jul. 2022, pp. 1693–1706. -
[146]
R. Rudinger, J. Naradowsky, B. Leonard, and B. Van Durme, “Gender bias in coreference resolution,” in
*Proc. 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)*, New Orleans, Louisiana, Jun. 2018. -
[147]
J. Zhao, T. Wang, M. Yatskar, V. Ordonez, and K.-W. Chang, “Gender bias in coreference resolution: Evaluation and debiasing methods,” in
*Proc. 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)*, New Orleans, Louisiana, Jun. 2018. -
[148]
G. Stanovsky, N. A. Smith, and L. Zettlemoyer, “Evaluating gender bias in machine translation,” in
*Proc. 57th Annual Meeting of the Association for Computational Linguistics*, Florence, Italy, Jul. 2019. -
[149]
S. Kiritchenko and S. Mohammad, “Examining gender and race bias in two hundred sentiment analysis systems,” in
*Proc. Seventh Joint Conference on Lexical and Computational Semantics*, New Orleans, Louisiana, Jun. 2018. -
[150]
A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman, “GLUE: A multi-task benchmark and analysis platform for natural language understanding,” in
*Proc. 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP*, Brussels, Belgium, Nov. 2018. -
[151]
N. Nangia, C. Vania, R. Bhalerao, and S. R. Bowman, “CrowS-pairs: A challenge dataset for measuring social biases in masked language models,” in
*Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, Online, Nov. 2020. -
[152]
M. De-Arteaga, A. Romanov, H. M. Wallach, J. T. Chayes, C. Borgs, A. Chouldechova, S. C. Geyik, K. Kenthapadi, and A. T. Kalai, “Bias in bios: A case study of semantic representation bias in a high-stakes setting,” in
*Proc. Conference on Fairness, Accountability, and Transparency, FAT* 2019, Atlanta, GA, USA, January 29-31, 2019*, 2019, pp. 120–128. -
[153]
A. Caliskan, J. J. Bryson, and A. Narayanan, “Semantics derived automatically from language corpora contain human-like biases,”
*Science*, vol. 356, pp. 183–186, 2017. -
[154]
C. May, A. Wang, S. Bordia, S. R. Bowman, and R. Rudinger, “On measuring social biases in sentence encoders,” in
*Proc. 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, Minneapolis, Minnesota, Jun. 2019. -
[155]
W. Guo and A. Caliskan, “Detecting emergent intersectional biases: Contextualized word embeddings contain a distribution of human-like biases,” in
*AIES ’21: AAAI/ACM Conference on AI, Ethics, and Society, Virtual Event, USA, May 19-21, 2021*, 2021, pp. 122–133. -
[156]
A. Lauscher and G. Glavaš, “Are we consistently biased? multidimensional analysis of biases in distributional word vectors,” in
*Proc. Eighth Joint Conference on Lexical and Computational Semantics (*SEM 2019)*, Minneapolis, Minnesota, Jun. 2019, pp. 85–91. -
[157]
A. Névéol, Y. Dupont, J. Bezançon, and K. Fort, “French CrowS-pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than English,” in
*Proc. 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, Dublin, Ireland, May 2022, pp. 8521–8531. -
[158]
S. Bansal, V. Garimella, A. Suhane, and A. Mukherjee, “Debiasing multilingual word embeddings: A case study of three indian languages,” in
*HT ’21: 32nd ACM Conference on Hypertext and Social Media, Virtual Event, Ireland, 30 August 2021 - 2 September 2021*, 2021, pp. 27–34. -
[159]
K. Karkkainen and J. Joo, “Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation,” in
*Proc. IEEE/CVF Winter Conference on Applications of Computer Vision*, 2021, pp. 1548–1558. -
[160]
P. P. Liang, I. M. Li, E. Zheng, Y. C. Lim, R. Salakhutdinov, and L. Morency, “Towards debiasing sentence representations,” in
*Proc. 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5-10, 2020*, 2020, pp. 5502–5515. -
[161]
S. Ravfogel, Y. Elazar, H. Gonen, M. Twiton, and Y. Goldberg, “Null it out: Guarding protected attributes by iterative nullspace projection,” in
*Proc. 58th Annual Meeting of the Association for Computational Linguistics*, Online, Jul. 2020, pp. 7237–7256. -
[162]
Z. Yang, Y. Yang, D. Cer, and E. Darve, “A simple and effective method to eliminate the self language bias in multilingual representations,” in
*Proc. 2021 Conference on Empirical Methods in Natural Language Processing*, Online and Punta Cana, Dominican Republic, Nov. 2021. -
[163]
K. Webster, X. Wang, I. Tenney, A. Beutel, E. Pitler, E. Pavlick, J. Chen, and S. Petrov, “Measuring and reducing gendered correlations in pre-trained models,”
*CoRR*, vol. abs/2010.06032, 2020. -
[164]
N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,”
*J. Mach. Learn. Res.*, vol. 15, pp. 1929–1958, 2014. -
[165]
F. Zhou, Y. Mao, L. Yu, Y. Yang, and T. Zhong, “Causal-debias: Unifying debiasing in pretrained language models and fine-tuning via causal invariant learning,” in
*Proc. 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, Toronto, Canada, Jul. 2023, pp. 4227–4241. -
[166]
L. Ranaldi, E. S. Ruzzetti, D. Venditti, D. Onorati, and F. M. Zanzotto, “A trip towards fairness: Bias and de-biasing in large language models,”
*CoRR*, vol. abs/2305.13862, 2023. -
[167]
E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,”
*CoRR*, vol. abs/2106.09685, 2021. -
[168]
A. Wang and O. Russakovsky, “Overwriting pretrained bias with finetuning data,” in
*Proc. IEEE/CVF International Conference on Computer Vision*, 2023, pp. 3957–3968. -
[169]
Y. Guo, Y. Yang, and A. Abbasi, “Auto-debias: Debiasing masked language models with automated biased prompts,” in
*Proc. 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, Dublin, Ireland, May 2022, pp. 1012–1023. -
[170]
J. Mattern, Z. Jin, M. Sachan, R. Mihalcea, and B. Schölkopf, “Understanding stereotypes in language models: Towards robust measurement and zero-shot debiasing,”
*CoRR*, vol. abs/2212.10678, 2022. -
[171]
H. Dhingra, P. Jayashanker, S. Moghe, and E. Strubell, “Queer people are people first: Deconstructing sexual identity stereotypes in large language models,”
*CoRR*, vol. abs/2307.00101, 2023. -
[172]
T. Schick, S. Udupa, and H. Schütze, “Self-diagnosis and self-debiasing: A proposal for reducing corpus-based bias in nlp,”
*Transactions of the Association for Computational Linguistics*, p. 1408–1424, Dec 2021. -
[173]
A. Conneau, R. Rinott, G. Lample, A. Williams, S. Bowman, H. Schwenk, and V. Stoyanov, “XNLI: Evaluating cross-lingual sentence representations,” in
*Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, Eds. Brussels, Belgium: Association for Computational Linguistics, Oct.-Nov. 2018, pp. 2475–2485. [Online]. Available: https://aclanthology.org/D18-1269 - [174] T. Nguyen, C. V. Nguyen, V. D. Lai, H. Man, N. T. Ngo, F. Dernoncourt, R. A. Rossi, and T. H. Nguyen, “Culturax: A cleaned, enormous, and multilingual dataset for large language models in 167 languages,” 2023.
-
[175]
H. Laurençon
*et al.*, “The bigscience roots corpus: A 1.6 tb composite multilingual dataset,”*Advances in Neural Information Processing Systems*, vol. 35, pp. 31 809–31 826, 2022. -
[176]
J. Kreutzer
*et al.*, “Quality at a glance: An audit of web-crawled multilingual datasets,”*Transactions of the Association for Computational Linguistics*, vol. 10, pp. 50–72, 2022. [Online]. Available: https://aclanthology.org/2022.tacl-1.4 -
[177]
I. Sen, D. Assenmacher, M. Samory, I. Augenstein, W. Aalst, and C. Wagner, “People make better edits: Measuring the efficacy of LLM-generated counterfactually augmented data for harmful language detection,” in
*Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 10 480–10 504. [Online]. Available: https://aclanthology.org/2023.emnlp-main.649 -
[178]
J. Zhao, T. Wang, M. Yatskar, R. Cotterell, V. Ordonez, and K.-W. Chang, “Gender bias in contextualized word embeddings,” in
*Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, J. Burstein, C. Doran, and T. Solorio, Eds. Minneapolis, Minnesota: Association for Computational Linguistics, Jun. 2019, pp. 629–634. [Online]. Available: https://aclanthology.org/N19-1064 -
[179]
L. Yang, J. Li, P. Cunningham, Y. Zhang, B. Smyth, and R. Dong, “Exploring the efficacy of automatically generated counterfactuals for sentiment analysis,” in
*Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, C. Zong, F. Xia, W. Li, and R. Navigli, Eds. Online: Association for Computational Linguistics, Aug. 2021, pp. 306–316. [Online]. Available: https://aclanthology.org/2021.acl-long.26 -
[180]
I. Sen, M. Samory, F. Flöck, C. Wagner, and I. Augenstein, “How does counterfactually augmented data impact models for social computing constructs?” in
*Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, Eds. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, Nov. 2021, pp. 325–344. [Online]. Available: https://aclanthology.org/2021.emnlp-main.28 -
[181]
S. Goldfarb-Tarrant, A. Lopez, R. Blanco, and D. Marcheggiani, “Bias beyond English: Counterfactual tests for bias in sentiment analysis in four languages,” in
*Findings of the Association for Computational Linguistics: ACL 2023*, A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 4458–4468. [Online]. Available: https://aclanthology.org/2023.findings-acl.272 -
[182]
I. Sen, M. Samory, C. Wagner, and I. Augenstein, “Counterfactually augmented data and unintended bias: The case of sexism and hate speech detection,” in
*Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, Eds. Seattle, United States: Association for Computational Linguistics, Jul. 2022, pp. 4716–4726. [Online]. Available: https://aclanthology.org/2022.naacl-main.347 -
[183]
N. Joshi and H. He, “An investigation of the (in)effectiveness of counterfactually augmented data,” in
*Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 3668–3681. [Online]. Available: https://aclanthology.org/2022.acl-long.256 -
[184]
H. Yadav and S. Sitaram, “A survey of multilingual models for automatic speech recognition,” in
*Proceedings of the Thirteenth Language Resources and Evaluation Conference*, N. Calzolari*et al.*, Eds. Marseille, France: European Language Resources Association, Jun. 2022, pp. 5071–5079. [Online]. Available: https://aclanthology.org/2022.lrec-1.542 -
[185]
J. Hu, S. Ruder, A. Siddhant, G. Neubig, O. Firat, and M. Johnson, “Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalization,”
*CoRR*, vol. abs/2003.11080, 2020. -
[186]
P. Dufter and H. Schütze, “Identifying elements essential for BERT’s multilinguality,” in
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, B. Webber, T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, Nov. 2020, pp. 4423–4437. [Online]. Available: https://aclanthology.org/2020.emnlp-main.358 -
[187]
A. Nzeyimana and A. Niyongabo Rubungo, “KinyaBERT: a morphology-aware Kinyarwanda language model,” in
*Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 5347–5363. [Online]. Available: https://aclanthology.org/2022.acl-long.367 -
[188]
H. Naveed, A. U. Khan, S. Qiu, M. Saqib, S. Anwar, M. Usman, N. Barnes, and A. Mian, “A comprehensive overview of large language models,”
*arXiv preprint arXiv:2307.06435*, 2023. -
[189]
X. Pan, B. Zhang, J. May, J. Nothman, K. Knight, and H. Ji, “Cross-lingual name tagging and linking for 282 languages,” in
*Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, R. Barzilay and M.-Y. Kan, Eds. Vancouver, Canada: Association for Computational Linguistics, Jul. 2017, pp. 1946–1958. [Online]. Available: https://aclanthology.org/P17-1178 -
[190]
F. Liu, E. Bugliarello, E. M. Ponti, S. Reddy, N. Collier, and D. Elliott, “Visually grounded reasoning across languages and cultures,”
*arXiv preprint arXiv:2109.13238*, 2021.

| Yuemei Xu is an associate professor in the School of Information Science and Technology, Beijing Foreign Studies University. She received her PhD degree from Chinese Academy of Sciences in 2014, the B.E. from Beijing University of Posts and Telecommunications (China) in 2009. Her main research interests include Multilingual Natural Language Processing and Artificial Intelligence. |

| Ling Hu received the bachelor’s degree from Beijing University of Posts and Telecommunications (China) in 2021. She is currently pursing the master degree with the School of Information Science and Technology, Beijing Foreign Studies University. Her main research interests include Multilingual Natural Language Processing and Artificial Intelligence. |

| Jiayi Zhao is majoring in computer science and technology at the School of Information Science and Technology, Beijing Foreign Studies University. Her main research interests include Multilingual Natural Language Processing and Artificial Intelligence. |

| Zihan Qiu is majoring in computer science and technology at the School of Information Science and Technology, Beijing Foreign Studies University. Her main research interests include Multilingual Natural Language Processing and Artificial Intelligence. |

| Yuqi Ye is majoring in computer science and technology at the School of Information Science and Technology, Beijing Foreign Studies University. Her main research interests include Multilingual Natural Language Processing and Artificial Intelligence. |

| Hanwen Gu received the bachelor of Engineering degree from the School of Information Science at Beijing Language and Culture University in 2023. Currently, he is pursuing a master’s degree in the School of Information Science and Technology at Beijing Foreign Studies University. His primary research interests encompass natural language processing and artificial intelligence. |