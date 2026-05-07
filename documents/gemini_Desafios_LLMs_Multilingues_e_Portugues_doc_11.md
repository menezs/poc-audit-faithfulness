# Multilingual Large Language Models and Curse of Multilinguality

###### Abstract

Multilingual Large Language Models (LLMs) have gained large popularity among Natural Language Processing (NLP) researchers and practitioners. These models, trained on huge datasets, show proficiency across various languages and demonstrate effectiveness in numerous downstream tasks. This paper navigates the landscape of multilingual LLMs, providing an introductory overview of their technical aspects. It explains underlying architectures, objective functions, pre-training data sources, and tokenization methods. This work explores the unique features of different model types: encoder-only (mBERT, XLM-R), decoder-only (XGLM, PALM, BLOOM, GPT-3), and encoder-decoder models (mT5, mBART). Additionally, it addresses one of the significant limitations of multilingual LLMs - the curse of multilinguality - and discusses current attempts to overcome it.

Multilingual Large Language Models and Curse of Multilinguality

Daniil Gurgurov1,2 Tanja Bäumel2 Tatiana Anikina
2
1Universität des Saarlandes
2German Research Center for Artificial Intelligence (DFKI)
{daniil.gurgurov, tanja.baeumel, tatiana.anikina}@dfki.de

## 1 Introduction

Large Language Models (LLMs) Devlin et al. (2019); Lewis et al. (2020); Liu et al. (2019) have made a significant impact on the field of Natural Language Processing (NLP), showing effectiveness in various tasks. The remarkable aspect of LLMs is their capacity to learn a language during pre-training and enhance their expertise for specific tasks during fine-tuning. Pre-training involves the acquisition of knowledge, where a model grasps language structures by analyzing huge datasets. Fine-tuning, on the other hand, specializes the model by adjusting its parameters so that it can perform specific downstream tasks using a smaller set of examples compared to those used in pre-training.

Another significant advancement involves teaching a model to comprehend multiple languages, leading to the concept of multilingual LLMs Devlin et al. (2018); Pires et al. (2019); Conneau et al. (2020); Xue et al. (2021); Liu et al. (2020). While monolingual LLMs focus on understanding patterns within a single language, multilingual LLMs simultaneously learn from multiple languages. This is accomplished by exposing these models to data from various languages during the pre-training phase. Furthermore, variations in the architectures of multilingual LLMs contribute to their strengths in certain tasks while potentially limiting their effectiveness in others.

This paper aims to provide a brief overview of the architectures of the most prominent multilingual LLMs, including details such as their pre-training objective functions, data sources, tokenization schemas, the number of languages supported, and the peculiarities of each individual multilingual LLM. Subsequently, the primary challenge facing multilingual LLMs, known as the "curse of multilinguality" Conneau et al. (2020), and the current attempts to solve it, are discussed.

| Model | Architecture | Training Data Sources | Languages | Tokenization Schema |
|---|---|---|---|---|
| mBERT | Encoder | Wikipedia | 104 | WordPiece |
| XLM-R | Encoder | CC | 100 | SentencePiece |
| mBART | Encoder-Decoder | CC25 | 25 | SentencePiece |
| mT5 | Encoder-Decoder | C4 | 101 | SentencePiece |
| XGLM | Decoder | CC100-XL | 134 | SentencePiece |
| PALM | Decoder | Wikipedia, books, webpages, social media & source code | 124 | SentencePiece |
| BLOOM | Decoder | ROOTS, OSCAR | 46 | Byte Pair Encoding |
| GPT-3 | Decoder | CC, Wikipedia, WebText2, Books1&2 | >95 | Byte Pair Encoding |

## 2 Technicalities

This section delves into the technical details behind multilingual LLMs, providing a solid foundation for thorough conceptual understanding of their inner workings.

### 2.1 Architectures

LLMs, including multilingual LLMs, are typically constructed using the Transformer architecture Vaswani et al. (2017). These models can be categorized into three main architectural types: Encoder-only, Decoder-only, and Encoder-Decoder.

The original Transformer implementation was intended for machine translation purposes and was of the Encoder-Decoder type, since it included both an Encoder and a Decoder. The architecture is illustrated in figure 1 and described as follows.

The first stage of the architecture involves an Encoder. Initially, the input sentence is split into tokens, which are mapped to their vocabulary indices and converted into embeddings within the "Input embeddings" layer. Simultaneously, positional encoding vectors Gehring et al. (2017) are added to the original token embeddings to capture token position information. The embeddings then undergo a multi-head self-attention mechanism, allowing the model to focus on relevant information while generating the output and enhancing the ability to capture long-range dependencies. This is achieved by calculating attention scores based on the similarity between each token and all other tokens in the sequence. The attention step is followed by the "Add & Norm" layer Ba et al. (2016), serving as a normalization procedure and enhancing gradient flow by combining initial token embeddings to the multi-head attention output. Subsequently, the outputs from the previous step pass through a two-layer Feed Forward Neural Network (FFNN) Bebis and Georgiopoulos (1994), followed by another "Add & Norm" layer. FFNN refines and enriches the token representations coming from the self-attention mechanism, helping the model to capture both local and global contextual information. Finally, the representations are fed into a multi-head self-attention mechanism of the decoder.

The decoder follows a similar structure to the encoder but with some important differences. The input sequence is tokenized, converted into embeddings, and positionally encoded. The decoder’s self-attention is applied in two steps: masked self-attention and subsequent multi-head self-attention. The masked self-attention mechanism ensures that each position in the encoder can only attend to positions before it, which prevents information leakage from future tokens. After the first attention steps, "Add & norm" is applied, and outputs move to a decoder-encoder attention layer, enabling the encoder to attend to the decoder output. This mechanism facilitates the alignment of input and output sequences, helping with information capture for token generation. Then, "Add & Norm" is used, and token embeddings pass through another two-layer FFNN, followed by "Add & Norm". At this stage, the representations for output are generated, going through a final layer and producing a probability distribution over the vocabulary. The model outputs the token with the highest probability for each position. During training, the model is optimized using backpropagation and an optimization algorithm Dreyfus (1990).

#### 2.1.1 Encoder-Decoder

In the context of multilingual LLMs, the Encoder-Decoder architecture is utilized for tasks such as machine translation. An exemplary model employing this architecture is the mBART model Liu et al. (2020). mBART uses the Transformer architecture in an Encoder-Decoder style, where the encoder processes the input text, and the decoder generates the corresponding translated output. This architecture allows the model to effectively capture language-specific details during translation.

#### 2.1.2 Encoder-only

Multilingual Encoder-only architectures are designed for tasks that require understanding input sequences without the need for sequence generation. mBERT Devlin et al. (2018) is a prominent example of an Encoder-only model. Trained on a large set of languages, mBERT’s encoder is capable of extracting contextualized representations of input. This makes it valuable for various downstream tasks, such as sentiment analysis and named entity recognition, where understanding language context is important.

#### 2.1.3 Decoder-only

In specific multilingual cases, a Decoder-only architecture is beneficial. Models like XGLM Lin et al. (2022a) follow a Decoder-only configuration. This architecture is particularly useful for tasks where the focus is on generating sequences, such as language modeling or text completion. XGLM is a powerful decoder model that generates coherent sequences in various languages without the need for an encoder.

### 2.2 Objective Functions

In this section, the most popular objective functions used for pre-training multilingual LLMs are described: Masked Language Modeling (MLM) Devlin et al. (2019), Causal Language Modeling (CLM) Conneau and Lample (2019), Next Sentence Prediction (NSP) Devlin et al. (2019), and Translation Language Modeling (TLM) Conneau and Lample (2019). These pre-training objective functions serve as foundational tasks for training language models.

#### 2.2.1 MLM

Inspired by a Cloze task Taylor (1953), Masked Language Modeling involves randomly masking certain words in a sentence and training the model to predict these masked words based on the context provided by the surrounding words. This objective function teaches the model to understand the relationships and dependencies between different words in a sentence, promoting a deeper understanding of syntax, semantics, and contextual nuances within the text.

#### 2.2.2 CLM

Causal Language Modeling is designed for autoregressive models, whose focus is on the next token generation. It involves predicting the next word in a sequence given the preceding context. This pre-training objective is particularly effective for tasks where the order of the input sequence is important, as the model learns to capture dependencies and temporal relationships between words.

#### 2.2.3 NSP

Next Sentence Prediction trains models to predict whether a given pair of sentences is contiguous or not. The models learn to understand the coherence and logical flow between sentences. This pre-training objective is useful for tasks requiring comprehension of discourse and context, such as question-answering and document summarization.

#### 2.2.4 TLM

Translation Language Modeling extends the pre-training objective to involve parallel data in different languages. The sentences in different languages are concatenated, and words are randomly masked in each of them. Predicting the missing words in this case requires the model to understand the semantic relationships between sentences in different languages and helps in capturing cross-lingual representations. TLM is particularly beneficial for multilingual models, enabling them to transfer knowledge across languages.

### 2.3 Pre-training data

In the pre-training phase, multilingual LLMs leverage two primary types of data: large monolingual corpora in individual languages and parallel corpora across languages. The data for most models is taken from one of the following datasets: Wikipedia Vrandečić and Krötzsch (2014), Common Crawl (CC) Corpus Wenzek et al. (2020), ROOTS Laurençon et al. (2023), OSCAR Abadji et al. (2022), WebText2 Gao et al. (2020), Books1, and Books2 Brown et al. (2020). The choice of training corpora varies among multilingual LLMs. For instance, mBERT leverages Wikipedia as its training data, while XLM-R uses the more extensive CC corpus.

#### 2.3.1 Wikipedia

The Wikipedia dataset includes cleaned articles in various languages, sourced from Wikipedia dumps. Each language has a separate subset, with examples representing entire articles. The content is cleaned by removing markdown and extra sections.

#### 2.3.2 Common Crawl

The Common Crawl corpus is a vast collection of web data accumulated over more than 10 years through web crawling. It includes raw web page data, metadata extracts, and text extracts. This non-curated dataset includes web pages in numerous languages, providing a rich source for multilingual pre-training.

#### 2.3.3 ROOTS

The Responsible Open-science Open-collaboration Text Sources (ROOTS) corpus is a comprehensive dataset developed by the BigScience workshop, an international and multidisciplinary initiative focusing on researching and LLMs with an emphasis on ethics, harm, and governance. Spanning 1.6 terabytes, the ROOTS corpus serves as a foundational resource for training large-scale language models.

#### 2.3.4 OSCAR

The OSCAR project is an open-source initiative aimed at providing web-based multilingual data for ML and AI applications. It offers large quantities of raw data obtained through high-performance data pipelines. OSCAR focuses on improving data quality, particularly for low-resource languages.

#### 2.3.5 WebText2

OpenWebText2 is an improved version of the original OpenWebTextCorpus. It covers all Reddit posts from 2005 to April 2020, with additional months becoming available over time. This dataset is another contribution to the diverse training material for multilingual language models.

#### 2.3.6 Books1 and Books2

Books1 and Books2 comprise internet-based book corpora, featuring a random sampling of public domain books, as well as modern published literature in e-book format. The contents are drawn from available online books, offering a mix of historical and contemporary literature. It was used for pre-training GPT-3 Brown et al. (2020) and does not seem to be publicly available.

### 2.4 Languages

Multilingual LLMs demonstrate diversity in terms of the number of languages they support. Models like XLM-R show huge multilingual capabilities, facilitating approximately 100 languages. In contrast, models like mBART target a smaller set of languages, as shown in Table LABEL:table:multilingual. Managing the imbalance in pre-training data among languages, especially when dealing with a large number of them, is a challenge. For example, high-resource languages like English have significantly more data available compared to lower-resource languages like Maltese or Odia Joshi et al. (2020). To address these imbalances, multilingual LLMs often employ exponentially smoothed weighting Devlin et al. (2018). This approach ensures relatively fair representation of low-resource languages in the model’s training data, preventing them from being underrepresented in the overall vocabulary.

### 2.5 Tokenization Techniques

Tokenization is a critical part in language processing, and several techniques have been developed to represent words effectively. In this section, three widely used tokenization methods are introduced: Byte Pair Encoding (BPE), WordPiece, and SentencePiece.

#### 2.5.1 Byte Pair Encoding

BPE Sennrich et al. (2016) is a subword tokenization technique that builds a vocabulary by iteratively merging the most frequent pairs of consecutive bytes. This method is particularly effective in representing both common words and rare subword units in a wide range of languages and character sets.

#### 2.5.2 WordPiece

WordPiece Schuster and Nakajima (2012) is another subword tokenization algorithm that starts with a vocabulary of individual characters and iteratively merges the most frequent character pairs. This process creates a vocabulary of subword units, allowing the model to represent words as combinations of subword tokens.

#### 2.5.3 SentencePiece

SentencePiece Kudo and Richardson (2018) is a text tokenizer that works at the subword level. Unlike the previous tokenizers, it takes the input sequence as a raw stream, including the space in the collection of characters. It employs a unigram language model to tokenize text into pieces, making it suitable for various languages. It was introduced as a solution for the problem posed by languages that do not use spaces to separate words.

## 3 Multilingual Large Language Models

This section presents prominent examples of multilingual LLMs categorized into three types: Encoder-Only, Decoder-Only, and Encoder-Decoder, along with their intricacies. Table LABEL:table:multilingual summarizes the important points regarding all presented models.

### 3.1 Encoder-Only Models

### 3.2 mBERT

The first notable example of a genuinely multilingual neural LLM is the multilingual Bidirectional Encoder Representations from Transformers (mBERT) Devlin et al. (2019); Wenzek et al. (2020). This model adapts the architecture similar to the original BERT Devlin et al. (2019), consisting of 12 Transformer layers and operating as an encoder-only model. The distinguishing factor lies in the model being trained on the entire Wikipedia data covering 104 languages Pires et al. (2019), in contrast to the original BERT, which was trained only on English. Additionally, mBERT employs a shared vocabulary for all languages based on the WordPiece tokenization schema. Pre-training mBERT involves two unsupervised tasks: MLM and NSP.

### 3.3 XLM-R

Cross-lingual Language Modeling RoBERTa (XLM-R) Conneau et al. (2020) is a 12-layers Transformer-based multilingual model that adopts an encoder-only architecture following the XLM Conneau and Lample (2019) approach. It integrates RoBERTa, introducing simple improvements to the learning procedure. Unlike the original XLM, which employed both MLM and TLM objectives for pre-training, XLM-R primarily utilizes the MLM objective. XLM-R is available in two configurations - XLM-R Base (270M parameters) and XLM-R XL (550M parameters). Both variants are trained on the clean CommonCrawl corpus covering 100 languages. The tokenization method utilized by the model is SentencePiece.

### 3.4 Decoder-Only Models

#### 3.4.1 XGLM

XGLM Lin et al. (2022b), or Cross-lingual Generative Language Model, is a family of multilingual models trained on the extensive CC100-XL dataset, this is CommonCrawl snapshots including data from 2013 to 2020. The models vary in size and configuration, with parameters ranging from 564 million to 7.5 billion, language coverage from 30 to 134, and number of layers from 24 to 48. They utilize the SentencePiece tokenizer and focus on CLM during pre-training, aiming to predict the next token given the previous ones. The largest XGLM model, with 7.5 billion parameters, achieved state-of-the-art performance in few-shot learning across multiple languages at the time of the release, surpassing GPT-3 in tasks like commonsense reasoning and natural language inference.

#### 3.4.2 PALM

PaLM Chowdhery et al. (2023), or Pathways Language Model, is a multilingual model trained on a diverse corpus consisting of content from Wikipedia, books, web pages, social media, and source code. While exact details on the training data are not provided, it is stated that PaLM is trained on 124 languages using the SentencePiece tokenizer. Pre-training is done using Pathways, a system enabling highly efficient pre-training across multiple TPUs. The objective function used for pre-training is autoregressive language modeling (CLM). The model comes in a few configurations: 8B parameters and 32 Transformer layers, 62B parameters and 64 Transformer layers, and 540B parameters and 118 transformer layers. The largest variant, PaLM 540B, achieves competitive results across various tasks, despite being trained on approximately 22% of non-English data out of the total 780 billion training tokens.

#### 3.4.3 BLOOM

BLOOM Scao et al. (2023), standing for Bloom Language Model, is a significant step towards open-sourcing LLMs. With a capacity of 176 billion parameters and 70 Transformer layers, BLOOM is an open-access LLM developed through collaboration among hundreds of researchers. BLOOM is a decoder-only Transformer language model trained on the ROOTS corpus, which consists of data from hundreds of sources, including OSCAR, covering 46 natural languages and 13 programming languages. Similar to other models, BLOOM utilizes CLM during training and uses BPE tokenization. Despite being open-source, BLOOM demonstrates competitive performance across a wide range of benchmarks. Moreover, its performance can be further enhanced through multitask prompted fine-tuning. To support future research and applications using LLMs, the models and code for BLOOM are publicly released, which promotes transparency and collaboration in the field of NLP.

#### 3.4.4 GPT-3

GPT-3 Brown et al. (2020), short for Generative Pre-trained Transformer 3, is a widely known LLM developed by OpenAI. It is a decoder-only Transformer architecture, consisting of 175 billion parameters and 96 Transformer layers. GPT-3 is trained using the CLM task on a diverse dataset covering Common Crawl (CC), Wikipedia, WebText2, and Books1 & 2. These sources provide a wide range of textual data, enabling GPT-3 to understand and generate human-like text across various domains. The model covers over 95 languages, which is not explicitly stated in the original report and estimated by users. For tokenization, GPT-3 employs BPE. Additionally, GPT-3 incorporates advanced techniques such as few-shot learning, enabling it to perform new tasks with minimal instructions. Despite its impressive performance and wide adoption, GPT-3 is not openly accessible for training or fine-tuning by external researchers. However, its capabilities have captured significant interest, serving as a benchmark for the development of new language models.

### 3.5 Encoder-Decoder Models

### 3.6 mBART

The next model is the multilingual Bidirectional Autoregressive Transformer (mBART) - a multilingual sequence-to-sequence (Seq2Seq) denoising model Liu et al. (2020). It stands out as one of the pioneering models employing both an encoder and a decoder. The model is trained by applying the BART objectives Lewis et al. (2020) on extensive monolingual corpora across a variety of languages. This Seq2Seq model is pre-trained by denoising full texts in various languages, and it is primarily intended for translation tasks. During pre-training, the model tackles noise in the input by masking phrases and introducing sentence order permutations, which is a variant of MLM. Consequently, it learns to reconstruct the masked words based on context, developing an understanding of word relationships, and to correctly order sentences, thereby capturing relationships between sentences. A single Transformer model with 12 encoder and 12 decoder layers consisting of 680 million parameters is used to execute training. The pre-training data is the CC25 dataset extracted from the Common Crawl, which comprises of 25 languages and is tokenized using the SentencePiece schema.

### 3.7 mT5

The Multilingual Text-to-text Transfer Transformer, mT5 Xue et al. (2021), is the model that leverages the T5 architecture Raffel et al. (2020). Its training data comes from Multilingual C4 (mC4) dataset, comprising data in 101 languages sampled from Common Crawl and tokenized using the SentencePiece schema. An advantageous feature of the T5 architecture lies in its unified "Seq2seq" format, where the model generates text conditioned on given text inputs. Employing this "text-to-text" structure, mT5 uses a standard Encoder-Decoder Transformer architecture. For pre-training, a variation of MLM, involving the replacement of consecutive spans of input text with a masked token and predicting them, is used. Notably, mT5 exists in several variants, distinguished by the quantity of data used for training and the number of parameters in the model. These variants include mT5-Small, mT5-Base, mT5-Large, mT5-XL, and mT5-XXL. The number of parameters ranges from 300 million to 13 billion and number of layers - from 12 to 24 in both encoder and decoder.

## 4 Curse of Multilinguality

This chapter addresses one of the biggest challenges faced by multilingual LLMs - curse of multilinguality - and explores its potential solutions to overcome it.

### 4.1 Motivation

The curse of multilinguality was introduced by Conneau et al. and refers to the challenges and limitations that arise when developing multilingual LLMs Conneau et al. (2020). As the number of languages increases, a transfer-dilution trade-off occurs, diminishing the per-language capacity and consequently impacting model performance. Initially, adding similar higher-resource languages during pre-training can enhance the performance of low-resource languages. However, beyond a certain point, the curse of multilinguality starts to manifest, leading to decreased performance across all languages Aharoni et al. (2019). This emphasises the complexity of achieving universality in multilingual models and points at the need to address key factors contributing to the curse of multilinguality.

#### 4.1.1 Linguistic Diversity

Languages show significant diversity in grammar, syntax, vocabulary, and cultural nuances, posing a significant challenge for multilingual model development Conneau et al. (2020). Accommodating such diversity while maintaining model efficiency and performance becomes increasingly challenging with multiple languages, worsening the effects of the curse of multilinguality.

#### 4.1.2 Data Sparsity

The availability of training data varies widely across languages, with some languages possessing a lot of resources while others being severely under-resourced Joshi et al. (2020). The impact of data sparsity on training efficient multilingual models is large, as highlighted by Conneau et al. Conneau et al. (2020). Models may struggle to generalize well to languages with limited training data, further complicating the curse of multilinguality.

#### 4.1.3 Model Complexity and Scale

Multilingual models require increased complexity and scale to accommodate the diverse linguistic characteristics of multiple languages Conneau et al. (2020). Balancing model size and computational resources is crucial, as overly complex models may become impractical for real-world deployment. Conneau et al. emphasize the importance of addressing these challenges to mitigate the curse of multilinguality.

### 4.2 Potential Solutions

Recent studies have pointed at reduced monolingual and cross-lingual capabilities of models due to curse of multilinguality, particularly for low-resource languages not having a lot of pre-training data Wu and Dredze (2020); Lauscher et al. (2020); Artetxe et al. (2020). Various approaches have been proposed to address this problem and improve model performance.

#### 4.2.1 Modular Multilingual Architecture

Pfeiffer et al. propose X-MOD, a modular multilingual architecture that combines shared and language-specific parameters as a way of uplifting curse of multilinguality Pfeiffer et al. (2022). X-MOD initializes modular models during pre-training, facilitating inexpensive expansion to new languages afterwards.

##### Architecture:

X-MOD is an extension of the transformer-based architecture from mBERT and XLM-R, incorporating language-specific modules at every transformer layer. Each language has its own module, while attention and feed-forward components are shared, facilitating efficient training and inference without significantly increasing computational costs.

##### Pre-training:

X-MOD is pre-trained using MLM on combined monolingual corpora in multiple languages. Efficient utilization of language-specific modules ensures effective handling of linguistic diversity during pre-training.

##### Extending to New Languages:

The modular design allows X-MOD to be extended to new languages after pre-training, with minimal impact on performance in pre-trained languages. This adaptability is achieved through the learning of new embeddings and adapter modules for the target language via MLM.

##### Fine-tuning on Downstream Tasks:

X-MOD can be fine-tuned for cross-lingual downstream tasks by selectively updating shared weights on source language data while keeping modular components frozen, ensuring efficient adaptation to target languages.

#### 4.2.2 Alternative Approaches

Several alternative approaches have been proposed to extend multilingual and monolingual LLMs to other languages and improve their cross-lingual capabilities as a mitigation of curse of dimensionality.

##### Training New Embedding Layer:

Artetxe et al. propose training a new embedding layer with a corresponding target-language tokenizer to extend monolingual models to new languages Artetxe et al. (2020). This approach helps with language extension while maintaining model stability.

##### Transliteration and Subword Mappings:

##### Adapter-based Approaches:

Adapter-based approaches, proposed by Pfeiffer et al., offer efficient solutions for adapting multilingual LLMs to specific languages or extending multilingual them to unseen languages and overcoming curse of dimensionality Pfeiffer et al. (2020). While achieving significant performance gains, these approaches may build upon sub-optimal parameter initializations.

## 5 Conclusion

In summary, multilingual LLMs stand out as robust tools in NLP, showcasing proficiency across multiple languages and tasks. This paper has provided insights into key models such as mBERT, XLM-R, mBART, mT5, XGLM, PALM, BLOOM, and GPT-3, highlighting their underlying architectures and technical characteristics. These models represent a significant advancement in NLP, having deep multilingual understanding and potential for cross-lingual applications. Their development plays a great role in advancing language technology and preserving linguistic diversity across the world.

Despite their capabilities, multilingual LLMs face challenges such as the curse of multilinguality, which constrains their performance across different languages. However, various approaches have been developed and are being pursued to mitigate these challenges. These efforts aim to improve the effectiveness and adaptability of multilingual LLMs.

## References

-
Abadji et al. (2022)
Julien Abadji, Pedro Ortiz Suarez, Laurent Romary, and Benoît Sagot. 2022.
Towards a cleaner document-oriented multilingual crawled corpus.
In
*Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 4344–4355, Marseille, France. European Language Resources Association. -
Aharoni et al. (2019)
Roee Aharoni, Melvin Johnson, and Orhan Firat. 2019.
Massively multilingual neural machine translation.
In
*Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 3874–3884, Minneapolis, Minnesota. Association for Computational Linguistics. -
Artetxe et al. (2020)
Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020.
On the cross-lingual transferability of monolingual representations.
In
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 4623–4637, Online. Association for Computational Linguistics. -
Ba et al. (2016)
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016.
Layer normalization.
*arXiv preprint arXiv:1607.06450*. -
Bebis and Georgiopoulos (1994)
G. Bebis and M. Georgiopoulos. 1994.
Feed-forward neural networks.
*IEEE Potentials*, 13(4):27–31. -
Brown et al. (2020)
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020.
Language models are few-shot learners.
In
*Advances in Neural Information Processing Systems*, volume 33, page 1877–1901. Curran Associates, Inc. -
Chowdhery et al. (2023)
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2023.
Palm: Scaling language modeling with pathways.
*Journal of Machine Learning Research*, 24(240):1–113. -
Conneau et al. (2020)
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020.
Unsupervised cross-lingual representation learning at scale.
In
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 8440–8451, Online. Association for Computational Linguistics. -
Conneau and Lample (2019)
Alexis Conneau and Guillaume Lample. 2019.
Cross-lingual language model pretraining.
In
*Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc. - Devlin et al. (2018) Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Multilingual bert. https://github.com/google-research/bert/blob/master/multilingual.md.
-
Devlin et al. (2019)
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.
BERT: Pre-training of deep bidirectional transformers for language understanding.
In
*Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. -
Dreyfus (1990)
Stuart E Dreyfus. 1990.
Artificial neural networks, back propagation, and the kelley-bryson gradient procedure.
*Journal of guidance, control, and dynamics*, 13(5):926–928. - Gao et al. (2020) Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. The pile: An 800gb dataset of diverse text for language modeling.
-
Gehring et al. (2017)
Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin. 2017.
Convolutional sequence to sequence learning.
In
*International conference on machine learning*, pages 1243–1252. PMLR. -
Joshi et al. (2020)
Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020.
The state and fate of linguistic diversity and inclusion in the NLP world.
In
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 6282–6293, Online. Association for Computational Linguistics. -
Kudo and Richardson (2018)
Taku Kudo and John Richardson. 2018.
SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing.
In
*Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pages 66–71, Brussels, Belgium. Association for Computational Linguistics. - Laurençon et al. (2023) Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro Von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, Jörg Frohberg, Mario Šaško, Quentin Lhoest, Angelina McMillan-Major, Gerard Dupont, Stella Biderman, Anna Rogers, Loubna Ben allal, Francesco De Toni, Giada Pistilli, Olivier Nguyen, Somaieh Nikpoor, Maraim Masoud, Pierre Colombo, Javier de la Rosa, Paulo Villegas, Tristan Thrush, Shayne Longpre, Sebastian Nagel, Leon Weber, Manuel Muñoz, Jian Zhu, Daniel Van Strien, Zaid Alyafeai, Khalid Almubarak, Minh Chien Vu, Itziar Gonzalez-Dios, Aitor Soroa, Kyle Lo, Manan Dey, Pedro Ortiz Suarez, Aaron Gokaslan, Shamik Bose, David Adelani, Long Phan, Hieu Tran, Ian Yu, Suhas Pai, Jenny Chim, Violette Lepercq, Suzana Ilic, Margaret Mitchell, Sasha Alexandra Luccioni, and Yacine Jernite. 2023. The bigscience roots corpus: A 1.6tb composite multilingual dataset.
-
Lauscher et al. (2020)
Anne Lauscher, Vinit Ravishankar, Ivan Vulić, and Goran Glavaš. 2020.
From zero to hero: On the limitations of zero-shot language transfer with multilingual Transformers.
In
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 4483–4499, Online. Association for Computational Linguistics. -
Lewis et al. (2020)
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020.
BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.
In
*Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pages 7871–7880, Online. Association for Computational Linguistics. -
Lin et al. (2022a)
Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, and Xian Li. 2022a.
Few-shot learning with multilingual generative language models.
In
*Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 9019–9052, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. - Lin et al. (2022b) Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, and Xian Li. 2022b. Few-shot learning with multilingual language models.
-
Liu et al. (2020)
Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020.
Multilingual denoising pre-training for neural machine translation.
*Transactions of the Association for Computational Linguistics*, 8:726–742. - Liu et al. (2019) Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach.
-
Muller et al. (2021)
Benjamin Muller, Antonios Anastasopoulos, Benoît Sagot, and Djamé Seddah. 2021.
When being unseen from mBERT is just the beginning: Handling new languages with multilingual language models.
In
*Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 448–462, Online. Association for Computational Linguistics. -
Pfeiffer et al. (2022)
Jonas Pfeiffer, Naman Goyal, Xi Lin, Xian Li, James Cross, Sebastian Riedel, and Mikel Artetxe. 2022.
Lifting the curse of multilinguality by pre-training modular transformers.
In
*Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 3479–3495, Seattle, United States. Association for Computational Linguistics. -
Pfeiffer et al. (2020)
Jonas Pfeiffer, Ivan Vulić, Iryna Gurevych, and Sebastian Ruder. 2020.
MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer.
In
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 7654–7673, Online. Association for Computational Linguistics. -
Pires et al. (2019)
Telmo Pires, Eva Schlinger, and Dan Garrette. 2019.
How multilingual is multilingual BERT?
In
*Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 4996–5001, Florence, Italy. Association for Computational Linguistics. -
Raffel et al. (2020)
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020.
Exploring the limits of transfer learning with a unified text-to-text transformer.
*Journal of Machine Learning Research*, 21(140):1–67. - Scao et al. (2023) BigScience Workshop: Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, Dragomir Radev, Eduardo González Ponferrada, Efrat Levkovizh, Ethan Kim, Eyal Bar Natan, Francesco De Toni, Gérard Dupont, Germán Kruszewski, Giada Pistilli, Hady Elsahar, Hamza Benyamina, Hieu Tran, Ian Yu, Idris Abdulmumin, Isaac Johnson, Itziar Gonzalez-Dios, Javier de la Rosa, Jenny Chim, Jesse Dodge, Jian Zhu, Jonathan Chang, Jörg Frohberg, Joseph Tobing, Joydeep Bhattacharjee, Khalid Almubarak, Kimbo Chen, Kyle Lo, Leandro Von Werra, Leon Weber, Long Phan, Loubna Ben allal, Ludovic Tanguy, Manan Dey, Manuel Romero Muñoz, Maraim Masoud, María Grandury, Mario Šaško, Max Huang, Maximin Coavoux, Mayank Singh, Mike Tian-Jian Jiang, Minh Chien Vu, Mohammad A. Jauhar, Mustafa Ghaleb, Nishant Subramani, Nora Kassner, Nurulaqilla Khamis, Olivier Nguyen, Omar Espejel, Ona de Gibert, Paulo Villegas, Peter Henderson, Pierre Colombo, Priscilla Amuok, Quentin Lhoest, Rheza Harliman, Rishi Bommasani, Roberto Luis López, Rui Ribeiro, Salomey Osei, Sampo Pyysalo, Sebastian Nagel, Shamik Bose, Shamsuddeen Hassan Muhammad, Shanya Sharma, Shayne Longpre, Somaieh Nikpoor, Stanislav Silberberg, Suhas Pai, Sydney Zink, Tiago Timponi Torrent, Timo Schick, Tristan Thrush, Valentin Danchev, Vassilina Nikoulina, Veronika Laippala, Violette Lepercq, Vrinda Prabhu, Zaid Alyafeai, Zeerak Talat, Arun Raja, Benjamin Heinzerling, Chenglei Si, Davut Emre Taşar, Elizabeth Salesky, Sabrina J. Mielke, Wilson Y. Lee, Abheesht Sharma, Andrea Santilli, Antoine Chaffin, Arnaud Stiegler, Debajyoti Datta, Eliza Szczechla, Gunjan Chhablani, Han Wang, Harshit Pandey, Hendrik Strobelt, Jason Alan Fries, Jos Rozen, Leo Gao, Lintang Sutawika, M Saiful Bari, Maged S. Al-shaibani, Matteo Manica, Nihal Nayak, Ryan Teehan, Samuel Albanie, Sheng Shen, Srulik Ben-David, Stephen H. Bach, Taewoon Kim, Tali Bers, Thibault Fevry, Trishala Neeraj, Urmish Thakker, Vikas Raunak, Xiangru Tang, Zheng-Xin Yong, Zhiqing Sun, Shaked Brody, Yallow Uri, Hadar Tojarieh, Adam Roberts, Hyung Won Chung, Jaesung Tae, Jason Phang, Ofir Press, Conglong Li, Deepak Narayanan, Hatim Bourfoune, Jared Casper, Jeff Rasley, Max Ryabinin, Mayank Mishra, Minjia Zhang, Mohammad Shoeybi, Myriam Peyrounette, Nicolas Patry, Nouamane Tazi, Omar Sanseviero, Patrick von Platen, Pierre Cornette, Pierre François Lavallée, Rémi Lacroix, Samyam Rajbhandari, Sanchit Gandhi, Shaden Smith, Stéphane Requena, Suraj Patil, Tim Dettmers, Ahmed Baruwa, Amanpreet Singh, Anastasia Cheveleva, Anne-Laure Ligozat, Arjun Subramonian, Aurélie Névéol, Charles Lovering, Dan Garrette, Deepak Tunuguntla, Ehud Reiter, Ekaterina Taktasheva, Ekaterina Voloshina, Eli Bogdanov, Genta Indra Winata, Hailey Schoelkopf, Jan-Christoph Kalo, Jekaterina Novikova, Jessica Zosa Forde, Jordan Clive, Jungo Kasai, Ken Kawamura, Liam Hazan, Marine Carpuat, Miruna Clinciu, Najoung Kim, Newton Cheng, Oleg Serikov, Omer Antverg, Oskar van der Wal, Rui Zhang, Ruochen Zhang, Sebastian Gehrmann, Shachar Mirkin, Shani Pais, Tatiana Shavrina, Thomas Scialom, Tian Yun, Tomasz Limisiewicz, Verena Rieser, Vitaly Protasov, Vladislav Mikhailov, Yada Pruksachatkun, Yonatan Belinkov, Zachary Bamberger, Zdeněk Kasner, Alice Rueda, Amanda Pestana, Amir Feizpour, Ammar Khan, Amy Faranak, Ana Santos, Anthony Hevia, Antigona Unldreaj, Arash Aghagol, Arezoo Abdollahi, Aycha Tammour, Azadeh HajiHosseini, Bahareh Behroozi, Benjamin Ajibade, Bharat Saxena, Carlos Muñoz Ferrandis, Daniel McDuff, Danish Contractor, David Lansky, Davis David, Douwe Kiela, Duong A. Nguyen, Edward Tan, Emi Baylor, Ezinwanne Ozoani, Fatima Mirza, Frankline Ononiwu, Habib Rezanejad, Hessie Jones, Indrani Bhattacharya, Irene Solaiman, Irina Sedenko, Isar Nejadgholi, Jesse Passmore, Josh Seltzer, Julio Bonis Sanz, Livia Dutra, Mairon Samagaio, Maraim Elbadri, Margot Mieskes, Marissa Gerchick, Martha Akinlolu, Michael McKenna, Mike Qiu, Muhammed Ghauri, Mykola Burynok, Nafis Abrar, Nazneen Rajani, Nour Elkott, Nour Fahmy, Olanrewaju Samuel, Ran An, Rasmus Kromann, Ryan Hao, Samira Alizadeh, Sarmad Shubber, Silas Wang, Sourav Roy, Sylvain Viguier, Thanh Le, Tobi Oyebade, Trieu Le, Yoyo Yang, Zach Nguyen, Abhinav Ramesh Kashyap, Alfredo Palasciano, Alison Callahan, Anima Shukla, Antonio Miranda-Escalada, Ayush Singh, Benjamin Beilharz, Bo Wang, Caio Brito, Chenxi Zhou, Chirag Jain, Chuxin Xu, Clémentine Fourrier, Daniel León Periñán, Daniel Molano, Dian Yu, Enrique Manjavacas, Fabio Barth, Florian Fuhrimann, Gabriel Altay, Giyaseddin Bayrak, Gully Burns, Helena U. Vrabec, Imane Bello, Ishani Dash, Jihyun Kang, John Giorgi, Jonas Golde, Jose David Posada, Karthik Rangasai Sivaraman, Lokesh Bulchandani, Lu Liu, Luisa Shinzato, Madeleine Hahn de Bykhovetz, Maiko Takeuchi, Marc Pàmies, Maria A Castillo, Marianna Nezhurina, Mario Sänger, Matthias Samwald, Michael Cullan, Michael Weinberg, Michiel De Wolf, Mina Mihaljcic, Minna Liu, Moritz Freidank, Myungsun Kang, Natasha Seelam, Nathan Dahlberg, Nicholas Michio Broad, Nikolaus Muellner, Pascale Fung, Patrick Haller, Ramya Chandrasekhar, Renata Eisenberg, Robert Martin, Rodrigo Canalli, Rosaline Su, Ruisi Su, Samuel Cahyawijaya, Samuele Garda, Shlok S Deshmukh, Shubhanshu Mishra, Sid Kiblawi, Simon Ott, Sinee Sang-aroonsiri, Srishti Kumar, Stefan Schweter, Sushil Bharati, Tanmay Laud, Théo Gigant, Tomoya Kainuma, Wojciech Kusa, Yanis Labrak, Yash Shailesh Bajaj, Yash Venkatraman, Yifan Xu, Yingxin Xu, Yu Xu, Zhe Tan, Zhongli Xie, Zifan Ye, Mathilde Bras, Younes Belkada, and Thomas Wolf. 2023. Bloom: A 176b-parameter open-access multilingual language model.
-
Schuster and Nakajima (2012)
Mike Schuster and Kaisuke Nakajima. 2012.
Japanese and korean voice search.
In
*2012 IEEE international conference on acoustics, speech and signal processing (ICASSP)*, pages 5149–5152. IEEE. -
Sennrich et al. (2016)
Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016.
Neural machine translation of rare words with subword units.
In
*Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1715–1725, Berlin, Germany. Association for Computational Linguistics. -
Taylor (1953)
Wilson L Taylor. 1953.
“cloze procedure”: A new tool for measuring readability.
*Journalism quarterly*, 30(4):415–433. -
Vaswani et al. (2017)
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017.
Attention is all you need.
In
*Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc. -
Vernikos and Popescu-Belis (2021)
Giorgos Vernikos and Andrei Popescu-Belis. 2021.
Subword mapping and anchoring across languages.
In
*Findings of the Association for Computational Linguistics: EMNLP 2021*, pages 2633–2647, Punta Cana, Dominican Republic. Association for Computational Linguistics. -
Vrandečić and Krötzsch (2014)
Denny Vrandečić and Markus Krötzsch. 2014.
Wikidata: a free collaborative knowledgebase.
*Communications of the ACM*, 57(10):78–85. -
Wenzek et al. (2020)
Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. 2020.
CCNet: Extracting high quality monolingual datasets from web crawl data.
In
*Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 4003–4012, Marseille, France. European Language Resources Association. -
Wu and Dredze (2020)
Shijie Wu and Mark Dredze. 2020.
Are all languages created equal in multilingual BERT?
In
*Proceedings of the 5th Workshop on Representation Learning for NLP*, pages 120–130, Online. Association for Computational Linguistics. -
Xue et al. (2021)
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021.
mT5: A massively multilingual pre-trained text-to-text transformer.
In
*Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 483–498, Online. Association for Computational Linguistics.