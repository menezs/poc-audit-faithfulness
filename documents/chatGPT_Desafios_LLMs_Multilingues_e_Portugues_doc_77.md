# **MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages** 

**Jack FitzGerald** _[∗]_[1] **Christopher Hench**[1] **Charith Peris**[1] **Scott Mackie**[1] **Kay Rottmann**[1] **Ana Sanchez**[1] **Aaron Nash**[1] **Liam Urbach**[1] **Vishesh Kakarala**[1] **Richa Singh**[1] **Swetha Ranganath**[2] **Laurie Crist**[3] **Misha Britan**[1] **Wouter Leeuwis**[1] **Gokhan Tur**[1] **Prem Natarajan**[4] 

1Amazon 2Microsoft 3Tripadvisor 4Capital One 

## **Abstract** 

We present the MASSIVE dataset— Multilingual Amazon Slu resource package (SLURP) for Slot-filling, Intent classification, and Virtual assistant Evaluation. MASSIVE contains 1M realistic, parallel, labeled virtual assistant utterances spanning 51 languages, 18 domains, 60 intents, and 55 slots. MASSIVE was created by tasking professional translators to localize the English-only SLURP dataset into 50 typologically diverse languages from 29 genera. We also present modeling results on XLM-R and mT5, including exact match accuracy, intent classification accuracy, and slot-filling F1 score. We have released our dataset, modeling code, and models publicly. 

## **1 Introduction and Description** 

Natural Language Understanding (NLU) is a machine’s ability to understand the meaning and relevant entities from text. For instance, given the utterance what is the temperature in new york, an NLU model might classify the intent as weather_query and fill the slots as weather_descriptor: temperature and place_name: new york. Our particular focus of NLU is one component of Spoken Language Understanding (SLU), in which raw audio is first converted to text before NLU is performed (Young, 2002; Wang et al., 2005; Tur and Mori, 2011). SLU is the foundation of voice-based virtual assistants like Alexa, Siri, and Google Assistant. Though virtual assistants have advanced incredibly in the past decade, they still only support a small fraction of the world’s 7,000+ languages (Simons, 2022). Challenges 

*Corresponding author, jgmf@amazon.com. All authors were associated with Amazon at the time of publication. 

for multilingualism span the software stack and a variety of operational considerations, but one difficulty in creating massively multilingual NLU models is the lack of labeled data for training and evaluation, particularly data that is realistic for the task and that is natural for each given language. High naturalness typically requires human-based vetting, which is often costly. 

We present MASSIVE ( _M_ ultilingual _A_ mazon _S_ LU Resource Package (SLURP) for _S_ lot filling, _I_ ntent classification, and _V_ irtual assistant _E_ valuation), a new 1M-example dataset composed of realistic, human-created virtual assistant utterance text spanning 51 languages, 60 intents, 55 slot types, and 18 domains. With the English seed data included, there are 587k train utterances, 104k dev utterances, 152k test utterances, and 153k utterances currently held out for the MMNLU-22 competition, which will be released after the competition. We have released our data, code, and models[1] . 

MASSIVE was created by localizing the SLURP NLU dataset (created only in English) in a parallel manner. SLURP is described further in Section 2, linguistic analyses of the dataset in Section 3, and the localization process in Section 4.3. Results for Massively Multilingual NLU (MMNLU) modeling, in which a single model can perform NLU on any of the incoming languages, are given in Section 5. 

## **2 Related Work** 

Prior researchers have emphasized the need to explore the unique challenges of low-resource languages (Simpson et al., 2008; Strassel and Tracey, 2016; Cruz and Cheng, 2020; Lakew et al., 2020; Marivate et al., 2020; Magueresse et al., 2020; Goyal et al., 2021), while the growing number and 

> 1https://github.com/alexa/massive 

4277 

_Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics Volume 1: Long Papers_ , pages 4277–4302 July 9-14, 2023 ©2023 Association for Computational Linguistics 

size of language models (mBERT (Devlin, 2018), RoBERTa (Liu et al., 2019b), XLM (Lample and Conneau, 2019), XLM-R (Conneau et al., 2020), mBART (Liu et al., 2020), MARGE (Lewis et al., 2020), and mT5 (Xue et al., 2021) pre-trained on massively multilingual corpora have allowed for significant improvements in supporting them. However, the creation of evaluation datasets for specific tasks has not kept pace. Some tasks, such as Named Entity Recognition (NER) or translation, lend themselves to mining existing corpora (Tiedemann, 2012; Pan et al., 2017; Hu et al., 2020), while others such as NLU, the focus here, require the creation of new data and schema-specific annotations. Beyond the cost, even identifying a sufficient number of speakers for data generation and quality control can be difficult. Most studies have thus focused on collecting data for one such low-resource language and determining the utility of multilingual models or cross-lingual learning from more readily available languages. Moreover, such datasets are often isolated collections, creating an environment of multiple datasets not easily comparable across the different languages or tasks. There have been exceptions, such as SQuAD (Rajpurkar et al., 2016) and XQuAd (Artetxe et al., 2019), ATIS (Price, 1990), its Hindi and Turkish extension (Upadhyay et al., 2018), and MultiATIS++ (Xu et al., 2020), and Snips (Coucke et al., 2018) with its addition of French (Saade et al., 2019), where researchers have extended popular English benchmark datasets to new languages. This work focuses on the general multi-domain NLU task and builds off the SLURP (Bastianelli et al., 2020) benchmark dataset to extend to an unprecedented 50 new languages. 

For the task of NLU, the ATIS dataset has been popular in the NLP community since its first release. MultiATIS++ was one of the first efforts to extend an NLU dataset across a significant number of languages (nine), yet remained in the limited domain of airline bookings. While proving an asset, it has been questioned what is left to learn from such a dataset (Tur et al., 2010). Facebook released a general Intelligent Virtual Assistant (IVA) dataset across the domains of Alarm, Reminder, and Weather (Schuster et al., 2019) created for the purpose of demonstrating cross-lingual transfer learning; and so did not need to be parallel or have an equal number of datapoints, resulting in far fewer examples in Thai (5k) compared to Spanish (7.6k) and English (43k). The Snips datasets 

(both the original English only and the English and French releases) are most similar to the NLU contained in the MASSIVE dataset, spanning smart home and music domains for a generic voice-based virtual assistant. 

The first iteration for the foundation of the MASSIVE dataset was the NLU Evaluation Benchmarking Dataset, with 25k utterances across 18 domains (Liu et al., 2019a). The authors updated the dataset and added audio and ASR transcriptions in the release of the Spoken Language Understanding Resource Package (SLURP) (Bastianelli et al., 2020), allowing for full end-to-end Spoken Language Understanding (SLU) evaluation similar to the Fluent Speech Commands dataset (Lugosch et al., 2019) and Chinese Audio-Textual Spoken Language Understanding (CATSLU) (Zhu et al., 2019). An overview of selected existing NLU datasets can be seen in Table 1. 

We release the MASSIVE dataset along with baselines from large pre-trained models fine-tuned on the NLU slot and intent prediction tasks. Early cross-lingual and multilingual NLU modeling approaches used projection or alignment methods (Yarowsky et al., 2001), focusing on string matching, edit distance, or consonant signatures (Ehrmann et al., 2011), lookup lexicons for lowresource languages (Mayhew et al., 2017), and aligning (Xie et al., 2018) or jointly training word embeddings (Singla et al., 2018). More recently, researchers have borrowed encoders from pre-trained neural translation models before building subsequent classifiers and NER models (Eriguchi et al., 2018; Schuster et al., 2019), also focusing on language-agnostic and language specific features to learn what information to share between languages (Chen et al., 2019b). Generative parsing has been demonstrated using sequence-to-sequence models and pointer networks (Rongali et al., 2020). With the rise of BERT and large pre-trained language models, we have also seen impressive demonstrations of zero-shot performance, where subword tokenization WordPiece overlap helps but is not even necessary to realize improvements (Pires et al., 2019; K et al., 2020), as well as production multilingual NLU improvements with distillation and full fine-tuning (FitzGerald et al., 2022). The translation task has then been incoporated in the pretraining (Wang et al., 2021) of these models or even as part of the final NLU hypothesis for streamlined multilingual production systems (FitzGerald, 

4278 

|Name|# Lang|Utt per Lang|Domains|Intents|Slots|
|---|---|---|---|---|---|
|MASSIVE|51|19,521|18|60|55|
|SLURP (Bastianelli et al.,2020)|1|16,521|18|60|55|
|NLU Evaluation Data (Liu et al.,2019a)|1|25,716|18|54|56|
|Airline Travel Information System (ATIS) (Price,1990)|1|5,871|1|26|129|
|ATIS with Hindi and Turkish (Upadhyay et al.,2018)|3|1,315-5,871|1|26|129|
|MultiATIS++ (Xu et al.,2020)|9|1,422-5,897|1|21-26|99-140|
|Snips (Coucke et al.,2018)|1|14,484|-|7|53|
|Snips with French (Saade et al.,2019)|2|4,818|2|14-15|11-12|
|Task Oriented Parsing (TOP) (Gupta et al.,2018)|1|44,873|2|25|36|
|Multilingual Task-Oriented Semantic Parsing (MTOP) (Li et al.,2021)|6|15,195-22,288|11|104-113|72-75|
|Cross-lingual Multilingual Task Oriented Dialog (Schuster et al.,2019)|3|5,083-43,323|3|12|11|
|Microsoft Dialog Challenge (Li et al.,2018b)|1|38,276|3|11|29|
|Fluent Speech Commands (FSC) (Lugosch et al.,2019)|1|30,043|-|31|-|
|Chinese Audio-Textual Spoken Language Understanding (CATSLU) (Zhu et al.,2019)|1|16,258|4|-|94|



Table 1: Selected NLU benchmark datasets with number of languages, utterances per language, domain count, intent count, and slot count. 

2020). Researchers have propped up training data by translating and projecting labels into the target language (Xu et al., 2020) and discovered more sophisticated approaches to alignment such as translate and fill using mT5 to train the filler (Nicosia et al., 2021). Recent work has even delved into the application of these techniques to lower-resource languages such as Persian. For example, ParsiNLU explores a variety of NLU tasks for Parsi, finetuning mT5 of various sizes (Khashabi et al., 2021). Similarly these techniques have also been used, even a bit earlier, for text summarization (Farahani et al., 2021). 

## **3 Language Selection and Linguistic Analysis** 

## **3.1 Language Selection** 

The languages in MASSIVE were chosen according to the following considerations. First, we acquired cost and worker availability estimates for over 100 languages, providing a constraint to our choices given our fixed budget. Second, we determined existing languages available in major virtual assistants, such that the dataset could be used to benchmark today’s systems. Third, we categorized the full pool of languages according to their genera as taken from the World Atlas of Linguistic Structures (WALS) database (Dryer and Haspelmath, 2013), where a genus is a language group that is clear to most linguists without systematic comparative analysis. Genus is a better indicator of typological diversity, which we sought to maximize, than language family (Dryer, 1989). Fourth, we used the eigenvector centrality of Wikipedia articles, tweets, and book translations (Ronen et al., 2014) as proxies for the internet influence and thus 

the resource availability of a given language, particularly for self-supervised pretraining applications, and we chose languages spanning the breadth of resource availability. Fifth, we examined the script of each language, seeking to increase script diversity to drive experimentation in tokenization and normalization. 

Ultimately, we created 50 new, distinct text corpora, representing 49 different spoken languages. Mandarin Chinese was collected twice, once with native speakers who use the traditional set of characters, and once with native speakers who use the modern simplified set of characters. There are 14 language families in the dataset. The term “language family” usually refers to a group of languages which are known to be genetically related, that is, they all descend from a common ancestor language. In MASSIVE, we also include “language isolates” as families. These are languages that have no clear relationship to any known language. Our choices are given in Table 2. 

## **3.2 Scripts** 

There are 21 distinct scripts used in the dataset. The majority of languages in MASSIVE (28 including English) use some variety of the Latin alphabet, which is also the most widely used script in the world. The Arabic script is used for three languages, the Cyrillic script for two languages, and the remaining 18 languages have “unique” scripts, in the sense that only one language in the dataset uses that script. Fourteen scripts are unique to a single language, although they may belong to a larger family of writing systems. For example, the Dravidian languages in MASSIVE have their own scripts, but are all members of the general Brahmi 

4279 

|Code<br>Name<br>Script<br>Genus|Code<br>Name<br>Script<br>Genus|Code<br>Name<br>Script<br>Genus|
|---|---|---|
|af-ZA<br>Afrikaans<br>Latn<br>Germanic<br>am-ET<br>Amharic<br>Ethi<br>Semitic<br>ar-SA<br>Arabic<br>Arab<br>Semitic<br>az-AZ<br>Azerbaijani<br>Latn<br>Turkic<br>bn-BD<br>Bengali<br>Beng<br>Indic<br>cy-GB<br>Welsh<br>Latn<br>Celtic<br>da-DK<br>Danish<br>Latn<br>Germanic<br>de-DE<br>German<br>Latn<br>Germanic<br>el-GR<br>Greek<br>Grek<br>Greek<br>en-US<br>English<br>Latn<br>Germanic<br>es-ES<br>Spanish<br>Latn<br>Romance<br>fa-IR<br>Persian<br>Arab<br>Iranian<br>f-FI<br>Finnish<br>Latn<br>Finnic<br>fr-FR<br>French<br>Latn<br>Romance<br>he-IL<br>Hebrew<br>Hebr<br>Semitic<br>hi-IN<br>Hindi<br>Deva<br>Indic<br>hu-HU<br>Hungarian<br>Latn<br>Ugric|hy-AM<br>Armenian<br>Armn<br>Armenian<br>id-ID<br>Indonesian<br>Latn<br>Malayo-Sumbawan<br>is-IS<br>Icelandic<br>Latn<br>Germanic<br>it-IT<br>Italian<br>Latn<br>Romance<br>ja-JP<br>Japanese<br>Jpan<br>Japanese<br>jv-ID<br>Javanese<br>Latn<br>Javanese<br>ka-GE<br>Georgian<br>Geor<br>Kartvelian<br>km-KH<br>Khmer<br>Khmr<br>Khmer<br>kn-IN<br>Kannada<br>Knda<br>Southern Dravidian<br>ko-KR<br>Korean<br>Kore<br>Korean<br>lv-LV<br>Latvian<br>Latn<br>Baltic<br>ml-IN<br>Malayalam<br>Mlym<br>Southern Dravidian<br>mn-MN<br>Mongolian<br>Cyrl<br>Mongolic<br>ms-MY<br>Malay<br>Latn<br>Malayo-Sumbawan<br>my-MM<br>Burmese<br>Mymr<br>Burmese-Lolo<br>nb-NO<br>Norwegian<br>Latn<br>Germanic<br>nl-NL<br>Dutch<br>Latn<br>Germanic|pl-PL<br>Polish<br>Latn<br>Slavic<br>pt-PT<br>Portuguese<br>Latn<br>Romance<br>ro-RO<br>Romanian<br>Latn<br>Romance<br>ru-RU<br>Russian<br>Cyrl<br>Slavic<br>sl-SI<br>Slovenian<br>Latn<br>Slavic<br>sq-AL<br>Albanian<br>Latn<br>Albanian<br>sv-SE<br>Swedish<br>Latn<br>Germanic<br>sw-KE<br>Swahili<br>Latn<br>Bantoid<br>ta-IN<br>Tamil<br>Taml<br>Southern Dravidian<br>te-IN<br>Telugu<br>Telu<br>South-Central Dravidian<br>th-TH<br>Thai<br>Thai<br>Kam-Tai<br>tl-PH<br>Tagalog<br>Latn<br>Greater Central Philippine<br>tr-TR<br>Turkish<br>Latn<br>Turkic<br>ur-PK<br>Urdu<br>Arab<br>Indic<br>vi-VN<br>Vietnamese<br>Latn<br>Viet-Muong<br>zh-CN<br>Mandarin<br>Hans<br>Chinese<br>zh-TW<br>Mandarin<br>Hant<br>Chinese|



Table 2: The 51 languages of MASSIVE, including scripts and genera. 

class of scripts. The other two scripts are unique in that only one language in the dataset uses them, but they are more widely used in the real world: Ge’ez and Chinese. Ge’ez is represented by Amharic in the dataset, but is used for several languages in East Africa, such as Tigrinya. The Chinese script is represented by Mandarin, but is used by other languages in China such as Cantonese. 

## **3.3 Sentence Types** 

MASSIVE consists of utterances directed at a device, rather than a person, which has some consequences for the type of linguistic patterns it contains. Specifically, the corpus primarily consists of interrogatives (i.e., questions) and imperatives (commands or requests). There are relatively few declarative utterances in the set. This is in contrast to many large datasets from other sources (e.g., wikipedia, movie scripts, newspapers) which contain a high proportion of declaratives, since the language is collected from situations where humans are communicating with humans. 

In the context of a voice assistant, a user typically asks a device to perform an action or answer a question, so declaratives are less common. For instance, a person might use an imperative “tell me if it calls for rain today” or ask a question “will it rain today,” but they would not tell their device “it’s raining today.” When declaratives are used with voice assistants, they generally have the pragmatic effect of a directive. For instance, a virtual assistant can respond to the declarative “it’s cold in here” by turning up the temperature (Thattai et al., 2020). Although syntactically it looks like a declarative, such an utterance has the force of an imperative. 

The standard unit of analysis in linguistics is 

the declarative sentence, and there is relatively less known about imperatives and questions. MASSIVE presents an opportunity to study these sentence forms, and the parallel nature of the corpus makes cross-linguistic comparisons even easier. 

## **3.4 Word Order** 

Languages have intricate rules for ordering words depending on the word-type and sentence-type. In English, the word order for statements (“you are leaving”) is different from questions (“are you leaving?”). This is not mandatory, and sometimes the pitch of the voice is enough to indicate a question (e.g. “you’re leaving?” with a rising intonation). 

When considering word order at a typological level, it is common to simplify the situation and consider only affirmative declarative sentences and only three grammatical elements: the verb (V), its subject (S), and its object (O). This makes for six possible word orders: SVO, SOV, VOS, VSO, OVS, and OSV. All six orders have been documented, although the overwhelming majority of languages use Subject-initial ordering, while Object-initial ordering is extremely rare. 

In MASSIVE, 39 languages are subject-initial (24 SVO and 15 SOV), while only three are verb-initial (VSO specifically). No object-initial languages are represented. Five languages are marked in WALS as having no preferred word order, and four do not have any word order data at all. 

## **3.5 Imperative Marking** 

The languages in MASSIVE have a variety of ways of indicating the imperative mood of an utterance. The majority of them (33) use some kind of verb morphology, such as adding a suffix. About half of those languages (18) have distinct imperative 

4280 

marking for singular or plural addressees. The utterances in MASSIVE are technically directed at a single addressee, the voice assistant, but since some languages use the plural as an indicator of politeness (see below) all varieties of imperatives will likely occur in this dataset. There are ten languages without any special morphology, and they indicate imperative through other means, such as word order or vocabulary choice. 

Ten languages in the dataset have a specialized distinction between imperatives, for commands directed at another individual, and “hortatives”, where the command also includes the speaker. English verbs are not directly marked for hortative, but the auxiliary verb “let” can convey the mood instead. For example, “write this down” is an imperative and only the addressee need write anything, while “let’s write this down” is a hortative and the speaker is also expected to write. The pervasiveness of hortatives in the context of a voice assistant is an open question. 

Four languages have “optative” moods, which are subtly different from imperatives. In the optative, a speaker expresses a wish or desire, as opposed to giving a direct command. However, in the right context, an optative may carry the same pragmatic weight as an imperative, and strongly imply that someone ought to do something. English has no specific optative form, but a similar mood can be conveyed using conditionals. For example, “buy this bag for me” is an imperative while “if only someone would buy me this bag” is closer to an optative. Optative forms are not well studied in linguistics, as they require specific contexts which can be difficult to create during field work, but they may be more common in device-directed utterances. 

Lastly, some languages distinguish between imperatives, when telling someone to do something, and “prohibitives”, when telling someone not to do something. In the MASSIVE set, there are 18 languages with specialized negative particles which can only co-occur with imperative verbs. Vietnamese for instance uses the words “ch˘ang” or “không” to negate declarative sentences, but uses “chó” or “dung” to negate imperatives. Another ten languages have special verbs for the prohibitive, although these may overlap with other grammatical features of the language. In Spanish, for example, the prohibitive form of a verb is the same as the subjunctive form. 

## **3.6 Politeness** 

Many languages encode different levels of politeness through their use of pronouns. Many European languages distinguish between “familiar” and “formal” pronouns, with the “formal” pronouns often morphologically identical to a plural. In French, the second-person singular “tu” is used between friends, while the second-person plural “vous” is used when speaking to a group, or to an individual of higher social rank (such as an employee to a manager). These politeness systems are heavily influenced by social context, and the MASSIVE dataset gives us a chance to see how people adapt their language when speaking to a virtual assistant instead of another human. 

Nearly half of the languages in MASSIVE (21) make a two-way formal/informal distinction in their second-person pronouns. This is probably due to the fact that most MASSIVE languages are European, and the binary politeness distinctions are the most common strategy in that family. A further eight languages have more than two levels of formality, such as informal, formal, and honorific. Seven languages have an “avoidance” strategy, which means that pronouns are omitted entirely in a polite scenario. Finally, eleven languages have no data on politeness in WALS at all. 

## **4 Collection Setup and Execution** 

## **4.1 Heldout Evaluation Split** 

We randomly sampled a subset of the English seed data which was then paraphrased by professional annotators, resulting in new, more challenging utterances, including 49% more slots per utterance. These utterances were localized along with the other splits to be used as a held out evaluation set for the Massively Multilingual NLU-22 competition and workshop[2] . 

## **4.2 Vendor Selection and Onboarding** 

The MASSIVE dataset was collected using a customized workflow powered by Amazon MTurk. We required a vendor pool with the capability and resources to collect a large multilingual dataset. Our original vendor pool consisted of five vendors adjudicated based on previous engagements. This vendor pool was reduced to three based on engagement and resource availability. Vendors for each language were selected based on their resource 

2mmnlu-22.github.io 

4281 

availability and proposed cost. A majority of languages were supported by a single vendor, while some languages required cross-vendor support to be completed with the required quality and within the required timeline. 

We offered two mechanisms to vendors for evaluating workers to be selected for each language. The first, which was used to select workers for the translation task, was an Amazon MTurk-hosted fluency test where workers listen to questions and statements in the relevant language and were evaluated using a multiple-choice questionnaire. The second, which was used to select workers for the judgment task, was a test with a set of three judgments that the vendor could use to assess if workers were able to detect issues in the translated utterances. In order to further improve worker selection quality, we created a translator quiz using the Amazon MTurk instructions that were created for translation and judgment tasks, coupled with customized locallanguage examples. The workers were required to prove that they understood the instructions for the project based on a series of questions. 

Before commencing operations, an initial pilot run of this customized workflow was completed in three languages. A few workers per vendor were chosen to engage in this exercise. The pilot run helped improve clarity of instructions, determine reporting methods, and share open questions. 

## **4.3 Collection Workflows** 

The collection was conducted by locale on an individual utterance level. Each utterance from the “train,” “dev,” “test,” and “heldout” splits of the SLURP dataset went through two sequential task workflows and a judgment workflow. The first task is slot translation or localization (see Figure 1). Workers are presented the entire utterance with colored highlighting of the slot values for the utterance (if any) and then presented with each slot value and its corresponding label individually. The worker is asked to either localize or translate the slot, depending on whether the value should be translated (e.g., “tomorrow”) or localized (e.g., the movie “La La Land”, which in French is “Pour l’amour d’Hollywood.” Other entities like regionally known songs or artists could also be localized to a more relevant, known song or artist for that language or region). There is also an option to keep the slot as is, such as for names (e.g., “Taylor Swift”) or proper nouns where the original English spelling 

should be retained. The metadata of the released dataset includes whether the worker elected to “localize,” “translate,” or keep the slot “unchanged,” primarily for the purposes of researchers evaluating machine translation systems, where it would be unreasonable to expect the system to “localize” to a specific song name the worker selected. 

After the slot task, the second worker is asked to translate or localize the entire phrase using the slot task output provided by the first worker (see Figure 2). The phrase worker can decide to keep the slot as it was translated, modify it, or remove it entirely if it is not relevant for the language in that scenario. This worker is also responsible for aligning grammatical genders or prepositional affixes to any of the slots. 

Note that this two-step system alleviates the annotation burden often encountered with such work. Traditionally in such collections, workers would be given a light annotation guide and asked to highlight spans of the slots in a translated or localized utterance. In this system, the first step of slot translation and subsequent insertion obviates the need for workers to understand nuanced span notation, which can be complex for highly inflected languages (prepositions outside the span in English would not be carried over in the localization, but would be in the traditional span annotation workflow). 

## **4.4 Quality Assurance** 

The output of the second workflow (the fully localized utterance) is judged by three workers for (1) whether the utterance matches the intent semantically, (2) whether the slots match their labels semantically, (3) grammaticality and naturalness, (4) spelling, and (5) language identification—English or mixed utterances are acceptable if that is natural for the language, but localizations without any tokens in the target language were not accepted. See Figure 3 for how this is presented to the Amazon MTurk worker. These judgments are also included in the metadata of the dataset. In addition to the workers judging each other’s work, the collection system had alarms in place for workers with high rejection rates, high rates of slot deletion, and high rates of English tokens in the translations. Workers were also monitored to see if their tasks were primarily machine translated. Such workers were removed from the pool and all of their work was resubmitted to be completed by the other workers. 

4282 

Additionally, the authors performed several deep dives into languages with which they were familiar. 

## **5 Model Benchmarking** 

## **5.1 Setup** 

As initial model benchmarks, we fine-tuned publicly-available pre-trained language models on the MASSIVE dataset and evaluated them on intent classification and slot filling. Our models of choice for this exercise were XLM-Roberta (XLM-R; Conneau et al. 2020) and mT5 (Xue et al., 2021). 

In the case of XLM-R, we utilized the pretrained encoder with two separate classification heads trained from scratch, based on JointBERT (Chen et al., 2019a). The first classification head used the pooled output from the encoder to predict the intent and the second used the sequence output to predict the slots. As pooling for the intent classification head, we experimented with using hidden states from the first position, averaged hidden states across the sequence, and the maximally large hidden state from the sequence. 

With mT5, we explored two separate architectures. In one architecture, we only used the pre-trained encoder extracted from mT5, and we trained two classification heads from scratch similarly to the XLM-R setup. We refer to this setup as mT5 Encoder-Only. In the other architecture, we used the full sequence-to-sequence mT5 model in text-to-text mode, where the input is “Annotate:” followed by the unlabeled utterance. The decoder output is a sequence of labels (including the Other label) for all of the tokens followed by the intent. We did not add the slots and intents to the vocabulary, but we instead allowed them to be tokenized into subwords. We refer to this model as mT5 Text-to-Text. For all models, we used the Base size, which corresponds to 270M parameters for XLM-R, 258M parameters for mT5 Encoder-Only, and 580M parameters for mT5 Text-to-Text, including 192M parameters for embeddings for all three. 

For each model, we performed 128 trials of hyperparameter tuning using the Tree of Parzen Estimators algorithm and Asynchronous Successive Halving Algorithm (ASHA) (Li et al., 2018a) for scheduling, which are both part of the hyperopt library (Bergstra et al., 2013) integrated into the ray[tune] library (Liaw et al., 2018), which is itself integrated into the Trainer from the transformers library (Wolf et al., 2020), 

which we used for modeling and for our pretrained models. Our hyperparameter search spaces, sampling types, and final choices are given in Table 5. We trained our models with the Adam optimizer (Kingma and Ba, 2017) and chose the best performing model checkpoint based on overall exact match accuracy across all locales. Hyperparameter tuning and fine-tuning was performed using single p3dn.24xlarge instances (8 x Nvidia v100) for XLM-R and mT5 Text-to-Text and a single g4dn.metal instance (8 x Nvidia T4) for mT5 Encoder-Only. Hyperparameter tuning times were less than 4 days per model and training times were less than 1 day per model. 

Our dataset includes several languages where white spacing is not used as a word delimiter. In some cases, spaces do occur, but they might serve as phrase delimiters or denote the end of a sentence. Three of these written languages, Japanese, Chinese (Traditional), and Chinese (Simplified), do not use spaces anywhere except to identify the end of a sentence. For these languages, we separate each character in the unlabeled input with a whitespace. We leave exploration of more sophisticated techniques (such as MeCab for Japanese; Kudo 2005) to future work. We use the default spacing provided by annotators for all other languages. 

Zero-shot performance was also assessed, in which the models were trained on English data, validation was performed on all languages, and testing was performed on all non-English locales. 

## **5.2 Results and Analysis** 

Table 3 shows the results for each model and training setup, including those for the best performing locale, the worst performing locale, and locale-averaged results for intent accuracy, microaveraged slot F1 score, and exact match accuracy. Zero-shot exact match performance is 25-37 points worse than that of full-dataset training runs. Additionally, the variance in task performance across locales is significantly greater for the zero-shot setup than for full-dataset training. For example, there is a 15 point difference in exact match accuracy between the highest and lowest locales for mT5 Text-to-Text when using the full training set, while the gap expands to 44 points with zero-shot. 

We compared the pretraining data quantities by language for XLM-R to its per-language task performance values, and in the zero shot setup, we found a Pearson correlation of 0.54 for exact match 

4283 

|Model|Intent Acc (%)<br>Slot F1 (%)<br>Exact Match Acc (%)<br>High<br>Low<br>Avg<br>High<br>Low<br>Avg<br>High<br>Low<br>Avg|Intent Acc (%)<br>Slot F1 (%)<br>Exact Match Acc (%)<br>High<br>Low<br>Avg<br>High<br>Low<br>Avg<br>High<br>Low<br>Avg|Intent Acc (%)<br>Slot F1 (%)<br>Exact Match Acc (%)<br>High<br>Low<br>Avg<br>High<br>Low<br>Avg<br>High<br>Low<br>Avg|
|---|---|---|---|
|mT5 Base<br>Text-to-Text<br>mT5 Base<br>Encoder-Only<br>XLM-R Base|87.9_±_1.2<br>79.0_±_1.5<br>85.3_±_0.2<br>en-US<br>km-KH<br>89.0_±_1.1<br>79.1_±_1.5<br>86.1_±_0.2<br>en-US<br>km-KH<br>88.3_±_1.2<br>77.2_±_1.5<br>85.1_±_0.2<br>en-US<br>km-KH|86.8_±_0.7<br>67.6_±_0.4<br>76.8_±_0.1<br>th-TH<br>ja-JP<br>85.7_±_0.7<br>64.5_±_0.4<br>75.4_±_0.1<br>th-TH<br>ja-JP<br>83.5_±_0.7<br>63.3_±_0.4<br>73.6_±_0.1<br>th-TH<br>ja-JP|73.4_±_1.6<br>58.3_±_1.8<br>66.6_±_0.2<br>th-TH<br>ja-JP<br>72.3_±_1.6<br>57.8_±_1.8<br>65.9_±_0.2<br>th-TH<br>ja-JP<br>70.1_±_1.6<br>55.8_±_1.8<br>63.7_±_0.2<br>th-TH<br>ja-JP|
|(a) Test results when using the full training set||||
|Model|Intent Acc (%)<br>Slot F1 (%)<br>Exact Match Acc (%)<br>High<br>Low<br>Avg<br>High<br>Low<br>Avg<br>High<br>Low<br>Avg|||
|mT5 Base<br>Text-to-Text<br>mT5 Base<br>Encoder-Only<br>XLM-R Base|79.9_±_1.4<br>25.7_±_1.6<br>62.9_±_0.2<br>nl-NL<br>ja-JP<br>76.4_±_1.5<br>27.1_±_1.6<br>61.2_±_0.2<br>nl-NL<br>ja-JP<br>85.2_±_1.3<br>44.8_±_1.8<br>70.6_±_0.2<br>sv-SE<br>ja-JP|64.3_±_0.7<br>13.9_±_0.3<br>44.8_±_0.1<br>de-DE<br>ja-JP<br>59.5_±_1.0<br>6.3_±_0.2<br>41.6_±_0.1<br>th-TH<br>ja-JP<br>68.4_±_0.7<br>15.4_±_0.3<br>50.3_±_0.1<br>sv-SE<br>ja-JP|53.2_±_1.8<br>9.4_±_1.0<br>34.7_±_0.2<br>sv-SE<br>ja-JP<br>44.3_±_1.8<br>4.2_±_0.7<br>28.8_±_0.2<br>sv-SE<br>ja-JP<br>57.9_±_1.8<br>9.8_±_1.1<br>38.7_±_0.2<br>sv-SE<br>ja-JP|



(b) Zero-shot test results after training only on en-US 

Table 3: Modeling results for (a) training runs on the full training dataset and (b) zero-shot training runs, in which training was performed only with en-US data, validation was performed with all locales, and testing was performed on all locales except for en-US. Each table includes the highest locale, the lowest locale, and locale-averaged results for intent accuracy, micro-averaged slot F1 score, and exact match accuracy. Intervals for 95% confidence are given assuming normal distributions. 

accuracy, 0.58 for intent accuracy, and 0.46 for micro-averaged slot F1 score. In the full dataset training setup, the correlations decrease to 0.42 for exact match accuracy, 0.47 for intent accuracy, and 0.24 for micro-averaged slot F1 score. This suggests that the constant per-language data quantities in MASSIVE help to mitigate the effects of the language-skewed pretraining data distribution. 

In Thai, for which spacing is optional, the model can learn from artificial spacing in the input (around where the slots will be) to improve task performance. For Khmer, the workers had a difficult time adapting their translations and localizations to properly-slotted outputs given the space-optional nature of the language. Additionally, for Japanese and Chinese, we added spaces between all characters when modeling. These single-character inputs differ from the non-spaced inputs used during pretraining, which would be chunked into groups of characters by the tokenizer with corresponding embeddings. By splitting into single characters, we don’t allow the model to the use embeddings learned for chunks of characters. This is a likely major cause of the drop in exact match accuracy for Japanese from 58.3% when training on the full dataset to 9.4% for zero shot. In the zero shot setup, the model relies solely on pretrained data representations, and individually-spaced characters are rare 

in the pretraining data. That said, character spacing was necessary in order to properly assign the slots to the right characters. As mentioned in Section 5.1, we leave exploration of more sophisticated spacing techniques for slot filling (such as MeCab; Kudo 2005) to future work. 

Discounting for artificial spacing effects, Germanic genera and Latin scripts performed the best overall (See Appendix E), which is unsurprising given the amount of pretraining data for those genera and scripts, as well as the quantity of Germanic and Latin-script languages in MASSIVE. Within the Germanic genera, Swedish, English, Danish, Norwegian, and Dutch all performed comparably (within 95% confidence bounds) for exact match accuracy. Icelandic was the lowest-performing Germanic language, likely due to a lack of pretraining data, as well as to its linguistic evolution away from the others due to isolated conditions. 

## **6 Conclusion** 

We have released a truly MASSIVE multilingual dataset for NLU spanning 51 typologically diverse languages. Our hope is that MASSIVE will encourage many new innovations in massively multilingual NLU, other NLP tasks such as machine translation, and new linguistic analyses, such as with imperative morphologies. 

4284 

## **Limitations and Ethical Considerations** 

There are several significant limitations of the MASSIVE dataset and of our modeling. Starting with the dataset, the per-language data quantities are relatively small at 19.5k total records and 11.5k records for training. Second, there are some lowquality utterances, both in the seed data and in the translations. For the most part, these are surfaced through the judgment scores we provide for each record, but if a user does filtering based on these judgments, then the data size decreases even further. Third, the data were originally created through crowd-sourcing, not from a real virtual assistant, which introduces artificialities. Relatedly, allowing the worker to decide on translation versus localization of slot entities added further noise to the dataset, although we try to store this decision in the metadata. Fourth, our labeling schema is relatively simple when compared with hierarchical labeling schemata or flat schemata with more intent and slot options. Fifth, our collection system did not have a robust method to preserving or denoting native tokenization practices—some languages do not separate with whitespace, while others do but there is no set practice. This results in potentially easier (larger chunks to predict slot labels) or harder (each character individually predicted) tasks. Sixth, it’s possible, though unlikely, that some of our new crowd-sourced records may contain toxic or otherwise objectionable content. We performed analyses to check for such malicious activities and did not find any as such. Regarding modeling, we have only investigated base-sized models in relatively standard setups, leaving room for much more sophisticated modeling. The risks associated with this dataset and work are relatively low, given that we have released a research dataset meant to promote better multilinguality in NLP systems. 

## **References** 

- Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2019. On the cross-lingual transferability of monolingual representations. _CoRR_ , abs/1910.11856. 

- Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski, and Verena Rieser. 2020. SLURP: A spoken language understanding resource package. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 7252–7262, Online. Association for Computational Linguistics. 

Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. In _Proceedings of the 30th International Conference on Machine Learning_ , volume 28 of _Proceedings of Machine Learning Research_ , pages 115–123, Atlanta, Georgia, USA. PMLR. 

   - Qian Chen, Zhu Zhuo, and Wen Wang. 2019a. Bert for joint intent classification and slot filling. _ArXiv_ , abs/1902.10909. 

   - Xilun Chen, Ahmed Hassan Awadallah, Hany Hassan, Wei Wang, and Claire Cardie. 2019b. Multi-source cross-lingual model transfer: Learning what to share. In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 3098– 3112, Florence, Italy. Association for Computational Linguistics. 

   - Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 8440– 8451, Online. Association for Computational Linguistics. 

   - Alice Coucke, Alaa Saade, Adrien Ball, Théodore Bluche, Alexandre Caulier, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Francesco Caltagirone, Thibaut Lavril, Maël Primet, and Joseph Dureau. 2018. Snips voice platform: an embedded spoken language understanding system for privateby-design voice interfaces. 

   - Jan Christian Blaise Cruz and Charibeth Cheng. 2020. Establishing baselines for text classification in lowresource languages. 

   - Jacob Devlin. 2018. Multiligual bert. 

   - Matthew S. Dryer. 1989. Large linguistic areas and language sampling. _Studies in Language_ , 13:257– 292. 

   - Matthew S. Dryer and Martin Haspelmath, editors. 2013. _WALS Online_ . Max Planck Institute for Evolutionary Anthropology, Leipzig. 

   - Maud Ehrmann, Marco Turchi, and Ralf Steinberger. 2011. Building a multilingual named entityannotated corpus using annotation projection. In _Proceedings of the International Conference Recent Advances in Natural Language Processing 2011_ , pages 118–124, Hissar, Bulgaria. Association for Computational Linguistics. 

   - Akiko Eriguchi, Melvin Johnson, Orhan Firat, Hideto Kazawa, and Wolfgang Macherey. 2018. Zero-shot cross-lingual classification using multilingual neural machine translation. 

- James Bergstra, Daniel Yamins, and David Cox. 2013. 

4285 

- Mehrdad Farahani, Mohammad Gharachorloo, and Mohammad Manthouri. 2021. Leveraging parsbert and pretrained mt5 for persian abstractive text summarization. _2021 26th International Computer Conference, Computer Society of Iran (CSICC)_ . 

- Jack FitzGerald, Shankar Ananthakrishnan, Konstantine Arkoudas, Davide Bernardi, Abhishek Bhagia, Claudio Delli Bovi, Jin Cao, Rakesh Chada, Amit Chauhan, Luoxin Chen, Anurag Dwarakanath, Satyam Dwivedi, Turan Gojayev, Karthik Gopalakrishnan, Thomas Gueudre, Dilek Hakkani-Tur, Wael Hamza, Jonathan Hueser, Kevin Martin Jose, Haidar Khan, Beiye Liu, Jianhua Lu, Alessandro Manzotti, Pradeep Natarajan, Karolina Owczarzak, Gokmen Oz, Enrico Palumbo, Charith Peris, Chandana Satya Prakash, Stephen Rawls, Andy Rosenbaum, Anjali Shenoy, Saleh Soltan, Mukund Harakere Sridhar, Liz Tan, Fabian Triefenbach, Pan Wei, Haiyang Yu, Shuai Zheng, Gokhan Tur, and Prem Natarajan. 2022. Alexa teacher model: Pretraining and distilling multibillion-parameter encoders for natural language understanding systems). In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , KDD. ACM. 

- Jack G. M. FitzGerald. 2020. Stil – simultaneous slot filling, translation, intent classification, and language identification: Initial results using mbart on multiatis++. 

- Naman Goyal, Cynthia Gao, Vishrav Chaudhary, PengJen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzman, and Angela Fan. 2021. The flores-101 evaluation benchmark for low-resource and multilingual machine translation. 

- Sonal Gupta, Rushin Shah, Mrinal Mohit, Anuj Kumar, and Mike Lewis. 2018. Semantic parsing for task oriented dialog using hierarchical representations. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 2787–2792, Brussels, Belgium. Association for Computational Linguistics. 

- Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalization. 

- Karthikeyan K, Zihan Wang, Stephen Mayhew, and Dan Roth. 2020. Cross-lingual ability of multilingual bert: An empirical study. 

- Daniel Khashabi, Arman Cohan, Siamak Shakeri, Pedram Hosseini, Pouya Pezeshkpour, Malihe Alikhani, Moin Aminnaseri, Marzieh Bitaab, Faeze Brahman, Sarik Ghazarian, Mozhdeh Gheini, Arman Kabiri, Rabeeh Karimi Mahabadi, Omid Memarrast, Ahmadreza Mosallanezhad, Erfan Noury, Shahab Raji, Mohammad Sadegh Rasooli, Sepideh Sadeghi, Erfan Sadeqi Azer, Niloofar Safi Samghabadi, Mahsa Shafaei, Saber Sheybani, Ali Tazarv, and Yadollah Yaghoobzadeh. 2021. Parsinlu: A suite of language understanding challenges for persian. 

- Diederik P. Kingma and Jimmy Ba. 2017. Adam: A method for stochastic optimization. 

- Takumitsu Kudo. 2005. Mecab : Yet another part-ofspeech and morphological analyzer. 

- Surafel M. Lakew, Matteo Negri, and Marco Turchi. 2020. Low resource neural machine translation: A benchmark for five african languages. 

- Guillaume Lample and Alexis Conneau. 2019. Crosslingual language model pretraining. 

- Mike Lewis, Marjan Ghazvininejad, Gargi Ghosh, Armen Aghajanyan, Sida Wang, and Luke Zettlemoyer. 2020. Pre-training via paraphrasing. 

- Haoran Li, Abhinav Arora, Shuohui Chen, Anchit Gupta, Sonal Gupta, and Yashar Mehdad. 2021. MTOP: A comprehensive multilingual task-oriented semantic parsing benchmark. In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_ , pages 2950–2962, Online. Association for Computational Linguistics. 

- Liam Li, Kevin G. Jamieson, Afshin Rostamizadeh, Ekaterina Gonina, Moritz Hardt, Benjamin Recht, and Ameet S. Talwalkar. 2018a. Massively parallel hyperparameter tuning. _ArXiv_ , abs/1810.05934. 

- Xiujun Li, Yu Wang, Siqi Sun, Sarah Panda, Jingjing Liu, and Jianfeng Gao. 2018b. Microsoft dialogue challenge: Building end-to-end task-completion dialogue systems. 

- Richard Liaw, Eric Liang, Robert Nishihara, Philipp Moritz, Joseph E Gonzalez, and Ion Stoica. 2018. Tune: A research platform for distributed model selection and training. _arXiv preprint arXiv:1807.05118_ . 

- Xingkun Liu, Arash Eshghi, Pawel Swietojanski, and Verena Rieser. 2019a. Benchmarking natural language understanding services for building conversational agents. 

- Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020. Multilingual denoising pretraining for neural machine translation. 

- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019b. Roberta: A robustly optimized bert pretraining approach. 

- Loren Lugosch, Mirco Ravanelli, Patrick Ignoto, Vikrant Singh Tomar, and Yoshua Bengio. 2019. Speech Model Pre-Training for End-to-End Spoken Language Understanding. In _Proc. Interspeech 2019_ , pages 814–818. 

- Alexandre Magueresse, Vincent Carles, and Evan Heetderks. 2020. Low-resource languages: A review of past work and future challenges. 

4286 

- Vukosi Marivate, Tshephisho Sefara, Vongani Chabalala, Keamogetswe Makhaya, Tumisho Mokgonyane, Rethabile Mokoena, and Abiodun Modupe. 2020. Investigating an approach for low resource language dataset creation, curation and classification: Setswana and sepedi. In _Proceedings of the first workshop on Resources for African Indigenous Languages_ , pages 15–20, Marseille, France. European Language Resources Association (ELRA). 

- Stephen Mayhew, Chen-Tse Tsai, and Dan Roth. 2017. Cheap translation for cross-lingual named entity recognition. In _Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing_ , pages 2536–2545, Copenhagen, Denmark. Association for Computational Linguistics. 

- Massimo Nicosia, Zhongdi Qu, and Yasemin Altun. 2021. Translate & Fill: Improving zero-shot multilingual semantic parsing with synthetic data. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ , pages 3272–3284, Punta Cana, Dominican Republic. Association for Computational Linguistics. 

- Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. 2017. Cross-lingual name tagging and linking for 282 languages. In _ACL_ . 

- Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual BERT? In _Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics_ , pages 4996–5001, Florence, Italy. Association for Computational Linguistics. 

- P. J. Price. 1990. Evaluation of spoken language systems: the ATIS domain. In _Speech and Natural Language: Proceedings of a Workshop Held at Hidden Valley, Pennsylvania, June 24-27,1990_ . 

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. 

- Shahar Ronen, Bruno Gonçalves, Kevin Z. Hu, Alessandro Vespignani, Steven Pinker, and César A. Hidalgo. 2014. Links that speak: The global language network and its association with global fame. _Proceedings of the National Academy of Sciences_ , 111(52):E5616– E5622. 

- Subendhu Rongali, Luca Soldaini, Emilio Monti, and Wael Hamza. 2020. Don’t parse, generate! a sequence to sequence architecture for task-oriented semantic parsing. _Proceedings of The Web Conference 2020_ . 

- Alaa Saade, Alice Coucke, Alexandre Caulier, Joseph Dureau, Adrien Ball, Théodore Bluche, David Leroy, Clément Doumouro, Thibault Gisselbrecht, Francesco Caltagirone, Thibaut Lavril, and Maël Primet. 2019. Spoken language understanding on the edge. 

- Sebastian Schuster, Sonal Gupta, Rushin Shah, and Mike Lewis. 2019. Cross-lingual transfer learning for multilingual task oriented dialog. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 3795–3805, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Gary Simons, editor. 2022. _Ethnologue: Languages of the World_ , twenty-fifth edition. SIL International, Dallas, TX, USA. 

- Heather Simpson, Christopher Cieri, Kazuaki Maeda, Kathryn Baker, and Boyan Onyshkevych. 2008. Human language technology resources for less commonly taught languages: Lessons learned toward creation of basic language resources. _Collaboration: interoperability between people in the creation of language resources for less-resourced languages_ , 7. 

- Karan Singla, Dogan Can, and Shrikanth Narayanan. 2018. A multi-task approach to learning multilingual representations. In _Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 214–220, Melbourne, Australia. Association for Computational Linguistics. 

- Stephanie Strassel and Jennifer Tracey. 2016. LORELEI language packs: Data, tools, and resources for technology development in low resource languages. In _Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16)_ , pages 3273–3280, Portorož, Slovenia. European Language Resources Association (ELRA). 

- Govind Thattai, Gokhan Tur, and Prem Natarajan. 2020. New alexa features: Interactive teaching by customers. 

- Jörg Tiedemann. 2012. Parallel data, tools and interfaces in opus. In _Proceedings of the Eight International Conference on Language Resources and Evaluation (LREC’12)_ , Istanbul, Turkey. European Language Resources Association (ELRA). 

- Gokhan Tur, Dilek Hakkani-Tür, and Larry Heck. 2010. What is left to be understood in atis? In _2010 IEEE Spoken Language Technology Workshop_ , pages 19– 24. IEEE. 

- Gokhan Tur and Renato De Mori. 2011. Spoken language understanding: Systems for extracting semantic information from speech. 

- Shyam Upadhyay, Manaal Faruqui, Gokhan Tür, Hakkani-Tür Dilek, and Larry Heck. 2018. (almost) zero-shot cross-lingual spoken language understanding. In _2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , pages 6034–6038. 

- Chao Wang, Judith Gaspers, Thi Ngoc Quynh Do, and Hui Jiang. 2021. Exploring cross-lingual transfer learning with unsupervised machine translation. In 

4287 

_Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ , pages 2011–2020, Online. Association for Computational Linguistics. 

- Ye-Yi Wang, Li Deng, and Alex Acero. 2005. Spoken language understanding. _IEEE Signal Processing Magazine_ , 22:16–31. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_ , pages 38–45, Online. Association for Computational Linguistics. 

- Jiateng Xie, Zhilin Yang, Graham Neubig, Noah A. Smith, and Jaime Carbonell. 2018. Neural crosslingual named entity recognition with minimal resources. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 369–379, Brussels, Belgium. Association for Computational Linguistics. 

- Weijia Xu, Batool Haider, and Saab Mansour. 2020. End-to-end slot alignment and recognition for cross- 

   - lingual NLU. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 5052–5063, Online. Association for Computational Linguistics. 

- Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mT5: A massively multilingual pre-trained text-to-text transformer. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_ , pages 483–498, Online. Association for Computational Linguistics. 

- David Yarowsky, Grace Ngai, and Richard Wicentowski. 2001. Inducing multilingual text analysis tools via robust projection across aligned corpora. In _Proceedings of the First International Conference on Human Language Technology Research_ . 

- Steve J. Young. 2002. Talking to machines (statistically speaking). In _INTERSPEECH_ . 

- Su Zhu, Zijian Zhao, Tiejun Zhao, Chengqing Zong, and Kai Yu. 2019. Catslu: The 1st chinese audiotextual spoken language understanding challenge. In _2019 International Conference on Multimodal Interaction_ , ICMI ’19, pages 521–525, New York, NY, USA. Association for Computing Machinery. 

4288 

## **A Additional Linguistic Characteristics** 

Additional linguistic characteristics of our languages are given in Table 4. 

## **B The Collection System** 

Screenshots from our collection workflow are given in Figures 1, 2, and 3. 

## **C Hyperparameters** 

The hyperparameter search spaces and the chosen hyperparameters are given in Tables 5 and 6. 

## **D Results for All Languages** 

Results for all languages are given for exact match accuracy in Table 7, intent accuracy in Table 8, and micro-averaged slot-filling F1 in Table 9. 

## **E A summary of model performance on language characteristics** 

We pick our best performing model, mT5 Text-toText, and provide a summary of its performance on different language characteristics in Figures 4 and 5. 

4289 

|Name|Code|WALS|ISO<br>639-3|Family|Subdivision|Script|Order|Politeness|Imperative Morphology|Imperative<br>Hortative|Optative|Prohibitive|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Afrikaans|af-ZA|afr|afr|Indo-European|Germanic|Latin|-|-|-|-|-|-|
|Albanian|sq-AL|alb|aln|Indo-European|Albanian|Latin|SVO|None|singular only|minimal|present|special negative|
|Amharic|am-ET|amh|amh|Afro-Asiatic|Semtic|Ge’ez|SOV|-|singular and plural|neither|-|special imperative and negative|
|Arabic|ar-SA|ams|arb|Afro-Asiatic|Semtic|Arabic|VSO|-|-|-|-|-|
|Armenian|hy-AM|arm|hye|Indo-European|Armenian|Armenian|None|binary|singular and plural|neither|absent|special negative|
|Azerbaijani|az-AZ|aze|azj|Turkic|Oghuz|Latin|SOV|-|-|-|present|-|
|Bengali|bn-BD|ben|ben|Indo-European|Indo-Aryan|Bengali|SOV|-|-|-|-|-|
|Burmese|my-MM|brm|mya|Sino-Tibetan|Lolo-Burmese|Burmese|SOV|avoidance|None|neither|absent|special negative|
|Danish|da-DK|dsh|dan|Indo-European|Germanic|Latin|SVO|binary|number neutral|neither|absent|normal imperative and negative|
|Dutch|nl-NL|dut|nld|Indo-European|Germanic|Latin|None|binary|number neutral|neither|-|normal imperative and negative|
|English|en-US|eng|eng|Indo-European|Germanic|Latin|SVO|None|None|neither|absent|normal imperative and negative|
|Finnish|f-FI|fn|fn|Uralic|Finno-Ugric|Latin|SVO|binary|singular and plural|minimal|absent|special negative|
|French|fr-FR|fre|fra|Indo-European|Romance|Latin|SVO|binary|singular only|neither|absent|normal imperative and negative|
|Georgian|ka-GE|geo|kat|Kartvelian|Karto-Zan|Georgian|SOV|binary|None|neither|present|-|
|German|de-DE|ger|deu|Indo-European|Germanic|Latin|None|binary|singular only|neither|absent|normal imperative and negative|
|Greek|el-GR|grk|ell|Indo-European|Hellenic|Greek|None|binary|singular and plural|minimal|absent|special imperative and negative|
|Hebrew|he-IL|heb|heb|Afro-Asiatic|Semtic|Hebrew|SVO|None|singular and plural|minimal|absent|special imperative and negative|
|Hindi|hi-IN|hin|hin|Indo-European|Indo-Aryan|Devanagari|SOV|multiple|singular and plural|neither|absent|special negative|
|Hungarian|hu-HU|hun|hun|Uralic|Finno-Ugric|Latin|None|multiple|None|minimal|absent|special negative|
|Icelandic|is-IS|ice|isl|Indo-European|Germanic|Latin|SVO|-|singular only|neither|absent|normal imperative and negative|
|Indonesian|id-ID|ind|ind|Austronesian|Malayic|Latin|SVO|avoidance|None|neither|absent|special negative|
|Italian|it-IT|ita|ita|Indo-European|Romance|Latin|SVO|binary|singular only|neither|-|special imperative|
|Japanese|ja-JP|jpn|jpn|Japonic|Japanese|Japanese|SOV|avoidance|number neutral|neither|absent|special negative|
|Javanese|jv-ID|jav|jav|Austronesian|Javanese|Latin|-|-|-|neither|-|-|
|Kannada|kn-IN|knd|kan|Dravidian|Southern|Kannada|SOV|multiple|singular and plural|minimal|absent|special imperative and negative|
|Khmer|km-KH|khm|khm|Austoasiatic|Khmeric|Khmer|SVO|avoidance|None|-|absent|special negative|
|Korean|ko-KR|kor|kor|Koreanic|Korean|Hangul|SOV|avoidance|number neutral|neither|absent|special negative|
|Latvian|lv-LV|lat|lav|Indo-European|Baltic|Latin|SVO|binary|plural only|neither|absent|normal imperative and negative|
|Malay|ms-MY|mly|zsm|Austronesian|Malayic|Latin|-|-|-|-|-|-|
|Malayalam|ml-IN|mym|mal|Dravidian|Southern|Malayalam|SOV|multiple|singular and plural|neither|absent|special negative|
|Mandarin (simp)|zh-CN|mnd|cmn|Sino-Tibetan|Sinitic|Simp Chinese|SVO|binary|None|neither|absent|special negative|
|Mandarin (trad)|zh-TW|mnd|cmn|Sino-Tibetan|Sinitic|Trad Chinese|SVO|binary|None|neither|absent|special negative|
|Mongolian|mn-MN|-|mon|Mongolic|Mongolian|Cyrillic|-|-|-|-|-|-|
|Norwegian|nb-NO|nor|nob|Indo-European|Germanic|Latin|SVO|binary|number neutral|neither|absent|normal imperative and negative|
|Persian|fa-IR|prs|pes|Indo-European|Indo-Iranian|Arabic|SOV|binary|singular only|maximal|absent|normal imperative and negative|
|Polish|pl-PL|pol|pol|Indo-European|Slavic|Latin|SVO|binary|singular and plural|neither|-|normal imperative and negative|
|Portuguese|pt-PT|por|por|Indo-European|Romance|Latin|SVO|binary|singular only|neither|-|special imperative|
|Romanian|ro-RO|rom|ron|Indo-European|Romance|Latin|SVO|multiple|singular only|minimal|-|special imperative|
|Russian|ru-RU|rus|rus|Indo-European|Slavic|Cyrillic|SVO|binary|singular and plural|neither|absent|normal imperative and negative|
|Slovenian|sl-SI|slo|slv|Indo-European|Slavic|Latin|SVO|-|singular and plural|neither|absent|normal imperative and negative|
|Spanish|es-ES|spa|spa|Indo-European|Romance|Latin|SVO|binary|singular and plural|neither|absent|special imperative|
|Swahili|sw-KE|swa|swh|Niger-Congo|Bantu|Latin|SVO|None|singular and plural|minimal|absent|special imperative and negative|
|Swedish|sv-SE|swe|swe|Indo-European|Germanic|Latin|SVO|binary|number neutral|neither|absent|normal imperative and negative|
|Tagalog|tl-PH|tag|tgl|Austronesian|Philippine|Latin|VSO|multiple|singular and plural|neither|present|special negative|
|Tamil|ta-IN|tml|tam|Dravidian|Southern|Tamil|SOV|multiple|singular and plural|-|-|special imperative and negative|
|Telugu|te-IN|tel|tel|Dravidian|South-Central|Telugu|SOV|-|singular and plural|-|absent|special negative|
|Thai|th-TH|tha|tha|Kra-Dai|Tai|Thai|SVO|avoidance|None|neither|absent|special negative|
|Turkish|tr-TR|tur|tur|Turkic|Oghuz|Latin|SOV|binary|singular and plural|minimal|absent|normal imperative and negative|
|Urdu|ur-PK|urd|urd|Indo-European|Indo-Aryan|Arabic|SOV|multiple|-|-|absent|-|
|Vietnamese|vi-VN|vie|vie|Austoasiatic|Vietic|Latin|SVO|avoidance|None|neither|absent|special negative|
|Welsh|cy-GB|wel|cym|Indo-European|Celtic|Latin|VSO|binary|singular and plural|neither|-|special negative|



Table 4: Additional linguistic characteristics of the MASSIVE languages. 

4290 

**==> picture [455 x 330] intentionally omitted <==**

Figure 1: Slot localization task as presented to Amazon MTurk worker. 

4291 

**==> picture [455 x 346] intentionally omitted <==**

Figure 2: Phrase localization task as presented to Amazon MTurk worker. 

4292 

**==> picture [455 x 610] intentionally omitted <==**

Figure 3: Judgment task as presented to Amazon MTurk worker. 

4293 

||XLM-R Base|mT5 Text-to-Text|mT5 Encoder-Only|
|---|---|---|---|
|Adam_β_1|[0.8, 0.9, 0.99]|[0.8, 0.9, 0.99]|[0.8, 0.9, 0.99]|
||choice|choice|choice|
||0.9|0.8|0.8|
|Adam_β_2|[0.95, 0.99, 0.999, 0.9999]|[0.95, 0.99, 0.999, 0.9999]|[0.95, 0.99, 0.999, 0.9999]|
||choice|choice|choice|
||0.9999|0.9999|0.999|
|Adam_ϵ_|[1e-06, 1e-07, 1e-08, 1e-09]|[1e-06, 1e-07, 1e-08, 1e-09]|[1e-06, 1e-07, 1e-08, 1e-09]|
||choice|choice|choice|
||1e-08|1e-09|1e-09|
|Batch Size|[32, 64, 128, 256, 512, 1024]|[8, 16, 32, 64]||
||choice|choice||
||1024|64||
|Dropout, Attention|[0.0, 0.5, 0.05]||[0.0, 0.5, 0.05]|
||quniform||quniform|
||0.0||0.45|
|Dropout, Feedforward|[0.0, 0.5, 0.05]|[0.0, 0.5, 0.05]|[0.0, 0.5, 0.05]|
||quniform|quniform|quniform|
||0.45|0.05|0.25|
|Encoder Layer Used|[7, 8, 9, 10, 11]||[7, 8, 9, 10, 11]|
||choice||choice|
||11||9|
|Generation Num Beams||[1, 2, 3]||
|||choice||
|||2||
|Gradient Accumulation Steps|||[4, 8, 16, 32, 64]|
||||choice|
||||64|
|Hid Dim Class Head|[256, 512, 728, 1024, 2048]||[256, 512, 728, 1024, 2048]|
||choice||choice|
||2048||1024|
|Intent Class Pooling|[frst, max, mean]||[frst, max, mean]|
||choice||choice|
||max||frst|
|LR Scheduler|[linear, constant_with_warmup]|[linear, constant_with_warmup]|[linear, constant_with_warmup]|
||choice|choice|choice|
||constant_with_warmup|linear|constant_with_warmup|
|Learning Rate|[1e-07, 0.0001, 1e-07]|[1e-07, 0.001, 1e-07]|[1e-07, 0.001, 1e-07]|
||qloguniform|qloguniform|qloguniform|
||2.8e-05|8e-05|0.0003525|
|Num Layers Class Head|[0, 1, 2, 3]||[0, 1, 2, 3]|
||choice||choice|
||1||1|
|Slot Loss Coeffcient|[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]||[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]|
||choice||choice|
||4.0||4.0|
|Tot Epochs, LR Sched|[3, 30, 1]|[3, 30, 1]|[3, 30, 1]|
||quniform|quniform|quniform|
||26|22|15|
|Warmup Steps|[0, 1000, 100]|[0, 1000, 100]|[0, 1000, 100]|
||quniform|quniform|quniform|
||800|200|600|
|Weight Decay|[0.0, 0.5, 0.01]|[0.0, 0.5, 0.01]|[0.0, 0.5, 0.01]|
||quniform|quniform|quniform|
||0.21|0.16|0.07|



Table 5: The full-dataset hyperparameter search space, the sampling technique, and the chosen hyperparameter for our 3 models. The search space for the “quniform” and “qloguniform” sampling techniques is given as [min, max, increment]. 

4294 

||XLM-R Base|mT5 Text-to-Text|mT5 Encoder-Only|
|---|---|---|---|
|Adam_β_1|[0.8, 0.9, 0.99]|[0.8, 0.9, 0.99]|[0.8, 0.9, 0.99]|
||choice|choice|choice|
||0.99|0.8|0.8|
|Adam_β_2|[0.95, 0.99, 0.999, 0.9999]|[0.95, 0.99, 0.999, 0.9999]|[0.95, 0.99, 0.999, 0.9999]|
||choice|choice|choice|
||0.9999|0.999|0.9999|
|Adam_ϵ_|[1e-06, 1e-07, 1e-08, 1e-09]|[1e-06, 1e-07, 1e-08, 1e-09]|[1e-06, 1e-07, 1e-08, 1e-09]|
||choice|choice|choice|
||1e-09|1e-09|1e-08|
|Batch Size||||
|Dropout, Attention|[0.0, 0.5, 0.05]||[0.0, 0.5, 0.05]|
||quniform||quniform|
||0.35||0.4|
|Dropout, Feedforward|[0.0, 0.5, 0.05]|[0.0, 0.5, 0.05]|[0.0, 0.5, 0.05]|
||quniform|quniform|quniform|
||0.25|0.2|0.2|
|Encoder Layer Used|[7, 8, 9, 10, 11]||[7, 8, 9, 10, 11]|
||choice||choice|
||10||8|
|Freeze Layers|[xlmr.embeddings.word_embeddings.weight,|[shared.weight,|[mt5.shared.weight,|
||null]|shared.weight + lm_head.weight, null]|null]|
||choice|choice|choice|
||xlmr.embeddings.word_embeddings.weight|null|mt5.shared.weight|
|Generation Num Beams||[1, 2, 3]||
|||choice||
|||3||
|Gradient Accumulation Steps|[1, 2, 4, 8, 16, 32]|[4, 8, 16, 32, 64]|[4, 8, 16, 32, 64]|
||choice|choice|choice|
||8|64|32|
|Hid Dim Class Head|[728, 1024, 2048, 3072, 4096, 8192, 16384]||[256, 512, 728, 1024, 2048]|
||choice||choice|
||8192||2048|
|Intent Class Pooling|[frst, max, mean]||[frst, max, mean]|
||choice||choice|
||max||mean|
|LR Scheduler|[linear, constant_with_warmup]|[linear, constant_with_warmup]|[linear, constant_with_warmup]|
||choice|choice|choice|
||constant_with_warmup|linear|linear|
|Learning Rate|[1e-07, 0.0001, 1e-07]|[1e-07, 0.001, 1e-07]|[1e-07, 0.001, 1e-07]|
||qloguniform|qloguniform|qloguniform|
||4.7e-06|3.4e-05|6.19e-05|
|Num Layers Class Head|[0, 1, 2, 3]||[0, 1, 2, 3]|
||choice||choice|
||2||3|
|Slot Loss Coeffcient|[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]||[0.5, 1.0, 2.0, 4.0, 8.0, 16.0]|
||choice||choice|
||2.0||4.0|
|Tot Epochs, LR Sched|[50, 1500, 50]|[50, 1500, 50]|[30, 1500, 10]|
||quniform|quniform|quniform|
||850|950|300|
|Warmup Steps|[0, 1000, 100]|[0, 1000, 100]|[0, 1000, 100]|
||quniform|quniform|quniform|
||500|300|700|
|Weight Decay|[0.0, 0.5, 0.01]|[0.0, 0.5, 0.01]|[0.0, 0.5, 0.01]|
||quniform|quniform|quniform|
||0.11|0.0|0.35|



Table 6: The zero-shot hyperparameter search space, the sampling technique, and the chosen hyperparameter for our 3 models. The search space for the “quniform” and “qloguniform” sampling techniques is given as [min, max, increment]. 

4295 

|||Exact Match Accuracy (%)|Exact Match Accuracy (%)|Exact Match Accuracy (%)|||
|---|---|---|---|---|---|---|
||mT5 T2T Full|mT5 Enc Full|XLM-R Full|mT5 T2T Zero|mT5 Enc Zero|XLM-R Zero|
|th-TH|73.4_±_1.6|72.3_±_1.6|70.1_±_1.6|33.5_±_1.7|40.8_±_1.8|46.3_±_1.8|
|en-US|72.5_±_1.6|72.0_±_1.6|69.7_±_1.7||||
|sv-SE|71.2_±_1.6|70.6_±_1.6|69.7_±_1.7|53.2_±_1.8|44.3_±_1.8|57.9_±_1.8|
|da-DK|70.2_±_1.6|70.3_±_1.6|68.2_±_1.7|47.6_±_1.8|41.0_±_1.8|54.4_±_1.8|
|my-MM|70.1_±_1.6|69.4_±_1.7|65.5_±_1.7|24.4_±_1.5|22.2_±_1.5|33.1_±_1.7|
|nb-NO|70.0_±_1.6|68.8_±_1.7|66.8_±_1.7|48.5_±_1.8|41.0_±_1.8|53.7_±_1.8|
|nl-NL|69.4_±_1.7|68.1_±_1.7|66.6_±_1.7|52.4_±_1.8|41.0_±_1.8|51.7_±_1.8|
|ru-RU|69.2_±_1.7|67.2_±_1.7|66.2_±_1.7|50.5_±_1.8|42.6_±_1.8|52.8_±_1.8|
|f-FI|69.1_±_1.7|68.8_±_1.7|66.9_±_1.7|41.3_±_1.8|35.8_±_1.7|49.8_±_1.8|
|ms-MY|69.1_±_1.7|67.3_±_1.7|65.6_±_1.7|39.3_±_1.8|33.1_±_1.7|45.5_±_1.8|
|de-DE|69.0_±_1.7|68.9_±_1.7|65.7_±_1.7|52.0_±_1.8|40.0_±_1.8|45.4_±_1.8|
|ko-KR|68.8_±_1.7|68.0_±_1.7|67.5_±_1.7|33.7_±_1.7|24.1_±_1.5|44.8_±_1.8|
|ro-RO|68.6_±_1.7|65.1_±_1.7|64.5_±_1.7|45.4_±_1.8|35.7_±_1.7|51.6_±_1.8|
|id-ID|68.6_±_1.7|67.2_±_1.7|64.8_±_1.7|46.0_±_1.8|37.4_±_1.7|50.7_±_1.8|
|af-ZA|68.3_±_1.7|66.8_±_1.7|64.9_±_1.7|39.9_±_1.8|34.9_±_1.7|43.9_±_1.8|
|tr-TR|68.1_±_1.7|67.7_±_1.7|65.2_±_1.7|37.2_±_1.7|27.4_±_1.6|43.8_±_1.8|
|el-GR|67.8_±_1.7|66.7_±_1.7|64.0_±_1.7|43.5_±_1.8|36.8_±_1.7|41.9_±_1.8|
|pt-PT|67.6_±_1.7|66.0_±_1.7|64.6_±_1.7|47.6_±_1.8|39.8_±_1.8|48.6_±_1.8|
|hu-HU|67.2_±_1.7|67.7_±_1.7|65.4_±_1.7|38.7_±_1.8|33.7_±_1.7|44.7_±_1.8|
|az-AZ|67.2_±_1.7|66.2_±_1.7|65.2_±_1.7|28.3_±_1.6|20.2_±_1.4|37.2_±_1.7|
|is-IS|67.1_±_1.7|66.8_±_1.7|64.3_±_1.7|28.5_±_1.6|23.4_±_1.5|32.7_±_1.7|
|ml-IN|67.1_±_1.7|67.2_±_1.7|64.9_±_1.7|32.5_±_1.7|27.2_±_1.6|40.1_±_1.8|
|lv-LV|67.0_±_1.7|67.0_±_1.7|66.6_±_1.7|34.3_±_1.7|27.4_±_1.6|37.8_±_1.7|
|it-IT|66.8_±_1.7|64.8_±_1.7|63.1_±_1.7|45.1_±_1.8|38.1_±_1.7|45.2_±_1.8|
|all|66.6_±_0.2|65.9_±_0.2|63.7_±_0.2|34.7_±_0.2|28.8_±_0.2|38.7_±_0.2|
|jv-ID|66.6_±_1.7|65.4_±_1.7|59.3_±_1.8|19.0_±_1.4|15.3_±_1.3|11.7_±_1.2|
|sq-AL|66.5_±_1.7|65.1_±_1.7|63.6_±_1.7|35.5_±_1.7|28.9_±_1.6|35.1_±_1.7|
|he-IL|66.2_±_1.7|65.9_±_1.7|64.5_±_1.7|28.1_±_1.6|26.6_±_1.6|37.8_±_1.7|
|es-ES|66.2_±_1.7|64.3_±_1.7|62.8_±_1.7|50.4_±_1.8|39.7_±_1.8|47.6_±_1.8|
|fr-FR|66.2_±_1.7|65.1_±_1.7|62.2_±_1.7|47.2_±_1.8|39.5_±_1.8|48.6_±_1.8|
|bn-BD|66.2_±_1.7|66.0_±_1.7|63.4_±_1.7|27.3_±_1.6|21.6_±_1.5|36.3_±_1.7|
|hy-AM|66.1_±_1.7|65.8_±_1.7|63.1_±_1.7|34.8_±_1.7|26.3_±_1.6|39.0_±_1.8|
|mn-MN|66.0_±_1.7|65.4_±_1.7|63.4_±_1.7|24.3_±_1.5|16.4_±_1.3|33.3_±_1.7|
|fa-IR|65.9_±_1.7|67.3_±_1.7|67.0_±_1.7|38.7_±_1.8|31.5_±_1.7|49.6_±_1.8|
|sl-SL|65.9_±_1.7|65.6_±_1.7|64.3_±_1.7|36.3_±_1.7|29.9_±_1.6|38.4_±_1.7|
|tl-PH|65.6_±_1.7|65.6_±_1.7|61.1_±_1.8|34.3_±_1.7|26.9_±_1.6|26.9_±_1.6|
|hi-IN|65.4_±_1.7|64.7_±_1.7|63.4_±_1.7|35.1_±_1.7|29.4_±_1.6|42.6_±_1.8|
|km-KH|65.1_±_1.7|65.0_±_1.7|60.5_±_1.8|24.9_±_1.6|34.7_±_1.7|35.3_±_1.7|
|vi-VN|65.0_±_1.7|64.5_±_1.7|64.5_±_1.7|26.8_±_1.6|23.9_±_1.5|44.1_±_1.8|
|cy-GB|64.9_±_1.7|63.3_±_1.7|60.1_±_1.8|10.0_±_1.1|8.3_±_1.0|17.1_±_1.4|
|zh-CN|64.8_±_1.7|62.9_±_1.7|60.4_±_1.8|25.0_±_1.6|14.1_±_1.3|17.7_±_1.4|
|pl-PL|64.4_±_1.7|64.0_±_1.7|60.9_±_1.8|45.9_±_1.8|39.9_±_1.8|49.1_±_1.8|
|ar-SA|64.1_±_1.7|63.4_±_1.7|61.2_±_1.8|29.6_±_1.6|28.7_±_1.6|31.2_±_1.7|
|ur-PK|64.0_±_1.7|62.4_±_1.7|59.0_±_1.8|24.0_±_1.5|19.3_±_1.4|30.5_±_1.7|
|ta-IN|63.8_±_1.7|63.5_±_1.7|61.4_±_1.7|34.3_±_1.7|27.9_±_1.6|37.0_±_1.7|
|te-IN|63.8_±_1.7|65.3_±_1.7|62.2_±_1.7|28.3_±_1.6|22.5_±_1.5|36.6_±_1.7|
|ka-GE|63.6_±_1.7|63.5_±_1.7|62.8_±_1.7|32.5_±_1.7|30.5_±_1.7|36.8_±_1.7|
|am-ET|63.4_±_1.7|63.0_±_1.7|59.3_±_1.8|16.1_±_1.3|12.0_±_1.2|23.8_±_1.5|
|sw-KE|63.3_±_1.7|63.3_±_1.7|58.5_±_1.8|17.1_±_1.4|15.2_±_1.3|13.9_±_1.2|
|kn-IN|62.8_±_1.7|62.3_±_1.7|59.4_±_1.8|30.3_±_1.7|21.7_±_1.5|33.4_±_1.7|
|zh-TW|61.0_±_1.8|59.2_±_1.8|58.0_±_1.8|27.4_±_1.6|15.3_±_1.3|18.1_±_1.4|
|ja-JP|58.3_±_1.8|57.8_±_1.8|55.8_±_1.8|9.4_±_1.0|4.2_±_0.7|9.8_±_1.1|



Table 7: Exact match accuracy by language for our three models using the full dataset and the zero-shot setup. 

4296 

||||Intent Accuracy|(%)|||
|---|---|---|---|---|---|---|
||mT5 T2T Full|mT5 Enc Full|XLM-R Full|mT5 T2T Zero|mT5 Enc Zero|XLM-R Zero|
|en-US|87.9_±_1.2|89.0_±_1.1|88.3_±_1.2||||
|sv-SE|87.8_±_1.2|88.5_±_1.1|87.9_±_1.2|77.1_±_1.5|76.0_±_1.5|85.2_±_1.3|
|nb-NO|87.6_±_1.2|87.7_±_1.2|87.3_±_1.2|76.3_±_1.5|72.8_±_1.6|83.6_±_1.3|
|da-DK|87.5_±_1.2|88.0_±_1.2|86.9_±_1.2|76.8_±_1.5|73.4_±_1.6|83.1_±_1.3|
|ro-RO|87.2_±_1.2|87.0_±_1.2|86.9_±_1.2|73.0_±_1.6|70.1_±_1.6|80.8_±_1.4|
|nl-NL|87.2_±_1.2|87.6_±_1.2|86.8_±_1.2|79.9_±_1.4|76.4_±_1.5|82.1_±_1.4|
|ru-RU|87.0_±_1.2|86.8_±_1.2|87.2_±_1.2|76.2_±_1.5|73.8_±_1.6|81.3_±_1.4|
|id-ID|87.0_±_1.2|86.8_±_1.2|87.1_±_1.2|77.0_±_1.5|74.1_±_1.6|83.1_±_1.3|
|fr-FR|86.9_±_1.2|87.2_±_1.2|86.3_±_1.2|76.9_±_1.5|74.1_±_1.6|80.8_±_1.4|
|it-IT|86.8_±_1.2|87.6_±_1.2|86.6_±_1.2|72.3_±_1.6|71.5_±_1.6|76.4_±_1.5|
|ms-MY|86.8_±_1.2|86.9_±_1.2|86.1_±_1.2|69.9_±_1.6|66.0_±_1.7|76.7_±_1.5|
|es-ES|86.7_±_1.2|86.8_±_1.2|86.9_±_1.2|76.6_±_1.5|75.9_±_1.5|78.8_±_1.5|
|pt-PT|86.7_±_1.2|86.9_±_1.2|86.7_±_1.2|74.0_±_1.6|74.5_±_1.6|79.5_±_1.5|
|fa-IR|86.3_±_1.2|87.2_±_1.2|87.0_±_1.2|69.0_±_1.7|66.3_±_1.7|81.1_±_1.4|
|pl-PL|86.3_±_1.2|87.1_±_1.2|85.8_±_1.3|76.4_±_1.5|74.1_±_1.6|80.7_±_1.4|
|de-DE|86.2_±_1.2|86.8_±_1.2|85.7_±_1.3|77.3_±_1.5|73.9_±_1.6|77.6_±_1.5|
|az-AZ|86.2_±_1.2|86.4_±_1.2|86.2_±_1.2|57.0_±_1.8|55.5_±_1.8|70.9_±_1.6|
|tr-TR|86.1_±_1.2|87.1_±_1.2|86.3_±_1.2|66.5_±_1.7|63.7_±_1.7|78.4_±_1.5|
|ko-KR|86.1_±_1.2|86.4_±_1.2|86.5_±_1.2|60.0_±_1.8|61.9_±_1.7|77.0_±_1.5|
|af-ZA|86.0_±_1.2|86.9_±_1.2|85.6_±_1.3|68.5_±_1.7|66.5_±_1.7|71.7_±_1.6|
|ml-IN|86.0_±_1.2|86.5_±_1.2|85.1_±_1.3|60.6_±_1.8|57.8_±_1.8|70.1_±_1.6|
|sq-AL|85.9_±_1.3|86.4_±_1.2|86.4_±_1.2|62.9_±_1.7|62.0_±_1.7|67.6_±_1.7|
|sl-SL|85.9_±_1.3|86.8_±_1.2|86.3_±_1.2|61.5_±_1.7|59.8_±_1.8|69.5_±_1.7|
|el-GR|85.8_±_1.3|86.6_±_1.2|86.2_±_1.2|71.9_±_1.6|69.8_±_1.6|74.0_±_1.6|
|vi-VN|85.8_±_1.3|87.2_±_1.2|86.3_±_1.2|64.2_±_1.7|62.7_±_1.7|79.2_±_1.5|
|hi-IN|85.6_±_1.3|86.2_±_1.2|85.8_±_1.3|62.4_±_1.7|59.3_±_1.8|74.8_±_1.6|
|hu-HU|85.4_±_1.3|86.9_±_1.2|86.2_±_1.2|68.0_±_1.7|66.4_±_1.7|77.1_±_1.5|
|all|85.3_±_0.2|86.1_±_0.2|85.1_±_0.2|62.9_±_0.2|61.2_±_0.2|70.6_±_0.2|
|is-IS|85.3_±_1.3|85.9_±_1.3|85.3_±_1.3|59.0_±_1.8|55.9_±_1.8|66.7_±_1.7|
|f-FI|85.3_±_1.3|86.7_±_1.2|85.5_±_1.3|69.7_±_1.7|68.5_±_1.7|80.2_±_1.4|
|zh-CN|85.2_±_1.3|85.8_±_1.3|84.9_±_1.3|55.7_±_1.8|51.6_±_1.8|61.9_±_1.7|
|lv-LV|85.2_±_1.3|86.6_±_1.2|86.1_±_1.2|61.0_±_1.8|60.0_±_1.8|69.2_±_1.7|
|th-TH|85.2_±_1.3|85.5_±_1.3|84.7_±_1.3|72.8_±_1.6|69.6_±_1.7|77.4_±_1.5|
|tl-PH|85.1_±_1.3|87.0_±_1.2|84.6_±_1.3|64.9_±_1.7|63.8_±_1.7|63.7_±_1.7|
|mn-MN|84.9_±_1.3|86.0_±_1.2|84.3_±_1.3|47.8_±_1.8|47.2_±_1.8|64.4_±_1.7|
|kn-IN|84.9_±_1.3|84.9_±_1.3|84.0_±_1.3|56.7_±_1.8|51.8_±_1.8|63.5_±_1.7|
|te-IN|84.9_±_1.3|85.5_±_1.3|84.5_±_1.3|55.0_±_1.8|52.2_±_1.8|68.2_±_1.7|
|bn-BD|84.8_±_1.3|85.8_±_1.3|84.1_±_1.3|56.5_±_1.8|52.1_±_1.8|66.0_±_1.7|
|he-IL|84.6_±_1.3|86.2_±_1.2|85.9_±_1.3|64.7_±_1.7|64.0_±_1.7|73.2_±_1.6|
|my-MM|84.6_±_1.3|85.2_±_1.3|83.6_±_1.3|58.3_±_1.8|58.4_±_1.8|67.6_±_1.7|
|jv-ID|84.5_±_1.3|85.3_±_1.3|82.9_±_1.4|47.6_±_1.8|49.3_±_1.8|46.5_±_1.8|
|hy-AM|84.5_±_1.3|85.6_±_1.3|84.4_±_1.3|63.8_±_1.7|62.2_±_1.7|71.6_±_1.6|
|ta-IN|84.4_±_1.3|85.2_±_1.3|83.5_±_1.3|61.3_±_1.8|58.0_±_1.8|68.1_±_1.7|
|ur-PK|84.3_±_1.3|85.1_±_1.3|83.2_±_1.3|47.2_±_1.8|49.0_±_1.8|65.6_±_1.7|
|sw-KE|84.0_±_1.3|85.8_±_1.3|83.1_±_1.3|45.6_±_1.8|44.7_±_1.8|46.6_±_1.8|
|cy-GB|83.7_±_1.3|84.9_±_1.3|82.6_±_1.4|29.6_±_1.6|33.1_±_1.7|46.9_±_1.8|
|ja-JP|83.5_±_1.3|85.8_±_1.3|83.9_±_1.3|25.7_±_1.6|27.1_±_1.6|44.8_±_1.8|
|zh-TW|82.9_±_1.4|83.8_±_1.3|83.0_±_1.3|56.1_±_1.8|52.2_±_1.8|60.4_±_1.8|
|am-ET|82.7_±_1.4|84.2_±_1.3|81.7_±_1.4|36.8_±_1.7|36.6_±_1.7|51.9_±_1.8|
|ar-SA|81.8_±_1.4|82.2_±_1.4|80.7_±_1.4|59.0_±_1.8|58.8_±_1.8|62.8_±_1.7|
|ka-GE|79.9_±_1.4|81.3_±_1.4|80.3_±_1.4|59.3_±_1.8|58.4_±_1.8|61.2_±_1.8|
|km-KH|79.0_±_1.5|79.1_±_1.5|77.2_±_1.5|60.2_±_1.8|58.7_±_1.8|61.3_±_1.8|



Table 8: Intent accuracy by language for our three models using the full dataset and the zero-shot setup. 

4297 

|||Micro-Averaged Slot F1(%)|Micro-Averaged Slot F1(%)|Micro-Averaged Slot F1(%)|||
|---|---|---|---|---|---|---|
||mT5 T2T Full|mT5 Enc Full|XLM-R Full|mT5 T2T Zero|mT5 Enc Zero|XLM-R Zero|
|th-TH|86.8_±_0.7|85.7_±_0.7|83.5_±_0.7|34.5_±_0.9|59.5_±_1.0|57.4_±_1.0|
|my-MM|82.2_±_0.7|82.1_±_0.7|79.0_±_0.7|26.0_±_0.8|38.0_±_0.9|48.9_±_0.9|
|en-US|81.6_±_0.5|80.4_±_0.5|78.7_±_0.6||||
|km-KH|81.0_±_0.8|81.9_±_0.8|77.9_±_0.8|27.9_±_0.9|58.2_±_1.0|53.6_±_1.0|
|sv-SE|80.9_±_0.6|79.6_±_0.6|78.5_±_0.6|64.2_±_0.7|56.8_±_0.7|68.4_±_0.7|
|nb-NO|80.0_±_0.6|77.8_±_0.6|76.0_±_0.6|58.8_±_0.7|56.0_±_0.7|65.1_±_0.7|
|ko-KR|79.6_±_0.7|78.9_±_0.7|77.8_±_0.7|46.8_±_0.8|36.0_±_0.8|56.0_±_0.8|
|da-DK|79.4_±_0.6|79.1_±_0.6|77.7_±_0.6|58.5_±_0.7|54.6_±_0.7|64.6_±_0.7|
|f-FI|79.4_±_0.7|79.2_±_0.7|77.2_±_0.7|49.1_±_0.8|48.9_±_0.8|62.1_±_0.8|
|de-DE|78.8_±_0.6|78.6_±_0.6|76.2_±_0.6|64.3_±_0.7|55.6_±_0.7|60.0_±_0.7|
|ru-RU|78.7_±_0.6|76.3_±_0.6|74.9_±_0.6|61.6_±_0.7|55.4_±_0.7|63.3_±_0.7|
|ms-MY|78.4_±_0.6|77.4_±_0.6|75.5_±_0.6|51.5_±_0.7|48.2_±_0.7|55.9_±_0.7|
|af-ZA|78.3_±_0.6|76.5_±_0.6|74.6_±_0.6|51.9_±_0.7|52.3_±_0.7|57.3_±_0.7|
|is-IS|78.2_±_0.6|77.7_±_0.6|75.2_±_0.6|39.3_±_0.7|37.9_±_0.7|45.2_±_0.7|
|nl-NL|78.1_±_0.6|76.5_±_0.6|75.5_±_0.6|61.6_±_0.7|54.3_±_0.7|62.4_±_0.7|
|jv-ID|78.1_±_0.6|76.1_±_0.6|70.9_±_0.7|29.6_±_0.7|26.7_±_0.7|24.7_±_0.6|
|hu-HU|78.0_±_0.6|77.5_±_0.6|75.3_±_0.6|46.1_±_0.7|45.8_±_0.7|56.8_±_0.7|
|tr-TR|77.9_±_0.6|76.1_±_0.7|74.9_±_0.7|48.8_±_0.8|41.9_±_0.8|52.8_±_0.8|
|lv-LV|77.8_±_0.6|77.1_±_0.6|76.3_±_0.6|47.2_±_0.8|41.6_±_0.7|53.0_±_0.8|
|ka-GE|77.6_±_0.7|77.1_±_0.7|76.8_±_0.7|43.5_±_0.9|48.6_±_0.9|55.9_±_0.9|
|ro-RO|77.6_±_0.6|74.1_±_0.6|72.4_±_0.6|56.3_±_0.7|48.6_±_0.7|60.8_±_0.7|
|el-GR|77.0_±_0.6|75.5_±_0.6|73.4_±_0.6|54.8_±_0.7|51.7_±_0.7|54.4_±_0.7|
|id-ID|76.9_±_0.6|75.6_±_0.6|73.6_±_0.6|55.6_±_0.7|51.0_±_0.7|59.7_±_0.7|
|all|76.8_±_0.1|75.4_±_0.1|73.6_±_0.1|44.8_±_0.1|41.6_±_0.1|50.3_±_0.1|
|az-AZ|76.8_±_0.6|75.6_±_0.7|74.1_±_0.7|40.4_±_0.7|33.8_±_0.7|46.6_±_0.8|
|he-IL|76.7_±_0.6|75.1_±_0.7|74.0_±_0.7|30.6_±_0.7|35.5_±_0.7|49.3_±_0.8|
|pt-PT|76.6_±_0.6|74.9_±_0.6|73.3_±_0.6|56.3_±_0.7|46.6_±_0.7|58.2_±_0.7|
|ml-IN|76.6_±_0.7|76.1_±_0.7|74.8_±_0.7|42.1_±_0.8|45.5_±_0.8|52.5_±_0.8|
|it-IT|76.4_±_0.6|73.7_±_0.6|72.3_±_0.6|58.7_±_0.7|50.0_±_0.7|57.3_±_0.7|
|bn-BD|76.4_±_0.6|75.1_±_0.6|73.4_±_0.6|39.6_±_0.7|37.2_±_0.7|52.3_±_0.7|
|cy-GB|76.3_±_0.6|73.5_±_0.6|71.2_±_0.6|21.8_±_0.6|21.5_±_0.5|30.1_±_0.6|
|sq-AL|75.9_±_0.6|73.7_±_0.6|72.0_±_0.6|48.3_±_0.7|41.9_±_0.7|50.0_±_0.7|
|tl-PH|75.8_±_0.6|74.6_±_0.6|71.6_±_0.6|44.7_±_0.6|37.1_±_0.6|36.1_±_0.6|
|mn-MN|75.8_±_0.6|74.1_±_0.6|73.7_±_0.7|36.6_±_0.7|26.9_±_0.7|45.0_±_0.7|
|ar-SA|75.7_±_0.7|75.4_±_0.7|73.8_±_0.7|39.7_±_0.8|44.6_±_0.8|48.4_±_0.8|
|fr-FR|75.6_±_0.6|73.5_±_0.6|70.9_±_0.6|54.2_±_0.7|51.2_±_0.7|59.1_±_0.7|
|es-ES|75.5_±_0.6|72.8_±_0.6|71.0_±_0.6|61.1_±_0.7|50.4_±_0.7|57.1_±_0.7|
|fa-IR|75.4_±_0.6|76.6_±_0.6|76.6_±_0.6|49.4_±_0.7|46.9_±_0.7|60.2_±_0.6|
|sl-SL|75.4_±_0.6|74.3_±_0.6|72.2_±_0.7|49.0_±_0.7|45.6_±_0.7|53.1_±_0.7|
|hy-AM|75.3_±_0.7|74.1_±_0.7|72.4_±_0.7|41.7_±_0.7|39.1_±_0.7|50.0_±_0.8|
|hi-IN|75.0_±_0.6|73.5_±_0.6|72.3_±_0.6|49.6_±_0.7|45.1_±_0.7|54.6_±_0.7|
|zh-CN|74.5_±_0.5|71.2_±_0.5|70.0_±_0.5|33.4_±_0.5|20.9_±_0.5|24.8_±_0.5|
|ta-IN|74.3_±_0.7|72.6_±_0.7|71.8_±_0.7|45.8_±_0.8|45.9_±_0.8|50.3_±_0.8|
|vi-VN|74.2_±_0.5|72.3_±_0.5|73.3_±_0.5|28.8_±_0.5|36.0_±_0.6|53.9_±_0.6|
|am-ET|73.8_±_0.7|73.7_±_0.7|70.0_±_0.7|25.9_±_0.7|21.3_±_0.6|39.0_±_0.8|
|sw-KE|73.8_±_0.6|72.9_±_0.6|68.7_±_0.7|25.9_±_0.6|28.2_±_0.6|27.7_±_0.6|
|te-IN|73.0_±_0.7|74.7_±_0.7|71.4_±_0.7|41.1_±_0.7|39.4_±_0.7|51.6_±_0.7|
|ur-PK|73.0_±_0.6|71.2_±_0.6|68.0_±_0.6|40.1_±_0.6|32.6_±_0.6|41.4_±_0.6|
|zh-TW|72.9_±_0.5|68.8_±_0.6|68.7_±_0.6|34.4_±_0.6|22.6_±_0.5|25.2_±_0.5|
|pl-PL|72.9_±_0.7|71.7_±_0.7|69.0_±_0.7|53.4_±_0.7|49.3_±_0.7|58.0_±_0.7|
|kn-IN|72.2_±_0.7|71.3_±_0.7|69.2_±_0.7|40.4_±_0.8|38.3_±_0.8|47.8_±_0.8|
|ja-JP|67.6_±_0.4|64.5_±_0.4|63.3_±_0.4|13.9_±_0.3|6.3_±_0.2|15.4_±_0.3|



Table 9: Micro-averaged slot-filling F1 by language for our three models using the full dataset and the zero-shot setup. 

4298 

**==> picture [455 x 215] intentionally omitted <==**

**==> picture [455 x 215] intentionally omitted <==**

Figure 4: mT5 Text-to-Text performance grouped by Genus and Subdivision. The categories of each language characteristic are sorted by exact match accuracy for readability. The number of languages falling into each category is provided in the bar chart in the lowest panel for each characteristic. 

4299 

**==> picture [455 x 215] intentionally omitted <==**

**==> picture [455 x 214] intentionally omitted <==**

Figure 5: mT5 Text-to-Text performance grouped by Script, Family, Order, Politeness, Imperative Morphology, Imperative Hortative, Optative and Prohibitive. As with Figure 4, the categories of each language characteristic are sorted by exact match accuracy for readability. The number of languages falling into each category is provided in the bar chart in the lowest panel for each characteristic. 

4300 

## **ACL 2023 Responsible NLP Checklist** 

- **A For every submission:** 

- A1. Did you describe the limitations of your work? _Left blank._ 

- A2. Did you discuss any potential risks of your work? _Left blank._ 

- A3. Do the abstract and introduction summarize the paper’s main claims? _Left blank._ 

- A4. Have you used AI writing assistants when working on this paper? _Left blank._ 

- **B** □ 

- _Left blank._ 

- B1. Did you cite the creators of artifacts you used? _Left blank._ 

- B2. Did you discuss the license or terms for use and / or distribution of any artifacts? _Left blank._ 

- B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use, provided that it was specified? For the artifacts you create, do you specify intended use and whether that is compatible with the original access conditions (in particular, derivatives of data accessed for research purposes should not be used outside of research contexts)? _Left blank._ 

- B4. Did you discuss the steps taken to check whether the data that was collected / used contains any information that names or uniquely identifies individual people or offensive content, and the steps taken to protect / anonymize it? _Left blank._ 

- B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages, and linguistic phenomena, demographic groups represented, etc.? _Left blank._ 

- B6. Did you report relevant statistics like the number of examples, details of train / test / dev splits, etc. for the data that you used / created? Even for commonly-used benchmark datasets, include the number of examples in train / validation / test splits, as these provide necessary context for a reader to understand experimental results. For example, small differences in accuracy on large test sets may be significant, while on small test sets they may not be. _Left blank._ 

## **C** □ **Did you run computational experiments?** 

_Left blank._ 

- C1. Did you report the number of parameters in the models used, the total computational budget (e.g., GPU hours), and computing infrastructure used? _Left blank._ 

_The Responsible NLP Checklist used at ACL 2023 is adopted from NAACL 2022, with the addition of a question on AI writing assistance._ 

4301 

- C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values? _Left blank._ 

- C3. Did you report descriptive statistics about your results (e.g., error bars around results, summary statistics from sets of experiments), and is it transparent whether you are reporting the max, mean, etc. or just a single run? 

   - _Left blank._ 

- C4. If you used existing packages (e.g., for preprocessing, for normalization, or for evaluation), did you report the implementation, model, and parameter settings used (e.g., NLTK, Spacy, ROUGE, etc.)? 

   - _Left blank._ 

- **D** □ **Did you use human annotators (e.g., crowdworkers) or research with human participants?** _Left blank._ 

- D1. Did you report the full text of instructions given to participants, including e.g., screenshots, disclaimers of any risks to participants or annotators, etc.? _Left blank._ 

- D2. Did you report information about how you recruited (e.g., crowdsourcing platform, students) and paid participants, and discuss if such payment is adequate given the participants’ demographic (e.g., country of residence)? _Left blank._ 

- D3. Did you discuss whether and how consent was obtained from people whose data you’re using/curating? For example, if you collected data via crowdsourcing, did your instructions to crowdworkers explain how the data would be used? _Left blank._ 

- D4. Was the data collection protocol approved (or determined exempt) by an ethics review board? _Left blank._ 

- D5. Did you report the basic demographic and geographic characteristics of the annotator population that is the source of the data? _Left blank._ 

4302 

