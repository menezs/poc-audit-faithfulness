# **PORTULAN ExtraGLUE Datasets and Models: Kick-starting a Benchmark for the Neural Processing of Portuguese** 

**Tomás Freitas Osório[1] , Bernardo Leite[1] , Henrique Lopes Cardoso[1] , Luís Gomes[2] , João Rodrigues[2] , Rodrigo Santos[2] , António Branco[2]** 

1Artificial Intelligence and Computer Science Laboratory (LIACC), Faculdade de Engenharia da Universidade do Porto Rua Doutor Roberto Frias, s/n, 4200-465 Porto, Portugal tomas.s.osorio@gmail.com, {bernardo.leite, hlc}@fe.up.pt 

2University of Lisbon NLX—Natural Language and Speech Group, Dept. Informatics Faculdade de Ciências (FCUL), Campo Grande, 1749-016 Lisboa, Portugal {lmdgomes, jarodrigues, rsdsantos, antonio.branco}@fc.ul.pt 

## **Abstract** 

Leveraging research on the neural modelling of Portuguese, we contribute a collection of datasets for an array of language processing tasks and a corresponding collection of fine-tuned neural language models on these downstream tasks. To align with mainstream benchmarks in the literature, originally developed in English, and to kick start their Portuguese counterparts, the datasets were machine-translated from English with a state-of-the-art translation engine. The resulting PORTULAN ExtraGLUE benchmark is a basis for research on Portuguese whose improvement can be pursued in future work. Similarly, the respective fine-tuned neural language models, developed with a low-rank adaptation approach, are made available as baselines that can stimulate future work on the neural processing of Portuguese. All datasets and models have been developed and are made available for two variants of Portuguese: European and Brazilian. 

**Keywords:** Machine translation, Portuguese, Benchmark, LoRA 

## **1. Introduction** 

Neural language models are pervasive in Natural Language Processing (NLP) applications and have radically changed the state-of-the-art since the Transformer architecture (Vaswani et al., 2017) was proposed. This has given rise to encoder (Devlin et al., 2019), decoder (Radford et al., 2018), and encoder-decoder architectures (Raffel et al., 2020). To support the development of such models, several benchmarks have been created to assess their performance in several downstream tasks (Wang et al., 2018, 2019). However, most research in NLP has focused on the English language (Bender, 2011), and as a consequence, many other languages lack sufficient resources – in particular, benchmarks for neural language models. 

Developing benchmark datasets is hard, usually demanding labeling by experts, especially for complex semantic-level tasks. An alternative path that has been resorted to in the literature is to rely on state-of-the-art Machine Translation (MT) to produce dependable datasets, namely those that support the evaluation of neural models in downstream tasks (Conneau et al., 2018; Eger et al., 2018; Yang et al., 2019; Carrino et al., 2020; d’Hoffschmidt et al., 2020; Shavrina et al., 2020; Carvalho et al., 2021; Sousa et al., 2021; Žagar 

and Robnik-Šikonja, 2022). Though possibly imperfect, such datasets can fit the purpose of greatly leveraging research in less-resourced languages, possibly complemented with human-curated test sets. 

In this paper, we contribute to enriching the set of benchmarks publicly available for Portuguese by relying on MT applied to tasks from the well-known GLUE (Wang et al., 2018) and SuperGLUE (Wang et al., 2019) benchmarks, which were originally developed for English. We discuss the issues encountered with our approach and provide versions of several tasks for European (pt-PT) and Brazilian (pt-BR) Portuguese, which altogether we named PORTULAN ExtraGLUE. 

As a way of their practical validation, for most tasks, we include experimental evaluation of different Portuguese language models fine-tuned with the respective datasets. Hence, for many of them, these will be the first models to address that task in Portuguese, and we thus contribute the first baselines for them. To that end, we resort to the encoder Albertina language model (Rodrigues et al., 2023) and the low-rank adaptation approach (Hu et al., 2022). The resulting fine-tuned language models for these tasks are openly distributed as open source under an open license. 

24 

_BUCC 2024: The 17th Workshop on Building and Using Comparable Corpora_ , pages 24–34 20 May, 2024. © 2024 ELRA Language Resource Association: CC BY-NC 4.0 

## **2. Related Work** 

Producing benchmarks to evaluate language models in downstream tasks is a daunting endeavor. The more complex the task, the more difficult it is to produce quality data that can be used to train models in a fine-tuning approach and test their capabilities. While highly resourced languages, such as English, include quite elaborate benchmarks (Wang et al., 2018, 2019), few evaluation datasets are available for other, less-resourced languages.[1] The particular case of Portuguese is a paradigmatic example, with only a few tasks being available for this purpose (Fonseca et al., 2016; Real et al., 2020; Santos et al., 2006; Freitas et al., 2010). 

A few examples of manually produced multilingual parallel corpora exist (Yang et al., 2019; Artetxe et al., 2020b; Ponti et al., 2020; Sen et al., 2022), as well as collections of tasks in multiple languages (Srivastava et al., 2023). At the same time, machine translation has come to a point in which it can be useful to create corpora that, while lacking human curation, can, up to a certain extent, be used to evaluate language models in the target languages (Conneau et al., 2018; Eger et al., 2018; Yang et al., 2019; Carrino et al., 2020; d’Hoffschmidt et al., 2020). Some have been created to allow cross-lingual evaluation of pre-trained encoders (Hu et al., 2020; Liang et al., 2020). 

State-of-the-art MT systems still struggle to produce accurate translations in several situations. Short texts, for instance, often lack enough context to obtain proper translations (Wan et al., 2022). Because of this, translation at the sentence level often falls short of translating longer texts, which provide more context (Jin et al., 2023).Translating from mostly gender-poor to gender-rich languages is also often a source of translation errors (Savoldi et al., 2021). Idioms are among the most intricate artifacts for MT systems, which tend to over-generate compositional and literal translations (Dankers et al., 2022). Additionally, translationbased data can arguably be seen as a dialect of the target language (Volansky et al., 2013; Artetxe et al., 2020a), with the possible effect of over-estimating the performance in the target language of models trained on such data. Still, MT has progressed notably over the last few years; it can, we believe, be used to produce datasets that are useful as a proxy in assessing the comparative merits of different (monolingual) language models. 

Following this trend, some works have leveraged MT to produce corpora in Portuguese (Carvalho et al., 2021; Sousa et al., 2021). We leverage stateof-the-art MT in producing Portuguese variants of 

> 1For instance, treebank annotations (Nivre et al., 2020) are available, but do not comprise benchmarks _per se_ . 

several GLUE (Wang et al., 2018) and SuperGLUE (Wang et al., 2019) tasks. Similar efforts have been made for other languages (Shavrina et al., 2020; Žagar and Robnik-Šikonja, 2022). 

In tandem with developing and making these datasets available, and as a way of their practical validation, we also release low-ranked adaptations (Hu et al., 2022) of Albertina-based models (Rodrigues et al., 2023), arguably the best open encoder models for both European and Brazilian Portuguese available at the time of this writing. 

Low-ranked adaptations (LoRA) reduce the number of training parameters, alleviating storage requirements for language models adapted to specific tasks while outperforming other fine-tuning techniques. For that, pre-trained model weights are frozen, and two additional weight matrices are used to adapt the model to the downstream task. After training, such weights can be merged with the frozen weights so that no latency is added at inference time, which is a main advantage compared to other low-rank adapters (Houlsby et al., 2019; Mahabadi et al., 2021; He et al., 2022). Concerning LoRA, more recent proposals (Valipour et al., 2023; Audibert et al., 2023) rely on the GLUE benchmark (Wang et al., 2018) to report improvements. 

## **3. General Language Understanding Evaluation Benchmarks** 

The General Language Understanding Evaluation (GLUE) tasks are meant to measure the progress toward general-purpose language understanding technologies for English. Both GLUE and SuperGLUE are aggregations of existing public datasets accompanied by a single-number performance metric and an analysis toolkit. The tasks included in these benchmarks can be grouped as follows[2] . 

## **3.1. Single sentence tasks** 

The Corpus of Linguistic Acceptability ( **CoLA** )[G] (Warstadt et al., 2019) is a task including sentences annotated for grammatical acceptability by experts in linguistics. The Stanford Sentiment Treebank ( **SST-2** )[G] (Socher et al., 2013), in turn, is a task for predicting the sentiment polarity of movie reviews. 

## **3.2. Similarity tasks** 

The Microsoft Research Paraphrase Corpus ( **MRPC** )[G] (Dolan and Brockett, 2005) is a task for determining whether a pair of sentences are mutual paraphrases. Quora Question Pairs ( **QQP** )[G,][3] is 

2We superscript each task regarding its inclusion in (G)LUE, (S)uperGLUE, or both. 

> 3 `https://quoradata.quora.com/ First-Quora-Dataset-Release-Question-Pairs` 

25 

a task for determining whether a pair of questions are semantically equivalent. The Semantic Textual Similarity Benchmark ( **STS-B** )[G] (Cer et al., 2017) is a task for predicting a similarity score (from 1 to 5) for each sentence pair. Word-in-Context ( **WiC** )[S] (Pilehvar and Camacho-Collados, 2019) comprises a word sense disambiguation task, where given two sentences containing a polysemous target word, the aim is to determine whether the word is used in the same sense in both sentences. 

## **3.3. Inference tasks** 

The Multi-Genre Natural Language Inference Corpus ( **MNLI** )[G] (Williams et al., 2018) is a task to determine if a given premise sentence entails, contradicts, or is neutral to a hypothesis sentence; the task includes matched (in-domain) and mismatched (cross-domain) validation and test sets. Question NLI ( **QNLI** )[G] (Rajpurkar et al., 2016) is a question-answering task converted to determine whether the context sentence contains the answer to the question. Recognizing Textual Entailment ( **RTE** )[GS] is a task for determining whether a premise sentence entails a hypothesis sentence. Winograd Natural Language Inference ( **WNLI** )[G] (Levesque et al., 2012) is a pronoun resolution task formulated as sentence pair entailment classification where, in the second sentence, the pronoun is replaced by a possible referent. Similarly, the Winograd Schema Challenge ( **WSC** )[S] is a co-reference resolution task also formulated as sentence pair entailment classification, where each example comprises a sentence and a pair pronoun-noun, the objective being to determine if they are co-referent. CommitmentBank ( **CB** )[S] (de Marneffe et al., 2019) comprises short texts with embedded clauses; one such clause is extracted as a hypothesis and should be classified as neutral, entailment or contradiction. 

GLUE and SuperGLUE also include expertconstructed diagnostic datasets covering diverse linguistic phenomena. Broadcoverage Diagnostics ( **AX** _b_ )[GS] (Wang et al., 2018) is a Natural Language Inference (NLI) task designed to test models across a wide spectrum of linguistic, commonsense, and world knowledge; each instance contains a sentence pair labeled with entailment or not entailment. Winogender Schema Diagnostics ( **AX** _g_ )[S] (Rudinger et al., 2018) is a similar task, designed to measure gender bias, where each premise sentence includes a male or female pronoun and a hypothesis includes a possible referent for the pronoun. 

## **3.4. Question-answering tasks** 

Boolean Questions ( **BoolQ** )[S] (Clark et al., 2019) is a question-answering task where yes/no questions are given for short text passages. In the MultiSentence Reading Comprehension ( **MultiRC** )[S] 

task (Khashabi et al., 2018), given a context paragraph, a question, and an answer, the goal is to determine whether the answer is true; for the same context and question, more than one answer may be correct. In the Reading Comprehension with Commonsense Reasoning Dataset ( **ReCoRD** )[S] , each sample is a multiple-choice question including a news article passage and a Cloze-style question with one entity masked out; the aim is to predict the masked entity from a list of alternatives. 

## **3.5. Reasoning tasks** 

Choice of Plausible Alternatives ( **COPA** )[S] (Gordon et al., 2012) is a casual reasoning task: given a premise, two choices, and a cause/effect prompt, the system must choose one of the choices. 

## **4. PORTULAN ExtraGLUE** 

Creating a Portuguese version of the tasks introduced in the previous section via machine translation (MT) requires a thoughtful understanding of the nature of each task, together with the limitations of the selected MT engine. While we are aware that, for a small subset of these tasks, Portuguesetranslated versions have already been created (Rodrigues et al., 2023), such considerations have not been taken into account. In fact, the inner workings of MT and the differences between languages (in our case, English and Portuguese) may impact the validity of the gold labels in supervised tasks. This is something we analyze in this section before providing details on the PORTULAN ExtraGLUE datasets we obtained. 

For MT, we use DeepL[4] , a commercial MT tool that tailors translation to two Portuguese variants, European (pt-PT) and Brazilian (pt-BR). 

## **4.1. More than translation** 

Both statistical and neural sequence-to-sequence MT models are affected by language model probabilities. As a side effect, ill-formed or ungrammatical source sentences are affected in the translation process, hindering the faithfulness of the output in the target language as a direct counterpart of the input in the source language. In fact, MT has been used in grammatical error correction (Rozovskaya and Roth, 2016; Kementchedjhieva and Søgaard, 2023). For this reason, we abstain from machinetranslating the CoLA dataset, as the obtained translation may easily corrupt the target labels. As an example, the sentence “They drank the pub” (linguistically _ungrammatical_ ) is translated to pt-BR 

> 4All the examples in this section are obtained via DeepL’s web interface ( `https://www.deepl.com/ translator` ) at the time of writing. 

26 

as “Eles beberam _no_ bar” (“They drank _in the_ pub”, _grammatical_ ). As another example, the sentence “The professor talked us” ( _ungrammatical_ ) is translated to pt-PT as “O professor falou _-nos_ ” (“The professor talked _to_ us”, _grammatical_ ). 

## **4.2. Gendered nouns and pronoun resolution** 

English common nouns do not express grammatical gender. On the other hand, Portuguese common nouns do and are used with corresponding gendered determiners (as opposed to English gender-neutral _the_ or _a_ ). This exacerbates the difficulty of properly addressing pronoun reference resolution, given that third-person singular pronouns (and also plural in Portuguese) are gendered. Tasks specifically dealing with pronoun resolution or evaluating the gender robustness of language models are thus prone to corruption via MT. These include WNLI, WSC, and AX _g_ . While we provide translated versions of WNLI and AX _g_ , we conduct error analysis to diagnose the quality level of their Portuguese versions. 

An example of a translation issue in WNLI is as follows: “Tom said "Check" to Ralph as he took his bishop” / “Tom said "Check" to Ralph as he took Ralph’s bishop” is translated to pt-PT as “O Tomás disse "Xeque" ao Rafa quando _este_ lhe tirou o bispo” / “O Tomás disse "Xeque" ao Rafa quando tirou o bispo ao Rafa”. The first sentence in the pair is wrongly translated ( _este_ means _the latter_ ), and even though it does not make sense, the target label should change from _entailment_ to _not_  entailment_ . 

For WSC, the situation is more critical, as parts of the input are isolated words (usually nouns and pronouns). Thus, obtaining a proper Portuguese equivalent requires more than MT. An example is as follows: from “The _mothers_ of Arthur and Celeste have come to the town to fetch them. They are very happy to have them back, but they scold _them_ just the same because they ran away”, we want to determine whether the italicised words are coreferent. In this example, there is no separated word matching _them_ (which should translate to _eles_ ) in the translation “As mães do Artur e da Celeste vêm buscá-los à cidade. Estão muito contentes por _os_ terem de volta, mas repreendem _-nos_ na mesma por terem fugido”. 

AX _g_ focuses on gender bias, explicitly combining both concerns expressed above. For instance, the hypothesis “The investigator tried to get in contact” is translated into Portuguese as “O investigador tentou entrar em contacto”; its possibly accompanying premises “The investigator wanted to interview the witness in person, but [he | she] was unable to get in contact” are translated into Portuguese as “O 

investigador queria entrevistar a testemunha pessoalmente, mas não conseguiu entrar em contacto com _ela_ ” (for he), or to “O investigador queria entrevistar a testemunha pessoalmente, mas _ela_ não conseguiu entrar em contacto” (for _she_ ). In the latter case, limiting the possible referents of pronoun _ela_ (she) – the only feminine noun is _testemunha_ (witness), since _investigador_ (investigator) is masculine in Portuguese – renders the _entailment_ label wrong, as it should be changed to _not_  entailment_ . 

## **4.3. Named entities** 

Another issue we have encountered when using DeepL is the non-deterministic translation of common or proper names, which might make finetuning models in these datasets harder or even impact label quality. Consider the following example, taken from WNLI: “ _Jane_ gave _Joan_ candy because she wasn’t hungry” / “ _Jane_ wasn’t hungry” is translated to pt-PT as “A _Joana_ deu doces à _Joana_ porque ela não tinha fome” / “A _Joana_ não tinha fome”; in this example, one of the distinct proper names is lost. The reverse can also happen: “Bill passed the half-empty plate to _John_ because he was full” / “ _John_ was full” is translated to pt-PT as “O Bill passou o prato meio vazio ao _John_ porque estava cheio” / “O _João_ estava cheio”; in this case, a single entity, _John_ , is either kept or translated to _João_ in the same short text. 

As another example from the same dataset, now concerning the same common noun being translated differently, “I couldn’t put the _pot_ on the shelf because it was too tall” / “The _pot_ was too tall”. is translated to pt-PT as “Não podia colocar a _panela_ na prateleira porque era demasiado alta” / “O _pote_ era demasiado alto”. 

These issues may be prevalent in every dataset, particularly in pt-PT variants. 

## **4.4. Machine-translated tasks** 

The set of datasets that have been translated and are part of PORTULAN ExtraGLUE[5] are included in Table 1. As mentioned in Sections 4.1 and 4.2, we leave out the CoLA and WSC datasets. 

For MNLI, we provide translations only for the matched and mismatched validation and test sets due to the excessive size of the training set[6] . Likewise, we do not translate the QQP dataset[7] . 

Given the nature of the WiC task (based on word sense disambiguation), we posit that a (human or machine) translated version of this dataset is not viable and thus leave it out. Finally, given the focus 

> 5Made available at `https://huggingface.co/ datasets/PORTULAN/extraglue` . 

> 6The training set for MNLI contains 393k rows. 7QQP includes a total of 795k rows. 

27 

|**Task**|**Train**|**Val**|**Test**|**Tokens(en)**|**Version**|**Tokens(pt)**|**mt**_e_|**lab**_e_|**low**_q_|
|---|---|---|---|---|---|---|---|---|---|
|SST-2|67.3k|872|1.82k|686.1k|pt-PT<br>pt-BR|725.3k<br>724.9k|4%<br>4%|0%<br>0%|0%<br>0%|
|MRPC|3.67k|408|1.73k|254.3k|pt-PT<br>pt-BR|287.2k<br>284.7k|4%<br>6%|0%<br>0%|2%<br>2%|
|STS-B|5.75k|1.5k|1.38k|197.5k|pt-PT<br>pt-BR|220.6k<br>217.8k|2%<br>2%|0%<br>0%|0%<br>0%|
|MNLI<br>_matched|–|9.82k|9.8k|649.4k|pt-PT<br>pt-BR|660.6k<br>661.4k|0%<br>4%|0%<br>0%|0%<br>0%|
|MNLI<br>_mismatched|–|9.83k|9.85k|680.6k|pt-PT<br>pt-BR|710.3k<br>705.3k|6%<br>4%|0%<br>0%|0%<br>0%|
|QNLI|105k|5.46k|5.46k|4.82M|pt-PT<br>pt-BR|5.22M<br>5.14M|2%<br>0%|2%<br>0%|2%<br>0%|
|RTE|2.49k|277|3k|333.8k|pt-PT<br>pt-BR|364.4k<br>360.8k|2%<br>2%|0%<br>0%|0%<br>0%|
|WNLI|635|71|146|29.7k|pt-PT<br>pt-BR|30.2k<br>29.5k|6%<br>8%|4%<br>6%|4%<br>6%|
|CB|250|56|250|43.3k|pt-PT<br>pt-BR|40.4k<br>40.5k|6%<br>8%|2%<br>2%|2%<br>4%|
|AX_b_|–|–|1.1k|40.2k|pt-PT<br>pt-BR|43.3k<br>42.7k|20%<br>20%|4%<br>4%|14%<br>12%|
|AX_g_|–|–|356|8.7k|pt-PT<br>pt-BR|8.9k<br>8.8k|22%<br>20%|10%<br>6%|10%<br>8%|
|BoolQ|9.43k|3.27k|3.25k|1.93M|pt-PT<br>pt-BR|2.07M<br>2.06M|22%<br>18%|2%<br>2%|12%<br>8%|
|MultiRC|27.2k|4.85k|9.69k|12.99M|pt-PT<br>pt-BR|13.69M<br>13.65M|10%<br>10%|2%<br>4%|2%<br>4%|
|CoPA|400|100|500|19.5k|pt-PT<br>pt-BR|18.6k<br>19.3k|2%<br>2%|2%<br>2%|2%<br>2%|



Table 1: PORTULAN ExtraGLUE datasets. For each task, we include the size of each partition, the number of tokens in each Portuguese variant, and results from the sample analysis in percentages (mt _e_ = machine translation errors, lab _e_ = corrupted labels, and low _q_ = low-quality translated samples). 

of the ReCoRD task on named entities and the issues encountered and described in Section 4.3, we abstain from translating this dataset as well. 

To improve translation quality, we concatenate each dataset entry’s textual columns with a line break.This ensures that the MT model can access as much context as is available (which may be critical for datasets with very short text spans) and is in line with previous findings (Artetxe et al., 2020a). 

As it can be seen in Table 1, the number of tokens varies among the Portuguese language variants. To better assess how different these are in the resulting machine-translated datasets, we calculate the BLEU score (Papineni et al., 2002) between both variants. For that, we rely on 4-grams; BLEU is calculated independently for each feature (text column in a dataset) and then averaged for the whole dataset. The BLEU score averaged over both directions (pt-PT _→_ pt-BR and pt-BR _→_ pt-PT) and for all datasets is 57.3, with the lowest value of 46.7 on the CoPA dataset and the highest of 64.5 on RTE. These values demonstrate that there are significant 

differences between the translations obtained for each variant via DeepL. 

To assess the quality of each machine-translated dataset, we resort to sampling 50 randomly selected examples, which were manually checked by three of the authors[8] for translation correctness and target label consistency. The rightmost columns in Table 1 show the results of this analysis: obvious translation errors, label corruption, and low-quality entries that should be removed from the dataset, given its nature. 

The main translation problems we have observed concern pronoun resolution or gender issues (as already emphasized in Section 4.2), idiomatic expressions, inconsistent translations in pairs of sentences, and a few cases of ‘hallucinations,’ among other problematic mistranslations. In some cases, these problems have an impact on the correctness of the labels (mainly in WNLI and AX _g_ ); in other cases, they mostly imply a dataset of lesser quality (such as in AX _b_ and BoolQ). In the specific case of 

> 8Portuguese native speakers and fluent in English. 

28 

|**Hyper-parameter**|**Value**|**5.2.**<br>**Results**|
|---|---|---|
|r<br>alpha<br>dropout<br>batch size<br>learningrate<br>weight decay|8<br>32<br>0_._05<br>8<br>2_×_10_−_5<br>0_._05|The fne-tuning r<br>these models a<br>regarding these<br>Comparing th<br>variants (pt-PT <br>pt-BRvarianta|



The fine-tuning results are presented in Table 3. All these models are the first baselines for the tasks regarding these new datasets. 

Comparing the empirical results between the two variants (pt-PT and pt-BR), we observe that the pt-BR variant achieves better scores than the ptPT variant in seven tasks (SST-2, MRPC, STS-B, RTE, WNLI, CB, and BoolQ), while the pt-PT variant has better scores in three tasks (QNLI, MultiRC, and CoPA). It is worth noting, however, that the differences are marginal in most cases. The larger discrepancies are observed for the WNLI, BoolQ and CoPA tasks. The first two tasks yield better results with the pt-BR variant, whereas the CoPA task achieves a better outcome in the pt-PT variant. 

Table 2: LoRA hyper-parameters. 

AX _g_ , even when the translation is correct, it does not do justice to the nature of the task, which loses its purpose (e.g., _his_ / _her_ translate the same way to Portuguese). 

Despite these problems, machine translation errors amount to only an average of 8%, with a mode as low as 2%. Label errors are even lower, with an average of 2% and a zero mode. We did not observe relevant differences between Portuguese language variants. 

We can also compare the results with those available for a subset of tasks and the current state-ofthe-art Albertina models, as reported in Rodrigues et al. (2023). For the pt-PT variant: in MRPC we obtain 0.8969 accuracy compared to 0.9171 in the original 900M Albertina model; in STS-B we obtain a Pearson correlation of 0.8905 compared to Albertina’s 0.8801; in RTE we obtain 0.7870 accuracy against .8339; and in WNLI we obtain 0.6197 accuracy against 0.4225. For the pt-BR variant: in MRPC we obtain 0.9184 accuracy compared to 0.9071 in the original 900M Albertina model; in STS-B we obtain a Pearson correlation of 0.8940 compared to Albertina’s 0.8910; in RTE we obtain 0.7978 accuracy against 0.7545; and in WNLI we obtain 0.6901 accuracy against 0.4601. We note, however, that the translations of these tasks in PORTULAN ExtraGLUE may differ from the translations used by the authors of the Albertina model for their evaluations. This is certainly true for the pt-BR variant, as the MT model used differed. 

## **5. Albertina LoRA Models** 

We train and make available a set of fine-tuned lowrank adaptations of Albertina-based language models. For several PORTULAN ExtraGLUE datasets, we fine-tune a 1.5B Albertina language model for two Portuguese variants, European (pt-PT) and Brazilian (pt-BR). The resulting models are a practical validation for the created datasets. 

## **5.1. Set up** 

First, we adapt each task example for tokenization regarding their input components. For this, we concatenate the input features with a special token separator. On the MRPC and STS-B similarity tasks, we concatenate the first and second sentences. On the CB and RTE inference tasks, the hypothesis and premise; on QNLI, the sentence and question. For the BoolQ Question-answering task, we concatenate the passage and question; for MultiRC, the paragraph, question, and answer, truncating the paragraph if needed. For the CoPA reasoning task, we concatenate the premise and question and then join with each choice, resulting in two inputs. During tokenization, we truncate the examples with a maximum context length of 128 tokens, except in MultiRC, which uses 256 tokens. 

Table 3 also includes the results obtained by finetuning the multilingual XLM-RoBERTa-XL[9] model (Conneau et al., 2020) following the same LoRA approach. XLM-RoBERTa-XL is significantly larger (3.5B parameters) than Albertina 1.5B. Even so, we note the benefits of using monolingual models when comparing such results with our Albertina 1.5B LoRA models. In fact, we observe improvements in Albertina 1.5B LoRA models for all tasks and in both Portuguese variants. In some cases, improvements are significant. 

When comparing with the DeBERTa[10] (He et al., 2021) model (the foundation model for Albertina) applied to the original English datasets, the results of our low-rank adapters on the PORTULAN ExtraGLUE datasets fall behind in most cases. This 

After tokenization, we apply a low-rank adapter (Hu et al., 2022) with the hyper-parameters shown in Table 2. Due to hardware limitations, it was unfeasible to perform a grid search on these hyper-parameters. We chose the current hyperparameters by resorting to small-scale exploratory experiments. Because several datasets lack test labels, we fine-tuned models on the training split and evaluated them on the validation split. 

> 9 `https://huggingface.co/facebook/ xlm-roberta-xl` 

> 10 `https://huggingface.co/microsoft/ deberta-v2-xxlarge` 

29 

|**Task**<br>**Albertina 1.5B**<br>**pt-PT**<br>**pt-BR**|**XLM-RoBERTa-XL**<br>**pt-PT**<br>**pt-BR**|**DeBERTa-V2-XXLarge**<br>**en**|
|---|---|---|
|Single sentence<br>SST-2<br>0.9392<br>0.9450|0.9323<br>0.9392|0.9633|
|Similarity<br>MRPC<br>0.8969<br>0.9184<br>STS-B<br>0.8905<br>0.8940|0.8696<br>0.8651<br>0.8743<br>0.8734|0.9266<br>0.9170|
|Inference<br>QNLI<br>0.9398<br>0.9361<br>RTE<br>0.7870<br>0.7978<br>WNLI<br>0.6197<br>0.6901<br>CB<br>0.8385<br>0.8554|0.9237<br>0.9237<br>0.6571<br>0.6606<br>0.5634<br>0.5634<br>0.6280<br>0.6160|0.9608<br>0.8917<br>0.7887<br>0.8936|
|QA<br>BoolQ<br>0.7456<br>0.7807<br>MultiRC<br>0.7257<br>0.7169|0.6538<br>0.6587<br>0.6926<br>0.6925|0.8900<br>0.8243|
|Reasoning<br>CoPA<br>0.8500<br>0.8200|0.5000<br>0.5600|0.9200|



Table 3: Evaluation scores on validation sets for both variants regarding the different categories of datasets (Single Sentence, Similarity, Inference, Question-Answering, and Reasoning). Performance on SST-2, QNLI, RTE, WNLI, BoolQ, and CoPA is measured with accuracy; on MRPC, CB, and MultiRC with F1; and on STS-B with Pearson. For comparison, we include results for the multilingual XLM-RoBERTa-XL 3.5B model, fine-tuned using the same LoRA approach. For reference, we also include results for English by applying LoRA to the DeBERTa-V2-XXLarge 1.5B model (based on which Albertina has been developed). 

is expected for at least two reasons: first, Albertina was pre-trained with far fewer data than DeBERTa; second, we rely on machine translation to obtain the datasets for the tasks, which, as discussed before, isn’t without issues. Tasks exhibiting significant differences in performance include WNLI, which, as explained in Section 4.2, has issues related to pronoun resolution. 

prove this benchmark with manual curation of the datasets (in particular, the test sets) and expand it with new ones. Additionally, developing new datasets from scratch may better reflect the language and the cultures latent within language variants (which go well beyond European and Brazilian ones). Evolving these in a leaderboard would help foster research in the Portuguese language. 

## **6. Conclusion** 

## **Acknowledgements** 

We contribute an open benchmark suite to support the development of the neural processing of Portuguese. In this initial version, this suite comprises 14 datasets for downstream tasks of various types, including single sentence tasks, similarity tasks, inference tasks, and reasoning tasks. To kick-start benchmarking for this language, these datasets were machine-translated from mainstream benchmarks in the literature and designated as PORTULAN ExtraGLUE. We also make available baseline models for 10 of these tasks, developed with the low-rank adaptation approach over a state-of-theart and open language model for Portuguese. 

Even though MT datasets have their limitations and pitfalls, our manual analysis has found a relatively reduced amount of (translation and label) errors. We believe this renders our obtained datasets highly useful for assessing the comparative performance of neural language models for Portuguese. 

This research was partially supported by: PORTULAN CLARIN – Research Infrastructure for the Science and Technology of Language, funded by Lisboa 2020, Alentejo 2020 and FCT (PINFRA/22117/2016); ACCELERAT.AI – Multilingual Intelligent Contact Centers, funded by IAPMEI (C625734525-00462629); ALBERTINA – Foundation Encoder Model for Portuguese and AI, funded by FCT (CPCA-IAC/AV/478394/2022); and Base Funding (UIDB/00027/2020) and Programmatic Funding (UIDP/00027/2020) of the Artificial Intelligence and Computer Science Laboratory (LIACC) funded by national funds through FCT/MCTES (PIDDAC). 

In future work, it would be important to im- 

30 

## **Bibliographical References** 

- Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2020a. Translation artifacts in cross-lingual transfer learning. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 7674–7684, Online. Association for Computational Linguistics. 

- Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020b. On the cross-lingual transferability of monolingual representations. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 4623–4637, Online. Association for Computational Linguistics. 

- Alexandre Audibert, Massih R Amini, Konstantin Usevich, and Marianne Clausel. 2023. Lowrank updates of pre-trained weights for multi-task learning. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 7544– 7554, Toronto, Canada. Association for Computational Linguistics. 

- E. M. Bender. 2011. On achieving and evaluating language-independence in NLP. _Linguistic Issues in Language Technology_ , 6. 

- Casimiro Pio Carrino, Marta R. Costa-jussà, and José A. R. Fonollosa. 2020. Automatic Spanish translation of SQuAD dataset for multilingual question answering. In _Proceedings of the Twelfth Language Resources and Evaluation Conference_ , pages 5515–5523, Marseille, France. European Language Resources Association. 

- Nuno Ramos Carvalho, Alberto Simões, and José João Almeida. 2021. Bootstrapping a data-set and model for question-answering in Portuguese (short paper). In _10th Symposium on Languages, Applications and Technologies (SLATE 2021)_ . Schloss Dagstuhl-LeibnizZentrum für Informatik. 

- Daniel Cer, Mona Diab, Eneko Agirre, Iñigo LopezGazpio, and Lucia Specia. 2017. SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In _Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)_ , pages 1–14, Vancouver, Canada. Association for Computational Linguistics. 

- Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In _Proceedings of the 2019 Conference of the North_ 

_American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 2924–2936, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , pages 8440–8451, Online. Association for Computational Linguistics. 

- Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: Evaluating cross-lingual sentence representations. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_ , pages 2475–2485, Brussels, Belgium. Association for Computational Linguistics. 

- Verna Dankers, Christopher Lucas, and Ivan Titov. 2022. Can transformer be too compositional? analysing idiom processing in neural machine translation. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 3608– 3626, Dublin, Ireland. Association for Computational Linguistics. 

- Marie-Catherine de Marneffe, Mandy Simons, and Judith Tonhauser. 2019. The CommitmentBank: Investigating projection in naturally occurring discourse. _Proceedings of Sinn und Bedeutung_ , 23(2):107–124. 

- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Martin d’Hoffschmidt, Wacim Belblidia, Quentin Heinrich, Tom Brendlé, and Maxime Vidal. 2020. FQuAD: French question answering dataset. In _Findings of the Association for Computational Linguistics: EMNLP 2020_ , pages 1193–1208, Online. Association for Computational Linguistics. 

- William B. Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential 

31 

paraphrases. In _Proceedings of the Third International Workshop on Paraphrasing (IWP2005)_ . 

- Steffen Eger, Johannes Daxenberger, Christian Stab, and Iryna Gurevych. 2018. Cross-lingual argumentation mining: Machine translation (and a bit of projection) is all you need! In _Proceedings of the 27th International Conference on Computational Linguistics_ , pages 831–844, Santa Fe, New Mexico, USA. Association for Computational Linguistics. 

- E Fonseca, L Santos, Marcelo Criscuolo, and S Aluisio. 2016. ASSIN: Avaliacao de similaridade semantica e inferencia textual. In _Computational Processing of the Portuguese Language12th International Conference, Tomar, Portugal_ , pages 13–15. 

- Cláudia Freitas, Cristina Mota, Diana Santos, Hugo Gonçalo Oliveira, and Paula Carvalho. 2010. Second HAREM: Advancing the state of the art of named entity recognition in Portuguese. In _Proceedings of the Seventh International Conference on Language Resources and Evaluation (LREC’10)_ , Valletta, Malta. European Language Resources Association (ELRA). 

- Andrew Gordon, Zornitsa Kozareva, and Melissa Roemmele. 2012. SemEval-2012 task 7: Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In _*SEM 2012: The First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012)_ , pages 394–398, Montréal, Canada. Association for Computational Linguistics. 

- Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. 2022. Towards a unified view of parameter-efficient transfer learning. In _International Conference on Learning Representations_ . 

- Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2021. DeBERTa: Decodingenhanced BERT with disentangled attention. In _International Conference on Learning Representations_ . 

- Dan Hendrycks, Collin Burns, Steven Basart, and et al. 2021. Measuring massive multitask language understanding. In _International Conference on Learning Representations_ . 

- Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for 

NLP. In _Proceedings of the 36th International Conference on Machine Learning_ , volume 97 of _Proceedings of Machine Learning Research_ , pages 2790–2799. PMLR. 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ . 

- Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. XTREME: A massively multilingual multitask benchmark for evaluating cross-lingual generalisation. In _Proceedings of the 37th International Conference on Machine Learning_ , volume 119 of _Proceedings of Machine Learning Research_ , pages 4411–4421. PMLR. 

- Linghao Jin, Jacqueline He, Jonathan May, and Xuezhe Ma. 2023. Challenges in context-aware neural machine translation. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , pages 15246–15263, Singapore. Association for Computational Linguistics. 

- Yova Kementchedjhieva and Anders Søgaard. 2023. Grammatical error correction through round-trip machine translation. In _Findings of the Association for Computational Linguistics: EACL 2023_ , pages 2208–2215, Dubrovnik, Croatia. Association for Computational Linguistics. 

- Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan Roth. 2018. Looking beyond the surface: A challenge set for reading comprehension over multiple sentences. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , pages 252–262, New Orleans, Louisiana. Association for Computational Linguistics. 

- Hector J. Levesque, Ernest Davis, and Leora Morgenstern. 2012. The Winograd Schema Challenge. In _Proceedings of the Thirteenth International Conference on Principles of Knowledge Representation and Reasoning_ , KR’12, page 552–561. AAAI Press. 

- Yaobo Liang, Nan Duan, Yeyun Gong, Ning Wu, Fenfei Guo, Weizhen Qi, Ming Gong, Linjun Shou, Daxin Jiang, Guihong Cao, Xiaodong Fan, Ruofei Zhang, Rahul Agrawal, Edward Cui, Sining Wei, Taroon Bharti, Ying Qiao, Jiun-Hung Chen, Winnie Wu, Shuguang Liu, Fan Yang, Daniel Campos, Rangan Majumder, and Ming Zhou. 2020. XGLUE: A new benchmark dataset 

32 

for cross-lingual pre-training, understanding and generation. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 6008–6018, Online. Association for Computational Linguistics. 

- Rabeeh Karimi Mahabadi, James Henderson, and Sebastian Ruder. 2021. Compacter: Efficient lowrank hypercomplex adapter layers. In _Advances in Neural Information Processing Systems_ . 

- Joakim Nivre, Marie-Catherine de Marneffe, Filip Ginter, Jan Hajič, Christopher D. Manning, Sampo Pyysalo, Sebastian Schuster, Francis Tyers, and Daniel Zeman. 2020. Universal Dependencies v2: An evergrowing multilingual treebank collection. In _Proceedings of the Twelfth Language Resources and Evaluation Conference_ , pages 4034–4043, Marseille, France. European Language Resources Association. 

- Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In _Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics_ , pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. 

- Mohammad Taher Pilehvar and Jose CamachoCollados. 2019. WiC: the word-in-context dataset for evaluating context-sensitive meaning representations. In _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)_ , pages 1267–1273, Minneapolis, Minnesota. Association for Computational Linguistics. 

- Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vulić, and Anna Korhonen. 2020. XCOPA: A multilingual dataset for causal commonsense reasoning. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 2362–2376, Online. Association for Computational Linguistics. 

- Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of Machine Learning Research_ , 21(140):1–67. 

- Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ 

questions for machine comprehension of text. In _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_ , pages 2383–2392, Austin, Texas. Association for Computational Linguistics. 

- Livy Real, Erick Fonseca, and Hugo Goncalo Oliveira. 2020. The ASSIN 2 shared task: A quick overview. In _International Conference on Computational Processing of the Portuguese Language_ , pages 406–412. Springer. 

- João Rodrigues, Luís Gomes, João Silva, António Branco, Rodrigo Santos, Henrique Lopes Cardoso, and Tomás Freitas Osório. 2023. Advancing neural encoding of portuguese with transformer Albertina PT-*. In _Progress in Artificial Intelligence - 22nd EPIA Conference on Artificial Intelligence, EPIA 2023, Faial Island, Azores, September 5-8, 2023, Proceedings, Part I_ , volume 14115 of _Lecture Notes in Computer Science_ , pages 441–453. Springer. 

- Alla Rozovskaya and Dan Roth. 2016. Grammatical error correction: Machine translation and classifiers. In _Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 2205–2215, Berlin, Germany. Association for Computational Linguistics. 

- Rachel Rudinger, Jason Naradowsky, Brian Leonard, and Benjamin Van Durme. 2018. Gender bias in coreference resolution. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)_ , pages 8–14, New Orleans, Louisiana. Association for Computational Linguistics. 

- Diana Santos, Nuno Seco, Nuno Cardoso, and Rui Vilela. 2006. HAREM: An advanced NER evaluation contest for Portuguese. In _Proceedings of the Fifth International Conference on Language Resources and Evaluation (LREC’06)_ , Genoa, Italy. European Language Resources Association (ELRA). 

- Beatrice Savoldi, Marco Gaido, Luisa Bentivogli, Matteo Negri, and Marco Turchi. 2021. Gender bias in machine translation. _Transactions of the Association for Computational Linguistics_ , 9:845– 874. 

- Priyanka Sen, Alham Fikri Aji, and Amir Saffari. 2022. Mintaka: A complex, natural, and multilingual dataset for end-to-end question answering. In _Proceedings of the 29th International Conference on Computational Linguistics_ , pages 1604– 1619, Gyeongju, Republic of Korea. International Committee on Computational Linguistics. 

33 

- Tatiana Shavrina, Alena Fenogenova, Emelyanov Anton, Denis Shevelev, Ekaterina Artemova, Valentin Malykh, Vladislav Mikhailov, Maria Tikhonova, Andrey Chertok, and Andrey Evlampiev. 2020. RussianSuperGLUE: A Russian language understanding evaluation benchmark. In _Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ , pages 4717–4726, Online. Association for Computational Linguistics. 

- Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In _Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing_ , pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics. 

- Afonso Sousa, Bernardo Leite, Gil Rocha, and Henrique Lopes Cardoso. 2021. Cross-lingual annotation projection for argument mining in portuguese. In _Progress in Artificial Intelligence_ , pages 752–765, Cham. Springer International Publishing. 

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, and et al. 2023. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. _Transactions on Machine Learning Research_ . 

- Mojtaba Valipour, Mehdi Rezagholizadeh, Ivan Kobyzev, and Ali Ghodsi. 2023. DyLoRA: Parameter-efficient tuning of pre-trained models using dynamic search-free low-rank adaptation. In _Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics_ , pages 3274–3287, Dubrovnik, Croatia. Association for Computational Linguistics. 

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In _Advances in Neural Information Processing Systems_ , volume 30. Curran Associates, Inc. 

- Vered Volansky, Noam Ordan, and Shuly Wintner. 2013. On the features of translationese. _Digital Scholarship in the Humanities_ , 30(1):98–118. 

   - Alex Wang, Yada Pruksachatkun, Nikita Nangia, and et al. 2019. Superglue: A stickier benchmark for general-purpose language understanding systems. In _Procs. 33rd International Conference on Neural Information Processing Systems_ , Red Hook, NY, USA. Curran Associates Inc. 

   - Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In _Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP_ , pages 353–355, Brussels, Belgium. Association for Computational Linguistics. 

   - Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. 2019. Neural network acceptability judgments. _Transactions of the Association for Computational Linguistics_ , 7:625–641. 

   - Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In _Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)_ , pages 1112– 1122, New Orleans, Louisiana. Association for Computational Linguistics. 

   - Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. 2019. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_ , pages 3687–3692, Hong Kong, China. Association for Computational Linguistics. 

   - Aleš Žagar and Marko Robnik-Šikonja. 2022. Slovene SuperGLUE benchmark: Translation and evaluation. In _Proceedings of the Thirteenth Language Resources and Evaluation Conference_ , pages 2058–2065, Marseille, France. European Language Resources Association. 

   - Xuhui Zhou, Yue Zhang, Leyang Cui, and Dandan Huang. 2020. Evaluating commonsense in pretrained language models. In _Procs. 34th AAAI, New York, USA, February 7-12, 2020_ , pages 9733–9740. AAAI Press. 

- Yu Wan, Baosong Yang, Derek Fai Wong, Lidia Sam Chao, Liang Yao, Haibo Zhang, and Boxing Chen. 2022. Challenges of neural machine translation for short texts. _Computational Linguistics_ , 48(2):321–342. 

34 

