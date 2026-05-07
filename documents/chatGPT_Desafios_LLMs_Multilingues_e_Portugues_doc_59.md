## **Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training** 

## **Toan Tran, Ruixuan Liu**[*] **, Li Xiong** Emory University Atlanta, GA, USA {vtran29;ruixuan.liu2;lxiong}@emory.edu 

## **Abstract** 

Large language models (LLMs) have become the backbone of modern natural language processing but pose privacy concerns about leaking sensitive training data. Membership inference attacks (MIAs), which aim to infer whether a sample is included in a model’s training dataset, can serve as a foundation for broader privacy threats. Existing defenses designed for traditional classification models do not account for the sequential nature of text data. As a result, they either require significant computational resources or fail to effectively mitigate privacy risks in LLMs. In this work, we propose DuoLearn, a lightweight yet effective empirical privacy defense for protecting training data of language models by leveraging token-specific characteristics. By analyzing token dynamics during training, we propose a token selection strategy that categorizes tokens into hard tokens for learning and memorized tokens for unlearning. Subsequently, our training-phase defense optimizes a novel dual-purpose tokenlevel loss to achieve a Pareto-optimal balance between utility and privacy. Extensive experiments demonstrate that our approach not only provides strong protection against MIAs but also improves language modeling performance by around 10% across various LLM architectures and datasets compared to the baselines.[1] 

## **1 Introduction** 

Large language models (LLMs) have become the foundation of modern natural language processing with a wide range of applications in various domains (Chang et al., 2024). The rapidly increasing deployment of LLMs raises serious concerns about data privacy (Yao et al., 2024). LLMs have been shown to memorize the training data which can be later extracted by adversaries (Carlini et al., 2023). Membership inference attacks 

> *Corresponding author: ruixuan.liu2@emory.edu 

> 1The implementation code for DuoLearn is available at https://github.com/Emory-AIMS/duolearn 

(MIAs) (Shokri et al., 2017; Li et al., 2024a) aim to infer whether a sample is included in a model’s training data, serving as the foundation of broader privacy threats (Carlini et al., 2021b). 

Due to the importance of understanding and mitigating MIAs, a significant amount of research has been conducted to design MIA defenses (Hu et al., 2022b). However, most defenses focus on general machine learning models for classification tasks and do not account for the sequential nature of text data, while advanced MIAs for LLMs have leveraged this property. For example, the series of Min-K works (Zhang et al., 2025; Shi et al., 2024) uses the token-level loss on outlier tokens and significantly enhance MIAs for LLMs. Thus, conventional data sanitization or regularization techniques have limited defense effectiveness (Kandpal et al., 2022; Liu et al., 2024b). Even though the classic differentially private (DP) training algorithm (Abadi et al., 2016) provides a strong defense, this approach comes at the inevitable cost of increased computation and reduced utility (Li et al., 2022a; Bu et al., 2023b), which may not be desirable when the model trainer serves as the defender. 

In this paper, we propose a defense mechanism for membership inference attacks on LLMs – DuoLearn. A recent study (Lin et al., 2024) reveals that using a carefully selected subset of tokens during training can match or even surpass the performance of using all tokens in language modeling. In the meantime, MIAs mainly exploit loss-based signals associated with a sample (Mattern et al., 2023; Carlini et al., 2021a). We observe that during training, some certain tokens carry stronger MIA signals than others, making the sample vulnerable to MIAs. Thus, we leverage the token sequence nature of LLMs and propose a dynamic token selection strategy during training to proactively identify and categorize tokens into hard tokens (those the model struggles with) and memorized tokens (those 

22872 

_Findings of the Association for Computational Linguistics: ACL 2025_ , pages 22872–22888 July 27 - August 1, 2025 ©2025 Association for Computational Linguistics 

with strong signals for MIA risks). Accordingly, we design a dual-objective loss function that performs learning via gradient descent on the hard tokens and unlearning via gradient ascent on the memorized tokens simultaneously in one backward pass, which makes the model learn useful information but not memorize specific training samples. Our contributions can be summarized as follows: 

- We propose a dynamic token selection strategy that identifies hard tokens and memorized tokens during training, which provides insights for investigating language modeling and memorization. 

- We propose a simple but effective dualobjective training to perform learning over hard tokens and unlearning over memorized tokens, for mitigating privacy risk while maintaining model utility with small computing cost. 

- We empirically demonstrate the effectiveness of the proposed defense mechanism across various LLM architectures and datasets. Our results show that our defense mechanism can provide robust privacy protection against MIAs with minimal degradation on language modeling performance. 

## **2 Related Works** 

## **2.1 MIAs on LLMs** 

Membership inference attacks are a crucial privacy threat to machine learning models. There are a significant number of MIAs proposed for traditional classification models (Hu et al., 2022b). Shokri et al. (2017) introduce MIAs via training a binary classification model over behaviors collected from shadow models. Yeom et al. (2018) connect MIAs to the overfitting phenomenon and propose to use cross entropy loss as an MIA signal. However, due to the significant differences between LLMs and traditional classification models, some of these attacks are not applicable to LLMs, while others, though feasible, have limited attack performance. Therefore, there are non-trivial efforts to design suitable MIAs for LLMs. Carlini et al. (2021a) calibrate the sample loss with zlib entropy and reference models. Mattern et al. (2023) generate synthetic neighboring samples for each target sample then calibrate the loss of the target sample with the averaged loss of its neighboring samples as the 

MIA signal. Shi et al. (2024) consider only top _K_ lowest token losses for the MIA signal, while Zhang et al. (2025) perform z-score normalization for token losses, using the token vocabulary’s mean and standard deviation, then select top _K_ z-scores. Fu et al. (2024) prompt the target LLM to generate a dataset which is used to train a reference attack model. Duan et al. (2024) conduct systematic evaluations of MIAs on the pretrained LLMs. Hayes et al. (2025) scale reference-based MIAs on large-scale pretraining settings. Puerto et al. (2025) consider various scales of membership from sentences to collections of documents. Liu et al. (2024b) design a privacy backdoor that can increase the membership inference risks. Feng et al. (2025) investigate MIAs on preference data used for posttraining alignment. 

## **2.2 LLM Memorization** 

The billion-parameter scale enhances LLM capabilities but also magnifies the privacy concerns. Carlini et al. (2021a, 2023) demonstrate that LLMs can memorize parts of their training data. There is a risk that LLMs may generate the training data when prompted appropriately. These are known as _exact memorization_ which can be utilized by the adversaries to extract the exact training data. Nasr et al. (2025) demonstrated that the LLM safety alignment fails to mitigate the privacy risks. It is feasible to undo the safety alignment via finetuning and the adversaries can prompt the LLM to generate its training data. 

## **2.3 Defenses Against MIAs** 

Overfitting is the root of membership inference risks (Shokri et al., 2017). While regularization methods such as weight decay and dropout (Srivastava et al., 2014) mitigates overfitting and slightly reduces the membership inference risks in the traditional classification models (Song and Mittal, 2021), they are not sufficient to prevent memorization in LLMs (Tirumala et al., 2022; Lee et al., 2022). Other defenses which leverage adversarial training (Nasr et al., 2018) or ensemble architecture of models (Tang et al., 2022) are infeasible for LLMs due to the expensive computing cost. 

Generally, in the context of LLMs, there are still limited number of works on defense mechanisms against MIAs and memorization. There are two main approaches: sanitizing training data and differential privacy (DP). Pilán et al. (2022) propose a practical method to protect Personally Identifi- 

22873 

able Information (PII) by detecting and replacing PII with anonymized tokens. Shi et al. (2022) sanitize the PII tokens and pretrain on the sanitized data before conducting DP based fine-tuning on the original data. Lukas et al. (2023) demonstrate the effectiveness of sentence-level DP in mitigating the risks of leaking PII. These PII protection methods are effective but may not be sufficient to protect against MIAs because for each sample, the number of PII tokens is usually small (Li et al., 2024b). Liu et al. (2024a) propose a method to perturb the training texts by leveraging memorization triggers that can effectively protect a small fraction of the training data against MIAs. Deduplicating the training corpus can reduce the risks of MIAs but not entirely eliminate them (Kandpal et al., 2022). 

The second popular approach conducts training/fine-tuning with Differentially-Private Stochastic Gradient Descent (DPSGD). Li et al. (2022b); Yu et al. (2022); Li et al. (2024b); Amit et al. (2024) show LLMs are strong differentially private learners. Lowy et al. (2024) theoretically prove that DP with a loose privacy budget can defend against MIAs. Despite efforts to improve the computing efficiency of DPSGD (Bu et al., 2023b), DP inherently introduces computational overhead, architectural constraints, and significant utility trade-off at scale. McKenna et al. (2025) explore the scaling laws of DP LLMs and reveal challenges especially about the optimal training data size. To avoid the computational overhead and utility tradeoff of using DP on LLMs, Hans et al. (2024) proposes a non-DP practical masking mechanism, called Goldfish, that performs pseudo-random token masking for loss calculation to prevent memorization. Our method is also an empirical defense in the training stage. 

## **3 How Do Tokens Contribute to Membership Inference Risks?** 

Compared to conventional classification problems, membership inference attacks in language modeling have significant differences. In particular, each query in traditional classification models requires only one prediction. On the other hand, each query to language models involves multiple token predictions due to the sequential nature of text. This difference yields a question that how tokens contribute to overall sample-level membership inference risks. To answer this question, we perform a token-level analysis of membership inference risks. We calcu- 

late the MIA signal for each token as its prediction loss calibrated by a reference model (Carlini et al., 2021a). A smaller signal value indicates that the model has a significantly higher confidence than the reference model on predicting the token. 

**==> picture [216 x 81] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 250<br>Member sample<br>40 Non-member sample 50 200<br>30 100 150<br>20 150 100<br>10 200 50<br>0 250 0<br>2 0 2 0 400 800<br>MIA Signal Training Step<br>Token Index<br>Number of Tokens MIA Signal Ranking<br>**----- End of picture text -----**<br>


Figure 1: Token-level MIA signal analysis. The left figure presents the histogram of the MIA signals across tokens at the end of training, while the right figure illustrates the MIA signal ranking of tokens during training. 

Figure 1 (left) illustrates the histogram of MIA signal values for the tokens of a member sample and a non-member sample (see Figure 9 in Appendix B for additional histograms). The nonmember sample distribution centers around zero, while the member sample skews to the negative side. Consequently, the average aggregated MIA signal is below zero for the member but around zero for the non-member, leading to membership inference risks. Moreover, the MIA signal values vary for different tokens, so some tokens contribute more to the membership inference risks than the others. Figure 1 (right) illustrates the MIA signal ranking of tokens of a member sample over training steps (see Figure 10 in Appendix B for additional samples). There is a complex changing dynamic in ranking between tokens before it becomes more stable at the end when the training converges. Overall, the analysis suggests that the sample-level membership inference risk in language modeling stem from the cumulative effect of many tokens. This poses challenges for defense methods, as they need tokenlevel granularity to isolate and mitigate specific sources of leakage. Additionally, it is non-trivial to develop a defense method that widely affects a large number of tokens without disrupting the complex token dependencies essential for model utility. 

## **4 Proposed Methodology – DuoLearn** 

Motivated by the analysis, we propose DuoLearn– a training framework that dynamically identifies hard tokens (those with higher calibrated losses) for learning and memorized tokens (those with lower 

22874 

**==> picture [409 x 259] intentionally omitted <==**

**----- Start of picture text -----**<br>
Dual-Purpose Loss Calculation<br>Loss for Learning Loss for Unlearning<br>Backpropagation + - + +<br>Order numbered AB89 has been shipped <EOS><br>Token Selection<br>Order numbered AB89 has been shipped <EOS> Order numbered AB89 has been shipped <EOS><br>Multi-head Attention Layer<br>Q Training LLM K V Reference LLM<br>Wq Wq Wq<br><BOS> Order numbered AB89 has been shipped <BOS> Order numbered AB89 has been shipped<br>**----- End of picture text -----**<br>


Figure 2: DuoLearn overview. First, the tokens are passed through the training LLM and reference LLM. They are then categorized into hard tokens (in green) and memorized tokens (in red). At the end, a dual-purpose loss is applied which achieves two targets: learning on the hard tokens while unlearning for the memorized tokens. 

calibrated loss or stronger MIA signals) for unlearning simultaneously. This way, the model learns useful information without memorizing specific training samples. 

**Overview** . We assume the model trainer is the defender and the goal is to mitigate the privacy risk of the training data in the trained model. We further assume the trainer can get access to an auxiliary dataset for better calibrating the MIA signals, which can be a disjoint subset drawn from the same distribution of the training data. The general training process is illustrated in Figure 2. First, we train a reference model with the auxiliary dataset, which is feasible for the trainer based on previous works (Lin et al., 2024; Mindermann et al., 2022; Xie et al., 2023). Then, during training of the target model, we use the token losses of the current training model calibrated by the reference model to dynamically identify hard tokens and memorized tokens in each training iteration. A dual-purpose loss function is used to keep the model simultaneously learning on hard and necessary tokens to enhance model utiilty while unlearning on memorized tokens to mitigate MIA risks. 

**Reference Modeling** . Reference model ( _θ_ ref) shares an identical architecture with the training model ( _θ_ ). We fine-tune a reference model on a small portion of the original dataset (denoted as 

_T_ aux) that can reflect the desired data distribution by standard causal language modeling (CLM), i.e., implementing next-token-prediction cross entropy loss ( _LCE_ ): 

**==> picture [222 x 16] intentionally omitted <==**

**Token Selection** . As our previous analysis, tokens contribute differently in membership inference risks. Thus, considering all tokens equally is not optimal for privacy defense with respect to the utility and privacy trade-off. DuoLearn defines two sets of tokens: hard tokens ( _Th_ ) and memorized tokens ( _Tm_ ). Hard tokens are the tokens that the current training models ( _θ_ ) have difficulty predicting, while memorized tokens are the tokens that the model has already memorized. To identify these two sets of tokens, we propose a token selection mechanism based on the prediction loss of each token calibrated by the reference model. We implement the score _s_ ( _ti_ ) for each token _ti_ which is the difference between the cross-entropy loss of the training model and the reference model, as used in previous works (Lin et al., 2024; Mindermann et al., 2022): 

**==> picture [197 x 12] intentionally omitted <==**

The tokens with the highest scores are considered hard tokens _Th_ (highest calibrated loss), while 

22875 

the tokens with the lowest scores are considered memorized tokens _Tm_ (lowest calibrated loss and strongest MIA signals). Let _T_ be the set of all tokens in a batch. We select top _Kh_ hard tokens and bottom _Km_ memorized tokens to form _Th_ and _Tm_ , respectively. Additionally, we introduce a threshold _τ_ to filter out neutral tokens from _Tm_ which have scores close to zero or greater than zero, as these are not considered memorized. The token selection process is formulated as follows: 

**==> picture [128 x 21] intentionally omitted <==**

**==> picture [178 x 21] intentionally omitted <==**

**Dual-Purpose Loss** . We introduce a dual-purpose loss function designed to improve model performance on hard tokens while mitigating overfitting on memorized tokens. This loss function combines two components: the learning loss and the unlearning loss. The learning loss is the standard causal language modeling (CLM) loss applied to the hard tokens _Th_ . The unlearning loss, in contrast, is the negative CLM loss applied to the memorized tokens _Tm_ , effectively performing gradient ascent. The dual-purpose loss is defined as follows, where _α >_ 0 is a hyper-parameter that balances the learning and unlearning losses 

**==> picture [190 x 12] intentionally omitted <==**

## **5 Experiments and Results** 

## **5.1 Experiment Settings** 

**Datasets** . We conduct experiments on two datasets: CC-news[2] and Wikipedia[3] . CC-news is a large collection of news articles which includes diverse topics and reflects real-world temporal events. Meanwhile, Wikipedia covers general knowledge across a wide range of disciplines, such as history, science, arts, and popular culture. 

**LLMs** : We experiment on three models including GPT-2 (124M) (Radford et al., 2019), Pythia (1.4B) (Biderman et al., 2023), and Llama-2 (7B) (Touvron and et al., 2023). This selection of models ensures a wide range of model sizes from small to large that allows us to analyze scaling effects and generalizability across different capacities. 

> 2Huggingface: vblagoje/cc_news 

> 3Huggingface: legacy-datasets/Wikipedia 

**Evaluation Metrics** . For evaluating language modeling performance, we measure perplexity (PPL), as it reflects the overall effectiveness of the model and is often correlated with improvements in other downstream tasks (Kaplan et al., 2020; OpenAI, 2020). For defense effectiveness, we consider the attack area under the curve (AUC) value and True Positive Rate (TPR) at low False Positive Rate (FPR). In total, we perform 4 MIAs with different MIA signals. Given the sample _x_ , the MIA signal function _f_ is formulated as follows: 

_•_ Loss (Yeom et al., 2018) utilizes the negative cross entropy loss as the MIA signal. 

**==> picture [103 x 11] intentionally omitted <==**

_•_ Ref-Loss (Carlini et al., 2021a) considers the loss differences between the target model and the attack reference model. To enhance the generality, our experiments ensure there is no data contamination between the training data of the target, reference, and attack models. 

**==> picture [156 x 12] intentionally omitted <==**

_•_ Min-K (Shi et al., 2024) leverages top K tokens that have the lowest probabilities. 

**==> picture [219 x 30] intentionally omitted <==**

_•_ Zlib (Carlini et al., 2021a) calibrates the loss signal with the zlib compression size. 

**==> picture [136 x 12] intentionally omitted <==**

**Baselines** . We present the results of four baselines. _Base_ refers to the pretrained LLM without fine-tuning. _FT_ represents the standard causal language modeling without protection. _Goldfish_ (Hans et al., 2024) implements a masking mechanism. _DPSGD_ (Abadi et al., 2016; Yu et al., 2022) applies gradient clipping and injects noise to achieve sample-level differential privacy. 

**Implementation** . We conduct full fine-tuning for GPT-2 and Pythia. For computing efficiency, Llama-2 fine-tuning is implemented using LowRank Adaptation (LoRA) (Hu et al., 2022a) which leads to ~4.2M trainable parameters. Additionally, we use subsets of 3K samples to fine-tune the LLMs. The data used to train DuoLearn’s reference model is disjoint from either the target model’s or the reference attack model’s training data. We present additional implementation details in Appendix C.1. 

22876 

|**LLM**<br>**Method**|**Wikipedia**<br>PPL<br>L<br>Rf<br>Mik<br>Zlib|**CC-news**<br>PPL<br>L<br>Rf<br>Mik<br>Zlib|
|---|---|---|
||oss<br>e<br>n-<br>|oss<br>e<br>n-<br>|
|GPT2<br>_Base_<br>124M<br>FT<br>Goldfsh<br>DPSGD<br>DuoLearn|_34.429_<br>_0.473_<br>_0.513_<br>_0.446_<br>_0.497_<br>**12.729**<br>0.577<br>0.967<br>0.489<br>0.544<br>12.853<br>0.565<br>0.954<br>0.486<br>0.537<br>18.523<br>0.463<br>0.536<br>**0.448**<br>0.491<br>13.628<br>**0.454**<br>**0.463**<br>0.470<br>**0.485**|_29.442_<br>_0.505_<br>_0.498_<br>_0.520_<br>_0.500_<br>**21.861**<br>0.607<br>0.855<br>0.549<br>0.569<br>21.902<br>0.608<br>0.855<br>0.547<br>0.570<br>26.022<br>0.507<br>0.513<br>**0.521**<br>0.502<br>23.733<br>**0.502**<br>**0.495**<br>0.529<br>**0.499**|
|Pythia<br>_Base_<br>1.4B<br>FT<br>Goldfsh<br>DPSGD<br>DuoLearn|_10.287_<br>_0.466_<br>_0.503_<br>_0.464_<br>_0.489_<br>**6.439**<br>0.578<br>0.985<br>0.484<br>0.557<br>6.465<br>0.564<br>0.981<br>0.482<br>0.546<br>7.751<br>0.469<br>0.524<br>**0.462**<br>0.488<br>6.553<br>**0.468**<br>**0.485**<br>0.472<br>**0.485**|_13.973_<br>_0.507_<br>_0.512_<br>_0.528_<br>_0.501_<br>11.922<br>0.602<br>0.857<br>0.541<br>0.574<br>**11.903**<br>0.609<br>0.862<br>0.543<br>0.579<br>13.286<br>0.512<br>0.531<br>0.528<br>0.503<br>12.670<br>**0.501**<br>**0.460**<br>**0.524**<br>**0.499**|
|Llama-2<br>_Base_<br>7B<br>FT<br>Goldfsh<br>DPSGD<br>DuoLearn|_7.014_<br>_0.458_<br>_0.491_<br>_0.476_<br>_0.488_<br>**3.830**<br>0.524<br>0.936<br>0.494<br>0.530<br>3.839<br>0.518<br>0.929<br>0.492<br>0.525<br>4.490<br>0.466<br>0.516<br>**0.470**<br>0.487<br>4.006<br>**0.458**<br>**0.440**<br>0.473<br>**0.480**|_9.364_<br>_0.505_<br>_0.495_<br>_0.516_<br>_0.503_<br>**6.261**<br>0.559<br>0.798<br>0.536<br>0.548<br>6.280<br>0.552<br>0.780<br>0.533<br>0.541<br>6.777<br>0.509<br>0.538<br>0.523<br>0.504<br>6.395<br>**0.507**<br>**0.482**<br>**0.518**<br>**0.500**|



Table 1: Overall Evaluation: Perplexity (PPL) and AUC scores of the MIAs with different signals (Loss/Ref/Mink/Zlib). For all metrics, the lower the value, the better the result. _Base_ in the method column indicates the pretrained LLMs without fine-tuning, thus it indicates lower bound for both utility and privacy risk. 

## **5.2 Overall Evaluation** 

Table 1 provides the overall evaluation compared to several baselines across large language model architectures and datasets. Among these two datasets, CCNews is more challenging, which leads to higher perplexity for all LLMs and fine-tuning methods. Additionally, the reference-model-based attack performs the best and demonstrates high privacy risks with attack AUC on the conventional fine-tuned models at 0.95 and 0.85 for Wikipedia and CCNews, respectively. Goldfish achieves similar PPL to the conventional FT method but fails to defend against MIAs. This aligns with the reported results by Hans et al. (2024) that Goldfish resists exact match attacks but only marginally affects MIAs. DPSGD provides a very strong protection in all settings (AUC < 0.55) but with a significant PPL tradeoff. Our proposed DuoLearn guarantees a robust protection, even slightly better than DPSGD, but with a notably smaller tradeoff on language modeling performance. For example, on the Wikipedia dataset, DuoLearn delivers perplexity reduction by 15% to 27%. Moreover, Table 5 (Appendix D) provides the TPR at 1% FPR. Both DPSGD and DuoLearn successfully reduce the TPR to _∼_ 0.02 for all LLMs and datasets. _Overall, across multiple LLM architectures and datasets, DuoLearn consistently offers ideal privacy protection with little trade-off in language modeling performance._ 

**Privacy-Utility Trade-off.** To investigate the 

privacy-utility trade-off of the methods, we vary the hyper-parameters of the fine-tuning methods. Particularly, for DPSGD, we adjust the privacy budget _ϵ_ from (8, 1e-5)-DP to (100, 1e-5)-DP. We modify the masking percentage of Goldfish from 25% to 50%. Additionally, we vary the loss weight _α_ from 0.2 to 0.8 for DuoLearn. Figure 3 depicts the privacy-utility trade-off for GPT2 on the CCNews dataset. Goldfish, with very large masking rate (50%), can slightly reduce the risk of the reference attack but can increase the risks of other attacks. By varying the weight _α_ , DuoLearn offers an adjustable trade-off between privacy protection and language modeling performance. DuoLearn largely dominates DPSGD and improves the language modeling performance by around 10% with the ideal privacy protection against MIAs. 

## **5.3 Ablation Study** 

**DuoLearn without reference models.** To study the impact of the reference model, we adapt DuoLearn to a non-reference version which directly uses the loss of the current training model (i.e., _s_ ( _ti_ ) = _LCE_ ( _θ_ ; _ti_ )) to select the learning and unlearning tokens. This means the unlearning tokens are the tokens that have smallest loss values. Figure 4 presents the training loss and testing perplexity. There is an inconsistent trend of the training loss and testing perplexity. Although the training loss decreases overtime, the test perplexity increases. This result indicates that identifying 

22877 

**==> picture [220 x 184] intentionally omitted <==**

**----- Start of picture text -----**<br>
random guess Goldfish DPSGD<br>Base model Fine-tuned model DuoLearn<br>0.60 0.80<br>0.70<br>0.55<br>0.60<br>0.50 0.50<br>22 24 26 28 22 24 26 28<br>Eval PPL Eval PPL<br>0.58 random guess<br>Base model<br>0.54 0.56 Goldfish Fine-tuned model<br>DPSGD<br>0.54 DuoLearn<br>0.52<br>0.52<br>0.50 0.50<br>22 24 26 28 22 24 26 28<br>Eval PPL Eval PPL<br>Loss-Attack AUC Ref-Attack AUC<br>MinK-Attack AUC Zlib-Attack AUC<br>**----- End of picture text -----**<br>


Figure 3: Privacy-utility trade-off of the methods while varying hyper-parameters. The Goldfish masking rate is set to 25%, 33%, and 50%. The privacy budget _ϵ_ of DPSGD is evaluated at 8, 16, 50, and 100. The weight _α_ of DuoLearn is varied at 0.0, 0.1, 0.2, 0.5, and 0.8. 

appropriate unlearning tokens without a reference model is challenging and conducting unlearning on an incorrect set hurts the language modeling performance. 

**==> picture [160 x 108] intentionally omitted <==**

**----- Start of picture text -----**<br>
60<br>5.0<br>50<br>Training loss<br>4.5 Testing Perplexity<br>40<br>4.0<br>30<br>0 200 400 600<br>Training Step<br>Training loss<br>Testing Perplexity<br>**----- End of picture text -----**<br>


Figure 4: Training Loss and Test Perplexity of DuoLearn without a reference model. 

## **DuoLearn with out-of-domain reference models.** 

To examine the influence of the distribution gap in the reference model, we replace the in-domain trained reference model with the original pretrained base model. Figure 5 depicts the language modeling performance and privacy risks in this study. DuoLearn with an out-of-domain reference model can reduce the privacy risks but yield a significant gap in language modeling performance compared to DuoLearn using an in-domain reference model. **DuoLearn without Unlearning.** To study the effects of unlearning tokens, we implement DuoLearn which use the first term of the loss only ( _Lθ_ = _LCE_ ( _θ_ ; _Th_ )). Figure 5 provides the perplexity and MIA AUC scores in this setting. Generally, without gradient ascent, DuoLearn can marginally 

reduce membership inference risks while slightly improving the language modeling performance. The token selection serves as a regularizer that helps to improve the language modeling performance. Additionally, tokens that are learned well in previous epochs may not be selected in the next epochs. This slightly helps to not amplify the memorization on these tokens over epochs. 

**==> picture [137 x 105] intentionally omitted <==**

**----- Start of picture text -----**<br>
DuoLearn (out-of-domain)<br>0.8 DuoLearn (in-domain)<br>DuoLearn (w/o unlearning)<br>FT<br>0.7<br>0.6<br>0.5<br>22 24 26 28 30<br>Testing PPL<br>Ref-Attack AUC<br>**----- End of picture text -----**<br>


Figure 5: Privacy-utility trade-off of DuoLearn with different settings: in-domain reference model, out-domain reference model, and without unlearning 

## **5.4 Training Dynamics** 

**Memorization and Generalization Dynamics** . Figure 6 (left) illustrates the training dynamics of conventional fine-tuning and DuoLearn, while Figure 6 (middle) depicts the membership inference risks. Generally, the gap between training and testing loss of conventional fine-tuning steadily increases over time, leading to model overfitting and high privacy risks. In contrast, DuoLearn maintains a stable equilibrium where the gap remains more than 10 times smaller. This equilibrium arises from the dual-purpose loss, which balances learning on hard tokens while actively unlearning memorized tokens. By preventing excessive memorization, DuoLearn mitigates membership inference risks and enhances generalization. 

**Gradient Conflicts** . To study the conflict between the learning and unlearning objectives in our dualpurpose loss function, we compute the gradient for each objective separately. We then calculate the cosine similarity of these two gradients. Figure 6 (right) provides the cosine similarity between two gradients over time. During training, the cosine similarity typically ranges from -0.15 to 0.15. This indicates a mix of mild conflicts and nearorthogonal updates. On average, it decreases from 0.05 to -0.1. This trend reflects increasing gradient misalignment. Early in training, the model may not have strongly learned or memorized specific tokens, so the conflicts are weaker. Overtime, as the 

22878 

**==> picture [433 x 108] intentionally omitted <==**

**----- Start of picture text -----**<br>
3.3 0.9 0.2<br>3.2<br>0.1<br>0.8<br>3.1<br>3.0 0.7 FT 0.0<br>DuoLearn<br>2.9 FT training 0.1<br>FT testing 0.6<br>2.8<br>DuoLearn training<br>0.2<br>2.7 DuoLearn testing<br>0.5<br>0 200 400 600 800 0 200 400 600 800 0 200 400 600 800 1000<br>Training Step Training Step Training Step<br>)CE<br>Ref-Attack AUC Cosine Similarity<br>CE Loss Value (<br>**----- End of picture text -----**<br>


Figure 6: Training dynamics of DuoLearn and the conventional fine-tuning approach. The left and middle figures provide the training-testing gap and membership inference risks, respectively. The testing _LCE_ of FT and training _LCE_ of DuoLearn are significantly overlapping, we provide the breakdown in Figure 11 in Appendix D. The right figure depicts the cosine similarity of the learning and unlearning gradients of DuoLearn. Cosine similarity of 1 means entire alignment, 0 indicates orthogonality, and -1 presents full conflict. 

model learns more and memorization grows, the divergence between hard and memorized tokens increases, making the gradients less aligned. This gradient conflict is the root of the small degradation of language modeling performance of DuoLearn compared to the conventional fine-tuning approach. **Token Selection Dynamics** . Figure 7 illustrates the token selection dynamics of DuoLearn during training. The figure shows that the token selection process is dynamic and changes over epochs. In particular, some tokens are selected for unlearning from the beginning to the end of the training. This indicates that a token, even without being selected as a learning token initially, can be learned and memorized through the connections with other tokens. This also explains that simple masking as in Goldfish is not sufficient to protect against MIAs. Additionally, there are a significant number of tokens that are selected for learning in the early epochs but selected for unlearning in the later epochs. This indicates that the model gradually memorizes these tokens over epochs, and the during-training unlearning process is essential to mitigate the memorization risks. 

## **5.5 Privacy Backdoor** 

To study the worst case of privacy attacks and defense effectiveness under the state-of-the-art MIA, we perform a privacy backdoor – Precurious (Liu et al., 2024b). In this setup, the target model undergoes continual fine-tuning from a warm-up model. The attacker then applies a reference-based MIA that leverages the warm-up model as the attack’s reference. Table 2 shows the language modeling and MIA performance on CCNews with GPT2. Precurious increases the MIA AUC score by 5%. Goldfish achieves the lowest PPL, aligning 

**==> picture [154 x 157] intentionally omitted <==**

**----- Start of picture text -----**<br>
0<br>50<br>100<br>150<br>200<br>250<br>0 160 320 480 640 800<br>Training Step<br>Learning<br>Token Index<br>Unused<br>Unlearning<br>**----- End of picture text -----**<br>


Figure 7: Token Selection Dynamics of DuoLearn 

with Hans et al. (2024), where the Goldfish masking mechanism acts as a regularizer that potentially enhances generalization. Both DPSGD and DuoLearn provide strong privacy protection, with DuoLearn offering slightly better defense while maintaining lower perplexity than DPSGD. 

|**Metric**|**WU**<br>**FT**<br>**GF**<br>**DP**<br>**DuoL**|
|---|---|
|**PPL**<br>**AUC**|_23.318_<br>21.593<br>**21.074**<br>23.279<br>22.296<br>_0.500_<br>0.911<br>0.886<br>0.533<br>**0.499**|



Table 2: Experimental results of privacy backdoor for GPT2 on the CC-news dataset. WU stands for the warmup model leveraged by Precurious. GF, DP, and DuoL are abbreviations of Goldfish, DPSGD, and DuoLearn 

## **5.6 Pretraining** 

We conduct a small-scale pretraining experiment using a Llama-like architecture with 1.5 billion parameters. The experiment is to pretrain on a dataset of 1 billion tokens. We reuse the dataset 

22879 

|**Method**|**Eval Loss(↓)**|**MIA AUC(↓)**|**TPR @ 1% FPR(↓)**|**Training Time(↓)**|
|---|---|---|---|---|
|_Untrained model_<br>Conventional LM<br>Goldfsh<br>DuoLearn|_11.329_<br>3.003<br>**2.844**<br>3.244|_0.517_<br>0.930<br>0.905<br>**0.548**|_0.010_<br>0.376<br>0.426<br>**0.040**|_N/A_<br>**10.095**hours<br>10.159 hours<br>12.278 hours|



Table 3: Comparison of Methods on Evaluation Loss, Privacy Metrics, and Training Time for pretraining. 

and codebase developed by Sanyal et al. (2024)[4] . The pretraining corpus is collected from various sources and domains, including arXiv, books, Common Crawl, GitHub, StackExchange, and Wikipedia (Weber et al., 2024). To train the reference model, we use 10% of the data. Table 3 presents the performance of DuoLearn in this pretraining. Generally, DuoLearn successfully mitigates the MIA risk, reducing AUC from 0.9 to 0.55 and TPR at 1% FPR from 0.4 to 0.04, with minimal degradation on the model performance. 

## **5.7 Low-Prevalence Tokens vs High-Prevalence Tokens** 

To understand the bias of the methods towards lowprevalence and high-prevalence tokens, we conduct some visualization on the token-level MIA signals and token frequencies. We selected a set of 500 samples in the training data. We split the tokens into two sets, high-frequency and low-frequency, then visualize kernel density estimation (KDE) separately, illustrated in Figure 8. Notably, both conventional FT and DuoLearn yield different privacy risks for the two groups. The low-prevalence tokens have higher per-token MIA risks (with MIA signal ranging from -5 to 2, compared to -0.5 to 0.5 of high-prevalence tokens). DuoLearn successfully shifts the MIA-signal distribution to be centered at zero for both high-frequency and low-frequency tokens. This indicates that DuoLearn reduces the majority of MIA risks across both token groups. However, it slightly increases the variance of risks for both groups. 

## **6 Conclusion** 

We introduced DuoLearn, an effective training framework defending against MIAs for LLMs. The extensive experiments demonstrate its robustness in protecting privacy while maintaining strong language modeling performance across various datasets and architectures. Although our study fo- 

> 4Following this setting and due to limited computing resources, we use a batch size of 8 which is much smaller than practical pretraining. All methods implement the same learning rate and are evaluated at their 25K-th iteration. 

**==> picture [219 x 155] intentionally omitted <==**

**----- Start of picture text -----**<br>
5000 High-Prevalence Tokens 5000 High-Prevalence Tokens<br>Conventional FT DuoLearn<br>2500 2500<br>0.5 0.0 0.5 0.5 0.0 0.5<br>MIA Signal MIA Signal<br>Low-Prevalence Tokens Low-Prevalence Tokens<br>40 40<br>Conventional FT DuoLearn<br>20 20<br>0 0<br>2.5 0.0 2.5 2.5 0.0 2.5<br>MIA Signal MIA Signal<br>Frequency Frequency<br>Frequency Frequency<br>**----- End of picture text -----**<br>


Figure 8: Effects on high-prevalence and lowprevalence tokens. For MIA signal, we consider the Ref-Loss attack, a signal value closer to zero indicates lower risk. 

cuses on fine-tuning and small scale pretraining due to computational constraints, DuoLearn can be seamlessly applied to large-scale pretraining, as in prior selective pretraining work (Lin et al., 2024). By categorizing tokens and treating them appropriately, DuoLearn opens a novel pathway for MIA defense. Future work can explore improved token selection strategies and multi-objective training approaches. 

## **Limitations** 

The main limitation of our work is the small-scale experiment settings due to limited computing resources. However, we believe DuoLearn can be directly applied to large-scale pretraining without any modifications, as in previous reference-modelbased pretraining study (Lin et al., 2024). Another limitation is the reference model, which may be restrictive in highly sensitive or domain-limited settings (Tramèr et al., 2024). From a technical perspective, while DuoLearn performs well across different datasets and architectures, there is room for improvement. For example, future work could explore adaptive selection size or weighted token contribution. Additionally, as DuoLearn is an empirical defense, future work can investigate the convergence and overfitting analysis. 

22880 

## **Acknowledgment** 

This work is partially supported by the National Science Foundation under Award Numbers 2302968, 2124104, and 2125530, and by the National Institutes of Health under Award Numbers R01ES033241 and R01LM013712. The views and opinions expressed in this paper are those of the authors and do not necessarily reflect the views of the U.S. Government or any agency thereof. 

## **References** 

- Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. 2016. Deep learning with differential privacy. In _Proceedings of the 2016 ACM SIGSAC conference on computer and communications security_ , pages 308–318. 

- Guy Amit, Abigail Goldsteen, and Ariel Farkash. 2024. Sok: Reducing the vulnerability of fine-tuned language models to membership inference attacks. _Preprint_ , arXiv:2403.08481. 

- Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar Van Der Wal. 2023. Pythia: a suite for analyzing large language models across training and scaling. In _Proceedings of the 40th International Conference on Machine Learning_ , ICML’23. 

- Zhiqi Bu, Justin Chiu, Ruixuan Liu, Sheng Zha, and George Karypis. 2023a. Zero redundancy distributed learning with differential privacy. _arXiv preprint arXiv:2311.11822_ . 

- Zhiqi Bu, Ruixuan Liu, Yu-Xiang Wang, Sheng Zha, and George Karypis. 2023b. On the accuracy and efficiency of group-wise clipping in differentially private optimization. _Preprint_ , arXiv:2310.19215. 

- Zhiqi Bu, Yu-Xiang Wang, Sheng Zha, and George Karypis. 2023c. Automatic clipping: Differentially private deep learning made easier and stronger. In _Thirty-seventh Conference on Neural Information Processing Systems_ . 

- Zhiqi Bu, Yu-Xiang Wang, Sheng Zha, and George Karypis. 2023d. Differentially private optimization on large model at small cost. In _Proceedings of the 40th International Conference on Machine Learning_ , ICML’23. JMLR.org. 

- Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. 2023. Quantifying memorization across neural language models. In _The Eleventh International Conference on Learning Representations_ . 

- Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel. 2021a. Extracting training data from large language models. In _30th USENIX Security Symposium (USENIX Security 21)_ , pages 2633–2650. USENIX Association. 

- Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021b. Extracting training data from large language models. In _30th USENIX Security Symposium (USENIX Security 21)_ , pages 2633– 2650. 

- Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2024. A survey on evaluation of large language models. _ACM Trans. Intell. Syst. Technol._ , 15(3). 

- Cody Coleman, Christopher Yeh, Stephen Mussmann, Baharan Mirzasoleiman, Peter Bailis, Percy Liang, Jure Leskovec, and Matei Zaharia. 2020. Selection via proxy: Efficient data selection for deep learning. In _International Conference on Learning Representations_ . 

- Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, and Hannaneh Hajishirzi. 2024. Do membership inference attacks work on large language models? In _Conference on Language Modeling (COLM)_ . 

- Qizhang Feng, Siva Rajesh Kasa, Santhosh Kumar Kasa, Hyokun Yun, Choon Hui Teo, and Sravan Babu Bodapati. 2025. Exposing privacy gaps: Membership inference attack on preference data for llm alignment. _Preprint_ , arXiv:2407.06443. 

- Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, and Tao Jiang. 2024. Membership inference attacks against fine-tuned large language models via self-prompt calibration. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ . 

- Abhimanyu Hans, Yuxin Wen, Neel Jain, John Kirchenbauer, Hamid Kazemi, Prajwal Singhania, Siddharth Singh, Gowthami Somepalli, Jonas Geiping, Abhinav Bhatele, and Tom Goldstein. 2024. Be like a goldfish, don’t memorize! mitigating memorization in generative llms. _ArXiv_ , abs/2406.10209. 

- Jamie Hayes, Ilia Shumailov, Christopher A. ChoquetteChoo, Matthew Jagielski, George Kaissis, Katherine Lee, Milad Nasr, Sahra Ghalebikesabi, Niloofar Mireshghallah, Meenatchi Sundaram Mutu Selva Annamalai, Igor Shilov, Matthieu Meeus, YvesAlexandre de Montjoye, Franziska Boenisch, Adam Dziedzic, and A. Feder Cooper. 2025. Strong membership inference attacks on massive datasets 

22881 

and (moderately) large language models. _Preprint_ , arXiv:2505.18773. 

- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022a. LoRA: Low-rank adaptation of large language models. In _International Conference on Learning Representations_ . 

- Hongsheng Hu, Zoran Salcic, Lichao Sun, Gillian Dobbie, Philip S. Yu, and Xuyun Zhang. 2022b. Membership inference attacks on machine learning: A survey. _ACM Comput. Surv._ , 54(11s). 

- Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, and Matt Kusner. 2023. No train no gain: Revisiting efficient training algorithms for transformerbased language models. In _Thirty-seventh Conference on Neural Information Processing Systems_ . 

- Nikhil Kandpal, Eric Wallace, and Colin Raffel. 2022. Deduplicating training data mitigates privacy risks in language models. In _International Conference on Machine Learning_ , pages 10697–10707. PMLR. 

- Feiyang Kang, Hoang Anh Just, Yifan Sun, Himanshu Jahagirdar, Yuanzhi Zhang, Rongxing Du, Anit Kumar Sahu, and Ruoxi Jia. 2024. Get more for less: Principled data selection for warming up fine-tuning in LLMs. In _The Twelfth International Conference on Learning Representations_ . 

- Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. _Preprint_ , arXiv:2001.08361. 

- Angelos Katharopoulos and François Fleuret. 2018. Not all samples are created equal: Deep learning with importance sampling. In _Proceedings of the 35th International Conference on Machine Learning_ , pages 2525–2534. PMLR. 

- Kenji Kawaguchi and Haihao Lu. 2020. Ordered sgd: A new stochastic optimization framework for empirical risk minimization. In _Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics_ , volume 108 of _Proceedings of Machine Learning Research_ , pages 669–679. PMLR. 

- Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. 2022. Deduplicating training data makes language models better. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 8424–8445. Association for Computational Linguistics. 

- Haoran Li, Yulin Chen, Jinglong Luo, Jiecong Wang, Hao Peng, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, Zenglin Xu, Bryan Hooi, and Yangqiu Song. 2024a. Privacy in large language models: Attacks, defenses and future directions. _Preprint_ , arXiv:2310.10383. 

- Qinbin Li, Junyuan Hong, Chulin Xie, Jeffrey Tan, Rachel Xin, Junyi Hou, Xavier Yin, Zhun Wang, Dan Hendrycks, Zhangyang Wang, Bo Li, Bingsheng He, and Dawn Song. 2024b. Llm-pbe: Assessing data privacy in large language models. _Preprint_ , arXiv:2408.12787. 

- Xuechen Li, Daogao Liu, Tatsunori B Hashimoto, Huseyin A Inan, Janardhan Kulkarni, Yin-Tat Lee, and Abhradeep Guha Thakurta. 2022a. When does differentially private learning not suffer in high dimensions? _Advances in Neural Information Processing Systems_ , 35:28616–28630. 

- Xuechen Li, Florian Tramer, Percy Liang, and Tatsunori Hashimoto. 2022b. Large language models can be strong differentially private learners. In _International Conference on Learning Representations_ . 

- Yunshui Li, Binyuan Hui, Xiaobo Xia, Jiaxi Yang, Min Yang, Lei Zhang, Shuzheng Si, Ling-Hao Chen, Junhao Liu, Tongliang Liu, Fei Huang, and Yongbin Li. 2024c. One-shot learning as instruction data prospector for large language models. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . Association for Computational Linguistics. 

- Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, yelong shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, and Weizhu Chen. 2024. Not all tokens are what you need for pretraining. In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_ . 

- Ruixuan Liu, Toan Tran, Tianhao Wang, Hongsheng Hu, Shuo Wang, and Li Xiong. 2024a. Expshield: Safeguarding web text from unauthorized crawling and language modeling exploitation. _Preprint_ , arXiv:2412.21123. 

- Ruixuan Liu, Tianhao Wang, Yang Cao, and Li Xiong. 2024b. Precurious: How innocent pre-trained language models turn into privacy traps. In _Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security_ , CCS ’24, page 3511–3524, New York, NY, USA. Association for Computing Machinery. 

- Ilya Loshchilov and Frank Hutter. 2016. Online batch selection for faster training of neural networks. _Preprint_ , arXiv:1511.06343. 

- Andrew Lowy, Zhuohang Li, Jing Liu, Toshiaki KoikeAkino, Kieran Parsons, and Ye Wang. 2024. Why does differential privacy with large epsilon defend against practical membership inference attacks? _Preprint_ , arXiv:2402.09540. 

- Nils Lukas, Ahmed Salem, Robert Sim, Shruti Tople, Lukas Wutschitz, and Santiago Zanella-Beguelin. 2023. Analyzing Leakage of Personally Identifiable Information in Language Models . In _2023 IEEE Symposium on Security and Privacy (SP)_ , pages 346– 363. 

22882 

- Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schoelkopf, Mrinmaya Sachan, and Taylor Berg-Kirkpatrick. 2023. Membership inference attacks against language models via neighbourhood comparison. In _Findings of the Association for Computational Linguistics: ACL 2023_ , pages 11330– 11343, Toronto, Canada. Association for Computational Linguistics. 

- Ryan McKenna, Yangsibo Huang, Amer Sinha, Borja Balle, Zachary Charles, Christopher A. ChoquetteChoo, Badih Ghazi, George Kaissis, Ravi Kumar, Ruibo Liu, Da Yu, and Chiyuan Zhang. 2025. Scaling laws for differentially private language models. _Preprint_ , arXiv:2501.18914. 

- Sören Mindermann, Jan M Brauner, Muhammed T Razzak, Mrinank Sharma, Andreas Kirsch, Winnie Xu, Benedikt Höltgen, Aidan N Gomez, Adrien Morisot, Sebastian Farquhar, and Yarin Gal. 2022. Prioritized training on points that are learnable, worth learning, and not yet learnt. In _Proceedings of the 39th International Conference on Machine Learning_ , pages 15630–15649. 

- Milad Nasr, Javier Rando, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Florian Tramèr, and Katherine Lee. 2025. Scalable extraction of training data from aligned, production language models. In _The Thirteenth International Conference on Learning Representations_ . 

- Milad Nasr, Reza Shokri, and Amir Houmansadr. 2018. Machine learning with membership privacy using adversarial regularization. In _Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security_ , CCS ’18, page 634–646, New York, NY, USA. Association for Computing Machinery. 

- OpenAI. 2020. Language models are few-shot learners. volume 33, pages 1877–1901. Curran Associates, Inc. 

- Ildikó Pilán, Pierre Lison, Lilja Øvrelid, Anthi Papadopoulou, David Sánchez, and Montserrat Batet. 2022. The text anonymization benchmark (tab): A dedicated corpus and evaluation framework for text anonymization. _Preprint_ , arXiv:2202.00443. 

- Haritz Puerto, Martin Gubri, Sangdoo Yun, and Seong Joon Oh. 2025. Scaling up membership inference: When and how attacks succeed on large language models. _Preprint_ , arXiv:2411.00154. 

- Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. 

- Sunny Sanyal, Ravid Shwartz-Ziv, Alexandros G. Dimakis, and Sujay Sanghavi. 2024. Inheritune: Training smaller yet more attentive language models. _Preprint_ , arXiv:2404.08634. 

- Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. 2024. Detecting pretraining data from large language models. In _The Twelfth International Conference on Learning Representations_ . 

- Weiyan Shi, Ryan Shea, Si Chen, Chiyuan Zhang, Ruoxi Jia, and Zhou Yu. 2022. Just fine-tune twice: Selective differential privacy for large language models. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , pages 6296–6311. Association for Computational Linguistics. 

- Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017. Membership Inference Attacks Against Machine Learning Models . In _2017 IEEE Symposium on Security and Privacy (SP)_ , pages 3–18, Los Alamitos, CA, USA. IEEE Computer Society. 

- Liwei Song and Prateek Mittal. 2021. Systematic evaluation of privacy risks of machine learning models. In _30th USENIX Security Symposium (USENIX Security 21)_ , pages 2615–2632. USENIX Association. 

- Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: A simple way to prevent neural networks from overfitting. _Journal of Machine Learning Research_ , 15(56):1929–1958. 

- Xinyu Tang, Saeed Mahloujifar, Liwei Song, Virat Shejwalkar, Milad Nasr, Amir Houmansadr, and Prateek Mittal. 2022. Mitigating membership inference attacks by Self-Distillation through a novel ensemble architecture. In _31st USENIX Security Symposium (USENIX Security 22)_ , pages 1433–1450, Boston, MA. USENIX Association. 

- Kushal Tirumala, Aram H. Markosyan, Luke Zettlemoyer, and Armen Aghajanyan. 2022. Memorization without overfitting: Analyzing the training dynamics of large language models. In _Advances in Neural Information Processing Systems_ . 

- Hugo Touvron and Louis Martin et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _Preprint_ , arXiv:2307.09288. 

- Florian Tramèr, Gautam Kamath, and Nicholas Carlini. 2024. Position: Considerations for differentially private learning with large-scale public pretraining. In _Forty-first International Conference on Machine Learning_ . 

- Maurice Weber, Daniel Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, and Ce Zhang. 2024. Redpajama: an open dataset for training large language models. _Preprint_ , arXiv:2411.12372. 

22883 

- Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. 2023. Doremi: Optimizing data mixtures speeds up language model pretraining. In _Thirty-seventh Conference on Neural Information Processing Systems_ . 

- Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Zhibo Sun, and Yue Zhang. 2024. A survey on large language model (llm) security and privacy: The good, the bad, and the ugly. _High-Confidence Computing_ , 4(2):100211. 

- Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. 2018. Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting . In _2018 IEEE 31st Computer Security Foundations Symposium (CSF)_ , pages 268–282, Los Alamitos, CA, USA. IEEE Computer Society. 

- Da Yu, Saurabh Naik, Arturs Backurs, Sivakanth Gopi, Huseyin A Inan, Gautam Kamath, Janardhan Kulkarni, Yin Tat Lee, Andre Manoel, Lukas Wutschitz, Sergey Yekhanin, and Huishuai Zhang. 2022. Differentially private fine-tuning of language models. In _International Conference on Learning Representations_ . 

- Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Frank Yang, and Hai Li. 2025. Min-k%++: Improved baseline for pretraining data detection from large language models. In _The Thirteenth International Conference on Learning Representations_ . 

## **A Additional Related Works** 

## **A.1 Training Data Selection** 

Training data selection are methods that filter highquality data from noisy big data _before training_ to improve the model utility and training efficiency. There are several works leveraging reference models (Coleman et al., 2020; Xie et al., 2023), prompting LLMs (Li et al., 2024c), deduplication (Lee et al., 2022; Kandpal et al., 2022), and distribution matching (Kang et al., 2024). However, we do not aim to cover this data selection approach, as it is orthogonal and can be combined with ours. 

## **A.2 Selective Training** 

Selective training refers to methods that _dynamically choose_ specific samples or tokens _during training_ . Selective training methods are the most relevant to our work. Generally, sample selection has been widely studied in the context of traditional classification models via online batch selection (Loshchilov and Hutter, 2016; Katharopoulos and Fleuret, 2018; Kawaguchi and Lu, 2020). These batch selection methods replace the naive 

random mini-batch sampling with mechanisms that consider the importance of each sample mainly via their loss values. Mindermann et al. (2022) indeed choose highly important samples from regular random batches by utilizing a reference model. However, due to the sequential nature of LLMs, which makes the training significantly different from the traditional classification ML, sample-level selection is not effective for language modeling (Kaddour et al., 2023). Lin et al. (2024) extend the reference model-based framework to select meaningful tokens within batches. All of the previous methods for selective training aim to improve the training performance and compute efficiency. Our work is the first looking at this aspect for defending against MIAs. 

## **B Token-level membership inference risk analysis** 

Figures 9 and 10 present the analysis for additional samples. Generally, the trends are consistent with the one presented in Section 3. 

## **C Experiment settings** 

## **C.1 Implementation details** 

_•_ **FT** . We implement the conventional fine tuning using Huggingface Trainer. We manually tune the learning rate to make sure no significant underfitting or overfitting. The batch size is selected appropriately to fit the physical memory and comparable with the other methods’. 

_•_ **Goldfish** . Goldfish is also implemented with Huggingface Trainer, where we custom the compute_loss function. We implement the deterministic masking version rather than the random masking to make sure the same tokens are masked over epochs, potentially leading to better preventing memorization. The learning rate is also manually tuned, we noticed that the optimal Goldfish learning rate is usually slightly greater than FT’s. This can be the gradients of two methods are almost similar, Goldfish just removes some tokens’ contribution to the loss calculation. The batch size of FT can set as the same as FT, as Goldfish does not have significant overhead on memory. 

_•_ **DPSGD** . DPSGD is implemented by FastDP (Bu et al., 2023a). We implement DPSGD with fastDP (Bu et al., 2023a) which offers state-ofthe-art efficiency in terms of memory and training speed. We also use automatic clipping (Bu et al., 2023c) and a mixed optimization strategy (Bu et al., 

22884 

**==> picture [455 x 340] intentionally omitted <==**

**----- Start of picture text -----**<br>
40<br>30 40<br>30<br>20<br>20 20<br>10 10<br>0 0 0<br>3 2 1 0 1 2 3 3 2 1 0 1 2 3 3 2 1 0 1 2 3<br>MIA Signal MIA Signal MIA Signal<br>30<br>40<br>30<br>30 20<br>20<br>20<br>10 10<br>10<br>0 0 0<br>3 2 1 0 1 2 3 3 2 1 0 1 2 3 3 2 1 0 1 2 3<br>MIA Signal MIA Signal MIA Signal<br>30<br>40<br>40<br>30 20<br>20 20<br>10<br>10<br>0 0 0<br>3 2 1 0 1 2 3 3 2 1 0 1 2 3 3 2 1 0 1 2 3<br>MIA Signal MIA Signal MIA Signal<br>40 40 60<br>30 30 40<br>20 20<br>20<br>10 10<br>0 0 0<br>3 2 1 0 1 2 3 3 2 1 0 1 2 3 3 2 1 0 1 2 3<br>MIA Signal MIA Signal MIA Signal<br>Number of Tokens Number of Tokens Number of Tokens<br>Number of Tokens Number of Tokens Number of Tokens<br>Number of Tokens Number of Tokens Number of Tokens<br>Number of Tokens Number of Tokens Number of Tokens<br>**----- End of picture text -----**<br>


Figure 9: Histograms of MIA signal of tokens. Each figure depicts a sample. Blue means the member samples while orange represents the non-member samples. We limited the y-axis range to -3 to 3 for better visibility, so it can result in missing several non-significant outliers. 

**==> picture [128 x 99] intentionally omitted <==**

**==> picture [128 x 99] intentionally omitted <==**

**==> picture [137 x 99] intentionally omitted <==**

Figure 10: MIA signal ranking of tokens during training. Each figure illustrates a sample. 

2023d) between per-layer and per-sample clipping for robust performance and stability. 

_•_ **DuoLearn** . We implement DuoLearn using Huggingface Trainer, same as FT and Goldfish. The learning is reused from FT. The batch size of DuoLearn is usually smaller than FT and Goldfish when the model becomes large such as Pythia and Llama 2 due to the reference model, which consumes some memory. 

For a fair comparison, we aim to implement the same batch size for all methods if feasible. In case of OOM (out of memory), we perform gradient 

accumulation, so all the methods can have comparable batch sizes. We provide the hyper-parameters of method for GPT2 in Table 4. For Pythia and Llama 2, the learning rate, batch size, and number of epochs are tuned again, but the hyper-parameters regarding the privacy mechanisms remain the same. To make sure there is no naive overfitting, we evaluate the methods by selecting the best models on a validation set. Moreover, the testing and attack datasets remains identical for evaluating all methods. Additionally, we balance the number of member and non-member samples for MIA evaluation. 

22885 

It is worth noting that for the ablation study and analysis, if not state, the default model architecture and dataset are GPT2 and CC-news. 

## **D Additional Results** 

**==> picture [154 x 114] intentionally omitted <==**

**----- Start of picture text -----**<br>
3.25 FT testing<br>DuoLearn training<br>3.20<br>3.15<br>3.10<br>0 200 400 600 800<br>Training Step<br>)CE<br>CE Loss Value (<br>**----- End of picture text -----**<br>


Figure 11: Breakdown to the cross entropy loss values of FT on the testing set and DuoLearn on the training set during training. 

## **D.1 Overall Evaluation** 

Table 5 provides the True Positive Rate (TPR) at low False Positive Rate (FPR) of the overall evaluation. Generally, compared to CC-news, Wikipedia poses a significant higher risk at low FPR. For example, the reference-based attack can achieve a score of 0.57 on GPT2 if no protection. In general, Goldfish fails to mitigate the risk in this scenario, while both DPSGD and DuoLearn offer robust protection. 

## **D.2 Auxiliary dataset** 

We investigate the size of the auxiliary dataset which is disjoint with the training data of the target model and the attack model. In this experiment, the other methods are trained with 3K samples. Figure 12 presents the language modeling performance while varying the auxiliary dataset’s size. The result demonstrates that the better reference model, the better language modeling performance. It is worth noting that even with a very small number of samples, DuoLearn can still outperform DPSGD. Additionally, there is only a little benefit when increasing from 1000 to 3000, this indicates that the reference model is not needed to be perfect, as it just serves as a calibration factor. This phenomena is consistent with previous selective training works (Lin et al., 2024; Mindermann et al., 2022). 

## **D.3 Hyperparameter sensitivity analysis** 

_•_ **Varying** _Th_ **and** _Tm_ – Portion of tokens for learning and for unlearning. We keep other hy- 

**==> picture [176 x 147] intentionally omitted <==**

**----- Start of picture text -----**<br>
Base FT DuoLearn<br>Goldfish DPSGD<br>28<br>26<br>24<br>22<br>500 1000 1500 2000 2500 3000<br>Size of the auxiliary dataset<br>Test PPL<br>**----- End of picture text -----**<br>


Figure 12: Language modeling performance while varying the auxiliary dataset’s size. Note that the results of FT and Goldfish are significantly overlapping. 

perparamters as default and adjust _Th_ and _Tm_ separately. Figures 13 and 14 provides the results of this experiment. Generally, DuoLearn is robust while varying these hyperparameters with PPL ranging from 22 to 23 and AUC ranging from 0.46 to 0.56. 

**==> picture [176 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
PPL and AUC while varying  h<br>0.56<br>26<br>0.54<br>25<br>PPL 0.52<br>24 PPL FT (AUC 0.86)<br>PPL DPSGD (AUC 0.51) 0.50<br>AUC<br>23 0.48<br>22 0.46<br>0.2 0.4 0.6 0.8<br>h [ (portion of tokens for learning)]<br>PPL AUC<br>**----- End of picture text -----**<br>


Figure 13: Performance of DuoLearn while varying _Th_ 

**==> picture [176 x 128] intentionally omitted <==**

**----- Start of picture text -----**<br>
PPL and AUC while varying  m<br>26<br>0.54<br>25 0.52<br>PPL<br>24 PPL FT (AUC 0.86) 0.50<br>PPL DPSGD (AUC 0.51)<br>AUC 0.48<br>23<br>0.46<br>22 0.44<br>0.1 0.2 0.3 0.4<br>h [ (portion of tokens for learning)]<br>PPL AUC<br>**----- End of picture text -----**<br>


Figure 14: Performance of DuoLearn while varying _Tm_ 

_•_ **Varying** _α_ – Weight balance factor. Figure 15 illustrates the peformance while varying _α_ . Intuitively, the smaller _α_ , the less unlearining performed, it leads to a better language modeling performance and higher privacy risk. When _α_ is unreasonably high (i.e., 1.5 or 2.0), the unlearning 

22886 

|**LLM**|**Method**<br>**Hyper-parameter**<br>**Value**|
|---|---|
|**GPT2**|FT<br>Learning rate<br>1.75e-5<br>Batch size<br>96<br>Gradient accumulation steps<br>1<br>Number of epochs<br>20|
||Goldfsh<br>Learning rate<br>2e-5<br>Batch size<br>96<br>Grad accumulation steps<br>1<br>Number of epochs<br>20<br>MaskingRate<br>25%|
||DPSGD<br>Learning rate<br>1.5e-3<br>Batch size<br>96<br>Grad accumulation steps<br>1<br>Number of epochs<br>10<br>Clipping<br>automatic clipping<br>Privacybudget<br>(8,1e-5)-DP|
||DuoLearn<br>Learning rate<br>1.75e-3<br>Batch size<br>96<br>Grad accumulation steps<br>1<br>Number of epochs<br>20<br>_Kh_<br>60%<br>_Km_<br>20%<br>_τ_<br>0<br>_α_<br>0.8|



Table 4: Hyper-parameters of the methods for GPT2. 

|**LLM**<br>**Method**|**Wikipedia**<br>PPL<br>Loss<br>Ref<br>min-k<br>zlib|**CC-news**<br>PPL<br>Loss<br>Ref<br>min-k<br>zlib|
|---|---|---|
|GPT2<br>_Base_<br>124M<br>FT<br>Goldfsh<br>DPSGD<br>DuoLearn|_34.429_<br>_0.002_<br>_0.014_<br>_0.010_<br>_0.002_<br>**12.729**<br>0.018<br>0.574<br>0.016<br>0.014<br>12.853<br>0.018<br>0.632<br>0.016<br>0.010<br>18.523<br>**0.004**<br>0.036<br>0.018<br>0.006<br>13.628<br>0.014<br>**0.010**<br>**0.014**<br>**0.004**|_29.442_<br>_0.018_<br>_0.002_<br>_0.022_<br>_0.006_<br>**21.861**<br>0.030<br>0.026<br>0.016<br>0.016<br>21.902<br>0.030<br>0.024<br>0.028<br>0.016<br>26.022<br>**0.018**<br>**0.004**<br>**0.018**<br>0.008<br>23.733<br>0.030<br>0.022<br>0.026<br>**0.006**|
|Pythia<br>_Base_<br>1.4B<br>FT<br>Goldfsh<br>DPSGD<br>DuoLearn|_10.287_<br>_0.002_<br>_0.014_<br>_0.006_<br>_0.008_<br>**6.439**<br>0.020<br>0.440<br>0.010<br>0.020<br>6.465<br>0.016<br>0.412<br>0.010<br>0.020<br>7.751<br>**0.004**<br>**0.016**<br>0.010<br>**0.004**<br>6.553<br>0.008<br>0.030<br>**0.006**<br>0.006|_13.973_<br>_0.002_<br>_0.008_<br>_0.020_<br>_0.014_<br>11.922<br>0.014<br>0.008<br>0.022<br>0.020<br>**11.903**<br>0.014<br>0.008<br>0.024<br>0.018<br>13.286<br>**0.002**<br>**0.004**<br>**0.018**<br>**0.014**<br>12.670<br>0.004<br>0.020<br>**0.018**<br>0.016|
|Llama-2<br>_Base_<br>7B<br>FT<br>Goldfsh<br>DPSGD<br>DuoLearn|_7.014_<br>_0.006_<br>_0.016_<br>_0.016_<br>_0.010_<br>**3.830**<br>0.028<br>0.170<br>0.030<br>0.028<br>3.839<br>0.028<br>0.198<br>0.028<br>0.028<br>4.490<br>**0.006**<br>0.014<br>**0.020**<br>**0.010**<br>4.006<br>0.010<br>**0.002**<br>0.028<br>0.012|_9.364_<br>_0.006_<br>_0.006_<br>_0.024_<br>_0.006_<br>**6.261**<br>0.002<br>0.018<br>0.002<br>0.002<br>6.280<br>0.002<br>0.018<br>0.002<br>0.006<br>6.777<br>0.008<br>0.026<br>0.016<br>0.010<br>6.395<br>**0.002**<br>**0.020**<br>**0.004**<br>**0.002**|



Table 5: Overall Evaluation: Perplexity (PPL) and TPR at FPR of 1% scores of the MIAs with different signals (Loss/Ref/Min-k/Zlib). For all metrics, the lower the value, the better the result. 

part dominates the learning one, it leads to high **D.4 Training time** perplexity values of language modeling. 

We report the training time for full fine-tuning Pythia 1.4B. We manually increase the batch size 

22887 

**==> picture [176 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
PPL and AUC while varying<br>40 0.7<br>35 PPL 0.6<br>PPL FT (AUC 0.86)<br>PPL DPSGD (AUC 0.51)<br>30 AUC 0.5<br>25<br>0.4<br>0.0 0.5 1.0 1.5 2.0<br>h [ (portion of tokens for learning)]<br>PPL AUC<br>**----- End of picture text -----**<br>


Figure 15: Performance of DuoLearn while varying _α_ 

that could fit into the GPU’s physical memory. As a results, FT and Goldfish can run with a batch size of 48, while DPSGD and DuoLearn can reach the batch size of 32. We also implement gradient accumulation, so all the methods can have the same virtual batch size. 

|ual batch size.||
|---|---|
|**Training Time**|**1 epoch** (in minutes)|
|FT<br>Goldfsh<br>DPSGD<br>DuoLearn|2.10<br>2.10<br>3.19<br>2.85|



Table 6: Training time for one epoch of (full) Pythia 1.4B on a single H100 GPU 

Table 6 presents the training time for one epoch. Goldfish has little to zero overhead compared to FT. DPSGD and DuoLearn have a slightly higher training time due to the additional computation of the privacy mechanism. In particular, DPSGD has the highest overhead due to the clipping and noise addition mechanisms. Meanwhile, DuoLearn requires an additional forward pass on the reference model to select the learning and unlearning tokens. DuoLearn is also feasible to work at scale that has been demonstrated in the pretraining settings of the previous work (Lin et al., 2024). 

22888 

