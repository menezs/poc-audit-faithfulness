# CULTURAL FIDELITY IN LARGE-LANGUAGE MODELS: AN EVALUATION OF ONLINE LANGUAGE RESOURCES AS A DRIVER OF MODEL PERFORMANCE IN VALUE REPRESENTATION 

**Sharif Kazemi** _[∗]_ 

**Gloria Gerhardt Jonty Katz Caroline Ida Kuria Estelle Pan** 

## **Umang Prabhakar** 

Columbia University School of International and Public Affairs The Fu Foundation School of Engineering and Applied Science 

## **ABSTRACT** 

The training data for LLMs embeds societal values, increasing their familiarity with the language’s culture. Our analysis found that 44% of the variance in the ability of GPT-4o to reflect the societal values of a country, as measured by the World Values Survey, correlates with the availability of digital resources in that language. Notably, the error rate was more than five times higher for the languages of the lowest resource compared to the languages of the highest resource. For GPT-4-turbo, this correlation rose to 72%, suggesting efforts to improve the familiarity with the non-English language beyond the web-scraped data. Our study developed one of the largest and most robust datasets in this topic area with 21 country-language pairs, each of which contain 94 survey questions verified by native speakers. Our results highlight the link between LLM performance and digital data availability in target languages. Weaker performance in low-resource languages, especially prominent in the Global South, may worsen digital divides. We discuss strategies proposed to address this, including developing multilingual LLMs from the ground up and enhancing fine-tuning on diverse linguistic datasets, as seen in African language initiatives. 

_**K**_ **eywords** Low-resource languages _·_ Large-language models _·_ Value representation 

## **1 Introduction** 

In the digital age, human communication, and thus language, is increasingly filtered through technology and the internet. These digital means are not neutral mediums, but rather influence the language and our relationship with it in a variety of ways (Lee and Ta, 2023). Most-affected by this trend are so-called low-resource languages (LRLs) which despite being the vast majority of the world’s 7,000 languages, hold a reduced digital presence and thus are increasingly facing erosion (Lee and Ta, 2023). 

In the literature, what has often most defined LRLs is that they are not one of the ‘20 languages’ that have been focused on in natural language processing methods, such as English (which holds half of the world’s websites), Mandarin, and French (Magueresse, Carles, Heetderks, 2020; Lee and Ta, 2023). This language categorization reflects the simplistic division of most countries into the Global South, much like how the Global South is often defined by what it is not: the Global North. In both cases, these categories are characterized as the ’other,’ making them less clearly defined. Similar to Joshi et. al.(2020), we utilize a methodology for understanding the degree of a language’s resources in table 2 using the proportion of online websites in that language as the key metric. 

> _∗_ Correspondence to: ms6578@columbia.edu 

Cultural Fidelity in Large-Language Models 

Low representation in digital text means lower capacity for utilization as training datasets for Large-Language Models (LLMs) and chatbots, resulting in lower quality Artificial Intelligence (AI) models, even if a system can be trained on this language at all (Magueresse, Carles, Heetderks, 2020). The scarcity of digital text means that these languages cannot be easily ported to AI models, therefore, dominant languages compound in strength through increased use in the digital realm while more minor languages face greater corrosive pressures (Lee and Ta, 2023). Our results in Section 4.1 demonstrate that a substantial proportion of an LLM’s ability to mimic societal values can be correlated to the availability of digital text in that language. This has wide-ranging implications. 

With government services increasingly reliant on chatbots and other human-computer interfaces, even basic functions necessary to maintaining citizen engagement will be AI-dependent, and thus LRL communities would have to shift towards dominant languages to maintain these interactions or else be limited in their digital engagement (Jungherr, 2023). 

Mirroring colonization, these foreign-language AI models are forcing assimilation toward dominant languages that are often the language of the colonizer such as English, French, or Spanish (Lee and Ta, 2023). In societies that strongly identify with their local languages as part of their national identity, such as Paraguay with Guarani, this creates cultural unease over the loss of heritage (Al Qutaini, et. al., 2024). This further-intensifies the inequality between and within countries, as more dominant cultures and languages do not suffer the same degradation. A detailed discussion of these potential harms is discussed in Section 5. 

## **2 Literature review** 

## **2.1 Performance of GPT in non-English languages** 

## **2.1.1 GPT-4o** 

GPT-4o, trained on data up until October 2023, has been documented to show significant advancements in handling non-English languages compared to previous models, according to OpenAI’s official documentation. While GPT-4o matches the performance of GPT-4 Turbo on English text and code, it notably exhibits marked improvements in processing non-English languages. These enhancements are largely due to the introduction of a new tokenizer that optimizes the model’s efficiency in handling non-English text. For instance, the tokenizer reduces the number of tokens required for Hindi by 2.9 times—from 90 to 31—and achieves similar efficiency gains across other languages, such as German (1.2x fewer tokens), English (1.1x fewer tokens), and Vietnamese (1.5x fewer tokens). These reductions not only streamline processing but also reduce the cost and time associated with non-English language tasks, which makes GPT-4o 50% cheaper to use via the API. 

Our findings in Section 4.5 support the claims of OpenAI for improved performance in non-English languages, as measured by value representation accuracy. However, we do note an interesting _increase_ in error-rate for high-resource languages. This is particularly true for English (USA), which can be partially explained by the previously high reliance of 4-turbo on data from this language as described in the following section. 

## **2.1.2 GPT-4-turbo** 

The GPT-4 model was reported to surpass previous language models in multilingual capabilities, according to OpenAI’s official technical report (2023). To evaluate GPT-4’s performance in different languages, the OpenAI team utilized the Multi-task Language Understanding benchmark—a collection of multiple-choice questions covering 57 subjects—translated into many different languages using Azure Translate. The report states, "GPT-4 outperforms the English-language performance of GPT-3.5 and existing language models (Chinchilla and PaLM) for the majority of languages we tested, including low-resource languages such as Latvian, Welsh, and Swahili." In these tests, English achieved an 85.5% accuracy score on the MMLU, while Swahili reached 78.5%, German 83.7%, Latvian 80.9%, and Welsh 77.5% (OpenAI, 2023, p. 8). 

However, the report also notes that the majority of the training data for GPT-4 was sourced from English, and the fine-tuning and evaluation processes were primarily focused on English, with a US-centric bias. The report notes, "Mitigations and measurements were mostly designed, built, and tested primarily in English and with a US-centric point of view. The majority of pretraining data and our alignment data is in English" (OpenAI, 2023, p. 21). This aligns with our findings in Section 4.5 which highlight a much greater correlation between websites in a language and model performance in mimicking values for 4-turbo (72%) than for 4o (44%). 

2 

Cultural Fidelity in Large-Language Models 

## **2.2 LLMs and societal value representation** 

A paper published by Anthropic researchers (Durmus et. al., 2024) comes closest to our approach wherein they investigate LLM responses to global opinion questions, derived from the Pew Global Attitudes Survey (PEW) and the World Values Survey (WVS), and compare to which extent the LLM’s answers align with the opinions of certain populations. The paper finds that LLMs responses tend to be more similar to the opinions from the USA, and some European and South American countries. When the model is prompted to consider a particular country’s perspective, responses shift to be more similar to the opinions of the prompted populations, but can reflect harmful cultural stereotypes. 

While this study demonstrates the existence of value bias/inaccuracy in LLMs when addressing value-laden questions, it does not delve deeply into the underlying causes, particularly the role of a language’s resource. Although our conclusions are similar, our contribution lies in exploring likely origins of this value bias/inaccuracy and its correlation with the availability of digital text in a given language. Moreover, the paper’s reliance on a language model to translate the questions can lead to differences between the translated and original questions (as noted by the authors). This gap between the questions poised to the LLM and those originally given to human respondents might necessitate differing responses which would artificially inflate the inaccuracy score. 

Other authors have also investigated language models’ representation of societal values. Arora, Kaffee, and Augenstein (2023) found that pre-trained language models do capture differences in values across cultures, but that these differences are not sufficiently defined to reflect cultures’ values as reflected in established surveys. Kharchenko, Roosta, Chadha, and Shah (2024) similarly applied a quantified methodology for representing a country’s values based on the 5 Hofstede Cultural Dimensions. Vimalendiran (2024) applies the Inglehart-Welzel Cultural Map, an academic measure of cultural value grouping, to derive most models to be closely aligned with value-sets commonly found in English-speaking and Protestant European countries. Santurkar et. al. (2023) utilized a similar approach as this paper with a focus on US public opinion and surveys to demonstrate the left-leaning tendencies of some human feedback-tuned language models. Low-resource language environments are also made more complex by the practice of code-switching, intermixing with languages that are typically higher-resource (such as English) within the same piece of text. Ochieng et. al. (2024) found that LLMs on average struggled with cultural nuances to mimic a clearer understanding of these mixed-language contexts. 

## **3 Methodology** 

## **Overview of methodology:** 

1. Country-language pairs are selected from the World Values Survey, for which a range of questions are transcribed and verified by native speakers. An average of a country-language pair’s answers for each language are collected. These serve as **original results** . 

2. The same questions from the World Values Survey are put to GPT-4, specifying the country and displaying the question in the language of interest. These serve as **generated results** . 

3. The difference between the original and generated results are measured. If the absolute difference is greater than, or equal to, 50% of the original value, then that question is counted as an error. The percentage of questions within a country-language pair that cross the 50% threshold sets that pair’s overall error-rate. 

4. The resource availability of a language is defined through the proportion of online websites. This acts as a proxy for our main explanatory variable. 

Steps 1 to 3 are summarised in the model architecture diagram found here 1. 

## **3.1 World Values Survey: selecting questions and languages** 

The World Values Survey (WVS) is a global research initiative that investigates people’s values and beliefs, tracking how they evolve over time across various domains. Since its inception in 1981, the WVS has been conducted in nearly 100 countries through national surveys. The questionnaires are largely uniform across countries, with only minor modifications (e.g., certain questions may be omitted in some countries). These national surveys are administered in the official language(s) of each country. For our paper, we utilized data from the latest WVS, Wave 7, which was conducted between 2018 and 2023 in 66 countries. Each country is surveyed once per wave, using random probability samples that are representative of the adult population. 

We chose to use WVS data for our study because it is a major survey that collects data on political and ethical values on a global scale, with the same set of comparable questions asked across different countries. 

3 

Cultural Fidelity in Large-Language Models 

The wave 7 questionnaire consists of a total of 259 core questions out of which we selected 94 questions across 11 categories (These questions are listed in Appendix D): 

- Societal values, attitudes, and stereotypes 

- Social capital, trust, and organizational membership 

- Economic values 

- Corruption 

- Migration 

- Security 

- Science & Technology 

- Religious Values 

- Ethical values and norms 

- Political Interest & Political Participation 

- Political Culture & Political Regimes 

## **3.1.1 Question selection** 

When selecting questions for our analysis, we focused on those that could be reasonably posed to a large language model (LLM) to simulate the values and opinions of a citizen of a specific country. For instance, we excluded questions related to respondents’ happiness, well-being, trust in neighbors, or factual knowledge (e.g., identifying the members of the UN Security Council), as these vary by personal experience and are less-related to the values that a country’s citizens are likely to share. Instead, we included questions that focus on political, social, or ethical values. These questions ask respondents to express their level of agreement or disagreement with specific statements, using scales ranging from 0-2 to 1-10. 

## **3.1.2 Language selection** 

When selecting languages for our analysis, we aimed to include a diverse range of geographical regions and societies, ensuring representation of different cultures, religions, and values. Additionally, we focused on languages that were used in a significant number of responses. In some countries surveyed by the WVS, the population was polled in multiple languages, leading to a lower number of responses in certain languages. To maintain robustness in our analysis, we included only those languages for which data was collected from at least 175 respondents. 

Furthermore, we particularly targeted languages that are predominantly spoken in one country. This was borne from the understanding that a language shared by many countries, such as Spanish, might have highly variant values encoded in the training dataset given the multiple sources and thus influence the LLM’s response. To better-isolate the relationship between the digital availability of a language and the accuracy of the LLM, languages that are largely unique to a country surveyed by a WVS were chosen (e.g., Indonesian, German, Burmese, Urdu, Filipino, Amharic, Hausa, Shona). The methodology for classifying languages as low-resource can be found in Section 3.4. 

## **3.1.3 Limitations and discrepancies** 

After selecting the questions and languages, we asked native speakers of each language to compare the English WVS questionnaire with the version used in their native language. This extra step was taken to ensure that when we later introduced questions to our LLM in English versus other languages, the questions would be as consistent as possible. 

We did not expect the non-English WVS questionnaires to be literal translations of the English version, as direct translations often fail to capture important contextual nuances. However, we asked our interpreters to identify any significant discrepancies between the English and non-English versions—such as questions being framed in a more leading way or the use of stronger language in other languages. The interpreters highlighted these differences, however, we did not alter the actual questionnaires, as the survey data in each country was based on those original questions. Nevertheless, it is worth highlighting some findings of our interpreters: 

4 

Cultural Fidelity in Large-Language Models 

|**Language**|**Comment from translator**|
|---|---|
|Shona|Some questions could be understood but are not quite right, unfamiliar<br>wordingwas used.|
|Hausa|“ta’addanci” is terrorism rather than simplycrime.|
|Swahili|"Imani" does not fullyimplyreligion.|
|Russian|"domashnyaya khozyayka" means something like owner who is at home<br>rather than housewife.|
|Vietnamese|Various translation inaccuracies spotted. For example, the English ques-<br>tion reads ’When a mother works for pay, the children suffer,’ but the<br>Vietnamese version says ’when a mother spends too much time working,<br>the children suffer.’ The translation also implies that men can do business<br>better than women.|
|Mandarin|Several important questions are missing, such as those concerning views<br>on elections, neighborhood security, and civil war. Additionally, some<br>translations do not align well with the English version, with certain<br>questions missing words compared to the English counterpart. For<br>example, in Q38, "long-term care" is omitted, and in Q189, "for a man"<br>is not included.|



Table 1: Translation notes from various languages 

In general, we included any question that may have a discrepancy between that language’s version and English, whilst noting the inconsistency in our dataset. The questions were consistently ordered across country surveys, except for in Japan which required a reordering by the translator to align with other questions. 

Through this verification process, decisions were made to reduce the dataset to a more comparable and robust set: 

- 5 questions that referenced the specific direction of writing of the language’s script were excluded as this might confuse the language model. 

- The survey in the People’s Republic of China contained 20 discrepancies, of which two were linguistic differences and 18 were missing. As we eventually excluded Mandarin Chinese from our results, we were able to include the results from these questions into the analysis. Earlier findings on this country-language pair and a discussion of reasons driving an inordinately high inaccuracy rate can be found in A. 

- 17 questions were excluded as they were not included in at least one country’s results - these included results from Iran, Zimbabwe, Tajikistan, and Nigeria. 

As the core of our analysis was to gauge the difference of inaccuracy between original and generated results for specific countries, differences in survey questions between countries was not a sufficient reason for excluding results. However, these discrepancies in question wording were still noted in our dataset to enable additional analyses later on when comparing results between countries. Excluding such questions would have significantly reduced the number of usable questions. Additionally, the interpretation of "significant differences" in translation can vary among interpreters, leading to potential inconsistencies in which questions might be excluded based on subjective judgments. However, it is important to acknowledge this limitation in our analysis, as the LLM’s responses may be influenced by how a particular question is phrased in different languages. 

## **3.2 Generation of Language Model Value Representation** 

Original results from the World Values Survey were grouped by the language of the interview and averaged according to the weighted responses given to each question. The responses in these surveys ranged in scales, from 0-2 to 1-10, and directionality, lowest number being most agree in some cases or strongly disagree in other cases. To offer comparable and numerical generated results, the LLM was prompted to respond to each question according to the original scale and directionality. 

Drawing from the results in Durmus et. al. (2023) and Kharchenko, Roosta, Chadha, and Shah (2024) which highlight that country-specific persona prompting results in steerable model values that more-closely reflect the country’s values, the prompt was engineered to refer to ask the LLM to respond as if a citizen of that country. 

5 

Cultural Fidelity in Large-Language Models 

## **3.3 Accuracy of Value Representation: Comparison between Original and LLM Generated Values** 

If the generated result from the LLM deviated by more than 50% from the original answer, then the result was counted as an error. An original result of 4/10 and a generated result of 6/10 or 2/10 would result in an absolute difference of 2/10, which is 50% of the original value. This would result in that question being counted as an error. The percentage of questions within a country-language pairs that counted as an error would set that pair’s overall error-rate, i.e., the degree to which the LLM performed poorly in mimicking its societal values. This threshold for error is set arbitrarily, but other values are graphed in Section 4.2. Limitations to this approach, including confining the LLM to closed-answer questions, are discussed in Appendix C . 

## Model architecture: **Gauging the Accuracy of Societal Value Representation in Large Language Models** 

**==> picture [365 x 432] intentionally omitted <==**

**----- Start of picture text -----**<br>
World Values Survey E.g., survey of Kenyans with interviews<br>{question} from conducted in Swahili Text of {question} E.g., {question}:<br>Start {country-language}survey pairing is is extracted society to have children.""It is a duty towards<br>selected<br>E.g.,<br>{question}:<br>Weighted numericalanswers to each meaning of the{scale} and {question} textverified in "Ni wajibukwa jamii<br>{question} are extracted extreme values language by kupata<br>according to the specific are extracted native speakers watoto."<br>{country-language} pair<br>Average answer was 8<br>Large Language Model<br>E.g., {scale}: 1-10,  Prompt formation:<br>10 being most agree. {question} asked in<br>specific language as if<br>citizen of relevant<br>{country}, according to<br>{scale}<br>Respond numerically to {question} with answers<br>based on {scale}. Respond as if you are answering<br>the World Values Survey for a citizen of {country},<br>expressing their views.<br>Generated answer<br>Original answer  collated in {scale}<br>collated in {scale} Original (WVS)<br>(average of 3 runs)<br>and generated (LLM)<br>answers are compared<br>(Original WVS) 8/10 4/10 (Generated LLM)<br>Inaccuracy percentage:<br>((8 - 4) / 8) = 50%<br>Inaccuracy<br>Not an error No greater or  Yes  Error<br>equal to<br>50%?<br>Collated results<br>Execute loop 94x21 times to cover by {country-<br>all curated questions for each of the language} and<br>21 {country-language} pairs   {question} topic<br>1. INPUTS<br>2. GENERATION<br>COMPARISON<br>3. COLLATION AND<br>**----- End of picture text -----**<br>


Figure 1: Model architecture diagram: steps 1 to 3 of methodology 

6 

Cultural Fidelity in Large-Language Models 

## **3.4 Low-Resource Languages** 

We included a balanced mix of low- and high-resource languages in our sample. To classify languages as low- or high-resource, we employed the following methodology: 

Taking the percentage of online content available in a given language, those accounting for less than 0.1% of online content (measured through Web Technology Surveys, 2024) are classified as low-resource. There is no universally defined cutoff for categorizing a language as low-resource. However, according to Web Technology Surveys (2024), only 38 of the more than 7,000 languages exceed the 0.1% threshold for web content representation. 

To understand the severity of impact in the real world of the language’s resources, we introduced a second methodology. This approach considers the ratio of the percentage of the world’s population that speaks a language to the percentage of online content available in that language. To calculate this ratio, we first determined the proportion of the global population that speaks a given language as their native language (L1). We then divided this figure by the percentage of websites available in that language. We also decided on a 0.1 cutoff on this ratio to classify a language as high vs low resource. One limitation of this methodology is that it does not account for languages with a significant number of second-language (L2) speakers, such as English, Swahili, or Indonesian. 

In total we selected 21 languages for our analysis which draw on a total of 29,590 respondents in respective surveys for the WVS. Out of these 21, 8 can be considered more low-resource and 13 considered more high-resource. These languages are detailed in table 2. Mandarin Chinese was excluded for reasons outlined in Appendix A, but is included in this table for illustrative purposes. 

|**% of**<br>**websites**|**Language**|**Native speakers**<br>**(L1)**|**% of world**<br>**population**|**Proportion**<br>**ratio**|**Resource**<br>**category***|
|---|---|---|---|---|---|
|49.7|English|380,000,000|4.69|10.59|High|
|5.9|Spanish|500,000,000|6.17|0.96|High|
|5.4|German|95,000,000|1.17|4.60|High|
|4.5|Japanese|120,000,000|1.47|3.04|High|
|4.1|Russian|150,000,000|1.85|2.21|High|
|3.7|Portuguese|260,000,000|3.21|1.15|High|
|1.8|Turkish|84,000,000|1.04|1.74|High|
|1.3|Farsi|72,000,000|0.89|1.46|High|
|1.2|Mandarin|941,000,000|11.62|0.10|High|
|1.2|Indonesian|43,000,000|0.53|2.26|High|
|1.1|Vietnamese|85,000,000|1.05|1.05|High|
|0.8|Korean|81,000,000|1.00|0.8|High|
|0.5|Greek|13,500,000|0.17|3.00|High|
|0.2|Serbian|12,000,000|0.15|1.35|High|
|0.04|Hindi|350,000,000|4.32|0.01|Low|
|0.0025|Burmese|33,000,000|0.41|0.006|Low|
|0.0015|Swahili|5,300,000|0.07|0.02|Low|
|0.0013|Filipino|29,000,000|0.36|0.004|Low|
|0.0010|Tajik|10,000,000|0.12|0.008|Low|
|0.0006|Amharic|35,000,000|0.43|0.001|Low|
|0.0001|Hausa|54,000,000|0.67|0.0002|Low|
|0.0001|Shona|6,500,000|0.08|0.001|Low|



Table 2: Language speakers and their web presence 

*It is important to acknowledge that there is no universally accepted definition in research to distinguish between lowand high-resource languages. Joshi et. al. (2020) introduce six classes for distinguishing between languages, using a mixture of labelled and un-labelled digital text availability to describe similar characteristics of languages within each class. However, under both methodologies applied in our paper, languages end up falling into the same category of low vs. high resource. Rather than treating the classification as a strict binary, it should be viewed as a spectrum. As later demonstrated in our results, the more resources a language has under either methodology, the less inaccuracy (error) in mimicking those values is observed. 

Kaplan et al. (2020) famously noted the power-law relationship between model performance and factors such as the size of the training dataset, model parameters, and compute used for training. We have chosen to use log base 10 of the number of online websites as a proxy for the non-linear impact that the availability of digital content (the number of tokens in a given language) has on a model’s ability to approximate an understanding of the society associated with that language. 

7 

Cultural Fidelity in Large-Language Models 

## **4 Findings** 

## **4.1 Language resource and error-rate in representing societal values** 

**Our main finding is that approximately 44% of GPT-4o’s ability to mimic an understanding of a society’s values is correlated to the language’s online presence.** While the source of training datasets for this model are unknown, these findings align with the knowledge that Common Crawl and publicly available data were a large source for the model’s training (OpenAI, 2024). The outlier performance of higher-resource languages in English, German, and Japanese could be a sign of additional fine-tuning on language-specific datasets for these societies. 

Swahili and Hindi demonstrate interesting outliers for lower-resource languages, and it can be hypothesized that this is due to the prevalence of English as a language in Kenya and India, respectively. 

**==> picture [469 x 254] intentionally omitted <==**

Figure 2: Societal representation error-rate vs. log number of websites, all country-language Pairs (GPT-4o) 

8 

Cultural Fidelity in Large-Language Models 

**==> picture [469 x 227] intentionally omitted <==**

Figure 3: Societal representation error-rate bands by country, mapped. 0-15% = low error-rate, 15-30% = medium error-rate, 30-45% = high error-rate (GPT-4o) 

This relationship is weaker if we exclude country-language pairs where the language is spoken to a significant degree in other countries. We can demonstrate this phenomenon by removing Portuguese, Spanish, and English to find that the correlation has decreased. This suggests some ’spillover’ benefits that these pairs enjoyed due to having datasets larger than those confined to their countries. 

**==> picture [469 x 254] intentionally omitted <==**

Figure 4: Societal representation error-rate vs. log number of websites, languages largely unique to that country (GPT-4o) 

9 

Cultural Fidelity in Large-Language Models 

## **4.2 Proportion thresholds** 

The threshold for a question counting as an error was set in this paper at 50%. In other terms, an error would be counted if the LLM-generated answer was more than 50% different than the original average answer for the country-language pair in the World Values Survey. 

Given 50% is an arbitrary cut-off, the correlative strength of a language’s resources and the error-rate are provided at different thresholds below. These demonstrate that the relationship becomes stronger until around 60% and then declines. Given the range of posssible answers are finitely defined within small scales (e.g., 0-2), there is an upper bound for how ’wrong’ an LLM can represent values, and thus this decline in strength for the upper thresholds is intuitive. 

Furthermore, it is notable that the lower thresholds (e.g., 10%) demonstrate that nearly all languages are exhibiting high error-rates, but that the relationship between error-rate and a language’s resource is weaker at these lower thresholds. **Put simply, LLMs make errors in representing societal values across all country-language pairs, but they demonstrate more significant errors if the language is lower-resource.** 

**==> picture [468 x 257] intentionally omitted <==**

Figure 5: Error-rate of languages across different proportion thresholds for the error (GPT-4o) 

10 

Cultural Fidelity in Large-Language Models 

## **4.3 Topic areas** 

Most topic areas exhibited similar error-rates across lower- and higher-resource languages. However, three categories demonstrated differentiated error-rates with higher inaccuracies for lower-resource languages: 

- Security 

- Ethical values and norms 

- Political culture and political regimes 

**==> picture [469 x 313] intentionally omitted <==**

Figure 6: Error-rate by topic area and language resource category (GPT-4o) 

## **4.4 Controversy and error-rate** 

Controversy, measured as the standard deviation of responses from the original WVS responses in that language, was calculated at the question-level for each country-language pair. It was hypothesized that more controversial topics may experience lower error-rates driven by these topics/questions being more often discussed in online forums which would increase the frequency of examples in the relevant training dataset for the LLM; Salminen et. al., (2020) discuss this phenomenon in relation to ‘online toxicity’ and the topic being covered in media. 

This did not turn out to be the case, however, and only negligible correlation was found between the controversy score and error-rate, whether this was measured at country-language or topic-level for each country-language pair. The high controversy score for English responses is notable and highlights the diversity of values within this very high-resource language. 

11 

Cultural Fidelity in Large-Language Models 

**==> picture [469 x 253] intentionally omitted <==**

Figure 7: Error rate and controversy score of country-language pairs (GPT-4o) 

**==> picture [469 x 257] intentionally omitted <==**

Figure 8: Error rate and controversy score of topic areas averaged for each country-language pair (GPT-4o) 

12 

Cultural Fidelity in Large-Language Models 

## **4.5 Differences across models** 

The relationship between online resources and accuracy in value representation was even stronger in GPT-4-turbo, implying a greater reliance on webscraping methods. 72% of the variance at language-level was captured by the log of number of websites, compared to 44% for GPT-4o. This supports the claim made in 2 that new datasets and methods were used to train GPT-4o and derive stronger performance in non-English languages. 

**==> picture [469 x 255] intentionally omitted <==**

Figure 9: Societal representation error-rate vs. log number of websites, all country-language pairs (GPT-4-turbo) 

However, while improvements are marked for low-resource languages, the picture is more complex when considering high-resource languages and English (USA) in particular. 

**==> picture [234 x 174] intentionally omitted <==**

Figure 10: Error-rate between 4-turbo and 4o, by resource language category 

13 

Cultural Fidelity in Large-Language Models 

**==> picture [469 x 345] intentionally omitted <==**

Figure 11: Error-rate between 4-turbo and 4o, by country-language pair 

This suggests a differentiated change in model performance at value representation for low- and high-resource languages when OpenAI introduced GPT-4o. Through a linear regression utilizing an interaction term for low-resource-languages and model of choice associated with the change in error-rate. The results show that low-resource-languages saw, on average, a 10 percentage point decrease in their error-rate when switching from GPT-4-turbo to GPT-4o, when compared with high-resource languages. 

Error-rate = _β_ 0 + _β_ 1 _·_ High-resource language + _β_ 2 _·_ 4-turbo + _β_ 3 _·_ (High-resource language _×_ 4-turbo) + _ϵ_ 

14 

Cultural Fidelity in Large-Language Models 

**==> picture [469 x 350] intentionally omitted <==**

Figure 12: Results from regression function. High-resource languages saw a 10 percentage point increase in error-rate, as compared to low-resource languages, when moving from 4-turbo to 4o. 

## **5 Implications** 

Our findings on the intensification of LLM inaccuracy in low-resource language settings may have significant broader societal implications. Generative AI (Gen-AI) tools based on LLMs are becoming an important tool of public and private service delivery, with use cases in industries ranging from news and media, public services, to medicine. More and more, we see daily useful knowledge consumed by proficient internet users being intermediated by LLM applications. Gen-AI tools are beginning to play the role of a content creator, advisor, and decision maker (Bhardwaj et al. 2023]. The risk is not just that culturally biased/inaccurate LLMs perpetuate flawed information, but that they make or lead users to make decisions that are suboptimal or even harmful to society. This is particularly concerning as Gen-AI becomes deeply embedded across the digital ecosystem, and where these values become embedded in a long chain of actions where the origin and impact of bias becomes near impossible to uncover by users. A list of potential use-cases affected by this finding is detailed below, with our approach outlined in B.1. 

## **5.1 Implications for LLM performance in common use-cases** 

The full extent of the harmful effects of culturally biased LLMs cannot fully be articulated ex-ante, and will range from positive to benign to harmful. For those monitoring the subject, we see the following framework emerging, guided by three key questions: What are the major large scale use cases of LLM based generative AI? In which of these settings does the cultural alignment of the AI tool have a significant impact on its output and decision? In which of these cases does the output have a meaningful and measurable impact on social good? 

15 

Cultural Fidelity in Large-Language Models 

Using this guide, a few key use cases (Rauser, 2023) of LLM applications that have high social risk emerging from cultural misalignment of AI emerge. We highlight here illustrative examples of harmful effects resulting from these applications below: 

- Allocation of resources 

   - Recruiting: Automated screening processes that look for keywords associated with characteristics that are underrepresented or not as valued in low-resource settings, could lead to prejudice against talented candidates from these settings (Li et al. 2023). 

   - Medicine: Treatment decisions in low-resource settings that are based on participant data and behaviours from high-resource settings, leading to prescription of incorrect or ineffective treatment options for users from low-resource settings. Similar effects have been identified in AI algorithms used for patient classification leading to significant bias against black patients (Z. Obermeyer et al. 2019). 

- Teaching and learning 

   - Queries on values: proliferation of values and opinions at odds with prevailing attitudes of a country, leading to social and cultural conflicts and discontentment, and potential global cultural homogenization (Cao, Lee et al. 2023). 

   - History: sharing historical information that is viewed with a western lens, which can either erase or reshape the understanding of events in low-resource settings that do not have a high level of digital representation of its history (Al Qutaini, et. al., 2024). 

- Generation of content and products 

   - News: Parsing and repackaging of news content in AI-generated news bites or editorials that portray a person, event, or policy in a prejudiced manner. 

   - Marketing and sales: Misleading or offensive messaging towards a certain customer segment that is poorly understood by the LLM e.g., people of a certain race or religious denomination. 

- Censorship and moderation: flagging or censoring social media content that is deemed incorrectly offensive, or permitting content that is harmful to an at-risk identity group in low resource settings. A number of research studies have looked at these effects in practice, identifying how AI moderators can be far less efficacious in non-majority languages like the low-resource languages we study (Cari Beth Head et al. 2023). 

## **5.2 Intersections of inaccurate value representation with censored digital ecosystems** 

We can relate these questions to the topic areas in which we see the highest error rates in our analysis to hypothesize the kinds of risks that may arise in low-resource societies. Our analysis showed the highest error rates in topics under “security,” “ethical values and norms,” and “political culture and political regimes.” The settings where there was greatest deviation between LLM answers and that of the WVS were Nigeria, Zimbabwe, Ethiopia, and Tajikistan, amongst others. All four countries rank “not free” on Freedom House (Freedom House, 2023). There are two potential implications of this. First, users who rely on or use LLM answers in these settings can face the risk of persecution if the alternative values of the LLMs shape their view publicly, in contradiction to the prevailing views of their autocratic governments. Second, and more importantly, is whether views expressed online in these settings can ever be true proxies for the beliefs of the underlying society. Three out of four of these countries score poorly on internet freedom (scores of less than 50/100) (Freedom House, 2022; Reporters without Borders, 2023). This means that, unless online platforms were to become a free public commons unbounded by restrictive censorship, LLMs trained on digital content from these regions may always differ from the honest perspectives of its people that are better captured in 1-1 surveys like the WVS. This limits the ability of LLMs to mimic the values of societies where digital content is a poor proxy for true felt values. The constraint of using digital content as a proxy for societal values is not limited to countries with high censorship only. It extends to all countries where there is a significant divide between the characteristics of people generating the bulk of online content, and the true demography of the country, divides that can be caused by factors correlated with values such as age, class, and gender (Shcradie, 2018). 

## **5.3 Broader cultural normativity and repercussions of homogenization** 

As researchers and practitioners concerned with cultural bias embedded in LLMs scale up LLM use across geographical and cultural settings, they will be confronted with one key question: what values should LLMs be centered around, if not the prevailing values of high-resource societies currently seen in our analysis? LLMs are ultimately reflections of the humans that have built and trained them, as well as the largely human-generated data that has trained them. As all humans are biased, influenced by their unique set of histories and cultures, the issue in front of AI ethics researchers and policy makers is that of whose bias LLMs be most closely aligned to. 

16 

Cultural Fidelity in Large-Language Models 

Researchers like Oliviera et al. (2023), and Benkler et al. (2023), amongst others, have attempted to understand this question. They posit that embedding cultural pluralism, or the co-existence of multiple cultures within a broader system, is one way forward. A culturally pluralistic LLM would have the ability to adapt its output based on the prompt and user description provided to it, aligning itself with the values of the user and their setting more closely. However, cultural pluralism is not devoid of its own ethical challenges, such as if an LLM should indeed reproduce highly restrictive values that penalize one sub-identity or group in a setting (such as LGBTQ+ populations in Islamic autocracies). Should the creators of LLMs “be better” than the context in which they are applied? Who decides what is better? These are questions that deserve further discussion at an international scale, and are best placed to be answered by coalitions of government, civil society, and market actors. At the very least, a creation of a common minimum standard of values/ethics should be agreed upon, much like coalitions of international parties have done with the Geneva Convention on treatment of civilians, prisoners of war, and soldiers. The OECD’s Principles for trustworthy AI (OECD, 2019) are one such example of broad based principles adaptable to regional and cultural settings, and have been adopted by over 40 countries worldwide. More such standards emerging from low-resource settings are expected to emerge in the coming years. 

We discuss a broader set of potential solutions in more detail in the following section 6. 

## **6 Discussion** 

Our research findings on language representation in large language models (LLMs) provide insightful perspectives on the current state of AI development and its implications for global linguistic diversity. As the above analysis reveals, the number of online websites in a particular language underscores the significant influence of digital presence on AI performance based on around 40% of the variance in GPT-4’s ability to represent a country’s societal values. 

The correlation between a language’s online presence and the accuracy of LLMs in representing the societal values of its speakers highlights a digital linguistic divide that may exacerbate existing inequalities, particularly in the Global South. This disparity in representation has far-reaching implications, potentially affecting access to information, the quality of AI-driven services, and the overall digital inclusion of underrepresented language communities. 

The graphical representation of the relationship between a language’s online presence and the LLM’s error rate in mimicking societal values provides a clear visualization of this divide. Languages with minimal online footprints, often from the Global South, exhibit significantly higher error rates. This finding emphasizes the need for targeted interventions to increase the digital representation of these languages. 

However, it is crucial to note that this digital divide, while presenting challenges, also offers unique opportunities for innovation and targeted development of AI solutions that cater to underserved linguistic communities. 

## **6.1 Proposed Solutions** 

Addressing the language representation gap in AI requires a comprehensive, multifaceted approach: 

1. Democratizing AI Development: Open-source initiatives in AI offer promising avenues for more inclusive language representation. Collaborative efforts among diverse linguistic communities can lead to the development of models that better reflect global linguistic diversity. The Masakhane research community demonstrates how grassroots African NLP research can contribute to closing the digital divide (Litre et al., 2022). 

2. Ethical Monitoring and Regulation: As the AI landscape continues to be dominated by a few tech giants, robust regulatory frameworks are essential. These should mandate transparency in training data composition and set clear diversity benchmarks for language representation in commercial AI products. Adelani et al. (2022) highlight the importance of local knowledge and community involvement in developing NLP technology for understudied languages, reducing dependence on industry powerhouses from the Global North. 

3. Collaborative Data Sharing and Local Digital Spaces: Partnerships between AI developers and local digital initiatives can unlock valuable linguistic data currently siloed in community-specific platforms. This approach not only enriches training datasets but also empowers local communities in shaping AI development. The development of AFRIMTE, a human evaluation dataset for 13 typologically diverse African languages, exemplifies this approach (Adelani et al., 2023). By involving local communities and experts in the creation of AFRIMTE, Adelani et al. (2023) not only produced a valuable resource for machine translation evaluation but also empowered these communities to play a direct role in the development of language technologies for their languages. This collaborative approach serves as a model for future efforts to improve language representation in AI for under-resourced languages. 

17 

Cultural Fidelity in Large-Language Models 

4. Targeted Digital Inclusion Initiatives: Addressing the root cause of underrepresentation requires concerted efforts to increase internet access and digital literacy in underserved regions. Ogayo et al. (2022) describe their “AfricanVoices” initiative, which aims to develop speech synthesis systems for low-resourced African languages. The researchers created new datasets and curated existing recordings for 12 African languages, including Luo, Suba, and Kenyan English. They demonstrated that it’s possible to develop synthesizers that generate intelligible speech with as little as 25 minutes of created speech, even when recorded in suboptimal environments. Ogayo et al. (2022) demonstrate that even with limited resources, it’s possible to create speech synthesis systems for African languages, which can improve digital accessibility. 

5. Multilingual LLMs from the Ground Up: Developing inherently multilingual LLMs, designed with linguistic diversity as a core principle rather than an afterthought, could revolutionize language representation in AI. This approach requires close collaboration with linguistic experts and local communities from the outset. The development of AfriCOMET, which leverages an African-centric multilingual encoder (AfroXLM-R), demonstrates the potential of this approach (Adelani et al., 2023). 

6. Fine-tuning on Diverse Linguistic Datasets: Creating open-access repositories of high-quality, diverse linguistic datasets can democratize the fine-tuning process, allowing researchers worldwide to improve LLM performance for specific languages and dialects. The creation of UlizaLlama, the first open-access Swahili Large Language Model, is a step in this direction (Jacaranda Health, 2023). 

Future research should focus on implementing and rigorously evaluating these proposed solutions. Longitudinal studies tracking changes in LLM performance as the online presence of low-resource languages grows could provide invaluable insights into the dynamics of linguistic representation in AI systems. Such studies could leverage methodologies similar to those used by Adelani et al. (2023) in creating AFRIMTE, applying them to a broader range of languages over time. This approach would not only measure improvements in LLM performance but also assess the effectiveness of targeted digital inclusion initiatives like those proposed by Ogayo et al. (2022). However, researchers must be cautious about potential biases in online content growth, as it may not accurately represent the full linguistic diversity of offline communities (as highlighted by our exposure ratio in table 2). 

As suggested by our findings for topics related to security, ethical values and norms, and political culture and political regimes, there are significant ethical considerations in AI-driven information dissemination. Our research revealed higher error rates in these sensitive areas, particularly for low-resource languages. This discrepancy highlights the need for careful consideration when deploying AI systems in contexts where certain types of information may be legally or culturally restricted. This includes developing frameworks for responsible AI deployment that balance the potential benefits of information access with considerations of user safety and legal compliance. These frameworks should be developed collaboratively with local communities, following the participatory approach demonstrated by Adelani et al. (2023). However, this raises complex questions about who gets to define cultural appropriateness and how to handle conflicts between global ethical standards and local cultural norms. Researchers must also consider the potential for AI systems to inadvertently perpetuate or exacerbate existing social inequalities, particularly in regions with complex historical and cultural dynamics. 

It is crucial to note that our findings are correlative rather than causative. In this regard, the proposed solutions should be considered with this limitation in mind. Further research is needed to establish causal relationships between language resources and LLM performance in representing societal values. Additionally, future studies should aim to quantify the content gap that low-resource languages need to close for better LLM performance. This could involve estimating the number of websites or tokens required to achieve parity with high-resource languages in terms of LLM accuracy. 

## **6.2 Limitations of Web Scraping in LLM Training** 

The reliance on web-scraped data, particularly through platforms like Common Crawl, has been a cornerstone of LLM training. However, this approach has significant limitations that demand critical reevaluation. Baack (2024) highlights that Common Crawl’s data, despite its size, is not a representative sample of the entire internet, and its use in LLM training has shaped builders’ expectations regarding model behavior in potentially problematic ways. The filtering processes applied to this data have been criticized for failing to remove harmful content adequately and potentially hurting the representational diversity of the training data. For instance, Baack notes that popular filtering techniques like using AI classifiers based on Reddit upvotes are problematic due to the homogeneity of Reddit users and the potential inclusion of content from toxic communities. These issues are particularly acute for languages and cultures with limited online presence, effectively amplifying existing digital divides. 

This realization necessitates a fundamental shift in how we approach data collection for AI training, especially for underrepresented languages and cultures. This realization necessitates a critical reevaluation of data collection methods for AI training. Future research should explore alternative approaches that can capture the linguistic and cultural richness 

18 

Cultural Fidelity in Large-Language Models 

of communities with limited online presence. This may involve collaborative efforts with linguistic experts, cultural institutions, and local communities to develop more representative and diverse datasets. Importantly, these efforts should consider the use of audio sources, recognizing the historical significance of radio as a form of communication where there is more diversity in the use of language, especially for marginalized communities. The role of oral traditions, such as the griots in West African cultures who pass down history and cultural knowledge through voice, should be acknowledged and incorporated into data collection strategies. 

Moreover, the approach to gathering digital data should be tailored to the unique ways in which people in low-resource settings interact with digital spaces. Unlike in Western contexts, website comment sections and chatrooms may not be as widely used. Instead, media outlets often engage viewers through different channels, such as SMS or voice messages, to gather opinions and feedback. Polls and other forms of digital interaction in these settings may provide valuable linguistic data that is currently underutilized. Training data should therefore be sourced differently, adapting to these local digital behaviors while also building upon the innovative approaches already being employed by initiatives like Lelapa AI, Masakhane, and collaborations such as UlizaLlama (Jacaranda Health, 2023). This approach is exemplified in the work of Adelani et al. 2023. The authors recognized the need for tailored data collection methods by creating "the largest news topic classification dataset for 16 typologically diverse languages spoken in Africa" (Adelani et al., 2023). Their work demonstrates an understanding that standard NLP approaches may not always be directly applicable to low-resource languages, as evidenced by their finding that "prompting LLMs like ChatGPT perform poorly on the simple task of text classification for several under-resourced African languages especially for non-Latin based scripts" (Adelani et al., 2023). 

By diversifying data collection methods to include these often-overlooked sources, as shown in the MasakhaNEWS project, AI models can better represent the full spectrum of linguistic and cultural expression in low-resource communities. This approach aligns with the paper’s conclusion that "existing supervised approaches work well for all African languages and that language models with only a few supervised samples can reach competitive performance, both findings which demonstrate the applicability of existing NLP techniques for African languages" (Adelani et al., 2023), highlighting the potential of tailored approaches in low-resource settings. 

The limitations of web-scraped data for LLM training extend beyond issues of representation and bias. In their seminal paper, Bender et al. (2021) highlight the potential risks associated with large language models and propose critical recommendations. One of their key suggestions, particularly relevant to our discussion, is "investing resources into curating and carefully documenting datasets rather than ingesting everything on the web" (Bender et al., 2021). This recommendation aligns with our findings and underscores the need for a more thoughtful and deliberate approach to data collection and curation, especially for low-resource languages. Rather than relying solely on web scraping to amass large quantities of data, researchers and developers should focus on creating high-quality, well-documented datasets that accurately represent the linguistic and cultural nuances of diverse communities. 

Our research reveals a complex relationship between a language’s online presence and LLM accuracy in representing societal values. While we found that 44% of GPT-4o’s ability to mimic societal values correlates with a language’s online resources, this leaves a significant portion unexplained. Notably, our analysis of controversy scores showed high diversity in values among English-speaking respondents, despite English being a high-resource language. This finding, coupled with the unexpected increase in error rate for English from GPT-4-turbo to GPT-4o (as noted in Section 4.5), suggests that even abundant online content may not accurately represent the full spectrum of societal values. These results indicate that online content, regardless of volume, may skew towards certain perspectives, potentially leading to biased representations in LLMs. This issue is likely exacerbated in low-resource languages, where the limited available content may not capture the full diversity of views and values within the speaking community. 

Although our study focuses on the relationship between language resources and LLM performance, it is important to acknowledge potential confounders that were not controlled for, such as GDP per capita and cultural similarity to wealthy countries. These factors may significantly influence the observed relationships. Future research should aim to disentangle these effects to provide a more comprehensive understanding of LLM performance judged more clearly against language availability. 

## **7 Conclusion** 

Our paper highlights a new dimension to the relationship between training dataset size and language model performance (Kaplan et al. 2020). Utilizing a high-quality survey verified by 23 native speakers, 44% GPT-4o’s ability to mimic an understanding of a local culture was shown to be associated with the log of online websites in that language. However, the improved performance from 4-turbo to 4o for low-resource languages and the decreased correlation with online resources demonstrates that LLM reliance on existing datasets is not a static phenomenon. 

19 

Cultural Fidelity in Large-Language Models 

With the majority of attention and funding in LLM development focused on high-resource language and more economically developed settings, the potential implications compound negatively for the Global South which is host to the vast majority of low-resource language speakers. It is our hope that these findings further-drive the global effort towards more inclusive LLM design and development. 

20 

Cultural Fidelity in Large-Language Models 

## **8 Author Contributions** 

**Sharif Kazemi** designed the research approach, led the project team, performed most of the analysis (including the LLM loop), prepared data visualizations, and wrote significant portions of the paper. **Gloria Gerhardt** led the research by selecting questions from the WVS, defining low-resource languages, and coordinating with native translators to deliver the key dataset; furthermore writing significant portions of the Methodology and Literature Review sections. **Jonty Katz** helped scope the research approach, conducted analysis on the controversy score, and contributed to the overall analysis flow and Git repository. **Caroline Ida Kuria** and **Umang Prabhakar** analysed policy implications from our findings, researched proposed solutions for low-resource settings, and examined the likely drivers for inaccuracies across topics and country-language pairs. **Min (Estelle) Pan** conducted a literature review on LLM performance in non-English settings, in particular rationales for the inaccuracies spotted in Mandarin Chinese. 

## **Acknowledgments** 

The authors thank **Dr. Tamar Mitts** for guidance throughout the research process. The authors are also grateful to **Dr. Cristhian Parra** , as well as anonymous reviewers, for their feedback on the draft manuscript. 

The questions from the World Values Survey evaluated in this research were carefully transcribed and validated by 23 native speakers: 

|**Language (Country)**|**Native Translator**|
|---|---|
|German(Germany)|Gloria Gerhardt|
|Greek(Greece)|Maria Chrysanthou|
|Indonesian(Indonesia)|AmyDarajati Utomo|
|Russian(Russia)|Ainur Tokpayeva|
|Farsi(Iran)|Sharif Kazemi|
|Portuguese(Brazil)|Rafaella Lopes|
|Spanish(Peru)|Tomas Villar de Rohde|
|Mandarin Chinese(People’s Republic of China)|Estelle Pan|
|Traditional Chinese(Taiwan)|Estelle Pan|
|Swahili(Kenya)|Caroline Kuria|
|Hindi(India)|UmangPrabhakar|
|Burmese(Myanmar)|Rachel Set Aung|
|Hausa(Nigeria)|Zahra Suleiman|
|Amharic(Ethiopia)|Matthew Seifu|
|Shona(Zimbabwe)|Anonymous|
|Kazakh(Kazakhstan)|Ainur Tokpayeva and Akmaral Bekbossynova|
|Tajik(Tajikistan)|Jasur Okhunov|
|Filipino(Philippines)|Ana Sta Isabel|
|Japanese(Japan)|Airi Nakata|
|Vietnamese(Vietnam)|Nguyen PhuongNguyen|
|Serbian(Serbia)|Ana Krstanovic|
|Turkish(Turkey)|Anonymous|
|Korean(South Korea)|SeongHwan Jeon|



Table 3: Native Translators by Language and Country 

This paper was inspired by public participation efforts on AI conducted in Paraguay in collaboration with the UNDP Accelerator Lab. One common remark we heard was, "AI models don’t understand Paraguayan values because they don’t speak Guarani." We are grateful to the UNDP Paraguay country office and everyone we spoke with during that project for inspiring this work. 

21 

Cultural Fidelity in Large-Language Models 

## **References** 

- [Adelani and et al.(2022)] D. Adelani and et al. A few thousand translations go a long way! leveraging pre-trained models for african news translation. _Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics_ , 2022. URL `https://aclanthology.org/2022.naacl-main. 223.pdf` . 

- [Adelani and et al.(2023)] D. Adelani and et al. Masakhanews: News topic classification for african languages. In _IJCNLP-AACL_ , 2023. 

- [Arora et al.(2023)Arora, Kaffee, and Augenstein] A. Arora, L.-A. Kaffee, and I. Augenstein. Probing pre-trained language models for cross-cultural differences in values. In _Proceedings of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP)_ , pages 114–130. Association for Computational Linguistics, 2023. URL `https://arxiv.org/pdf/2203.13722` . 

- [Ayoub et al.(2024)Ayoub, Balakrishnan, Ayoub, Barrett, David, and Gray] N. F. Ayoub, K. Balakrishnan, M. S. Ayoub, T. F. Barrett, A. P. David, and S. T. Gray. Inherent bias in large language models: A random sampling analysis. _Mayo Clinic Proceedings_ , June 2024. URL `https://www.mcpdigitalhealth.org/action/showPdf?pii= S2949-7612%2824%2900020-8` . n.d. 

- [Baack(2024)] S. Baack. A critical analysis of the largest source for generative ai training data: Common crawl. In _The 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’24)_ , Rio de Janeiro, Brazil, 2024. ACM. doi: 10.1145/3630106.3659033. URL `https://facctconference.org/static/papers24/ facct24-148.pdf` . 

- [Bang et al.(2023)Bang, Cahyawijaya, Lee, Dai, Su, Wilie, Lovenia, Ji, Yu, Chung, Do, Xu, and Fung] Y. Bang, S. Cahyawijaya, N. Lee, W. Dai, D. Su, B. Wilie, H. Lovenia, Z. Ji, T. Yu, W. Chung, Q. V. Do, Y. Xu, and P. Fung. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. _arXiv_ , 2023. URL `https://arxiv.org/pdf/2302.04023` . 

- [Bender et al.(2021)Bender, Gebru, McMillan-Major, and Shmitchell] Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In _FAccT ’21_ , pages 610–623, 2021. doi: 10.1145/3442188.3445922. 

- [Benkler et al.(2023)Benkler, Mosaphir, Friedman, Smart, and Schmer-Galunder] N. Benkler, D. Mosaphir, S. E. Friedman, A. Smart, and S. M. Schmer-Galunder. Assessing llms for moral value pluralism. _Smart Information Flow Technologies_ , 2023. URL `https://arxiv.org/pdf/2312.10075v1` . n.d. 

- [Bernstein(2024)] S. Bernstein. The role of digital privacy in ensuring access to abortion and reproductive health care in post-dobbs america, June 03 2024. URL `https://www.americanbar. org/groups/crsj/publications/human_rights_magazine_home/technology-and-the-law/ the-role-of-digital-privacy-in-ensuring-access-to-reproductive-health-care/` . 

- [Bhardwaj et al.(2023)Bhardwaj, Bookey, Ibironke, Kelly, and Sevik] P. Bhardwaj, L. Bookey, J. Ibironke, N. Kelly, and I. S. Sevik. A meta-analysis of the economic, social, legal, and cultural impacts of widespread adoption of large language models such as chatgpt. _Computer Science_ , September 14 2023. URL `https://www.oxjournal. org/economic-social-legal-cultural-impacts-large-language-models/` . 

- [Borders()] Reporters Without Borders. World press freedom index 2023. `https://rsf.org/en/index` . Accessed September 10, 2024. 

- [Brown and Liu(2020)] D. Brown and A. Liu. Democracy and minority language recognition. _Political Science_ , 2020. doi: 10.1093/obo/9780199756223-0308. 

- [Cao et al.(2023)Cao, Zhou, Lee, Cabello, Chen, and Hershcovich] Y. Cao, L. Zhou, S. Lee, L. Cabello, M. Chen, and D. Hershcovich. Assessing cross-cultural alignment between chatgpt and human societies: An empirical study. _ArXiv_ , abs/2303.17466, 2023. URL `https://doi.org/10.48550/arXiv.2303.17466` . 

- [Dave(2023)] P. Dave. Chatgpt is cutting non-english languages out of the ai revolution, May 31 2023. URL `https://www.wired.com/story/chatgpt-non-english-languages-ai-revolution/` . 

- E. Durmus, K. Nguyen, T. I. Liao, N. Schiefer, A. Askell, A. Bakhtin, C. Chen, Z. Hatfield-Dodds, D. Hernandez, N. Joseph, L. Lovitt, S. McCandlish, O. Sikder, A. Tamkin, J. Thamkul, J. Kaplan, J. Clark, and D. Ganguli. Towards measuring the representation of subjective global opinions in language models. _arXiv_ , 2023. URL `https://arxiv.org/pdf/2306.16388` . 

- [Education(2024)] Sahar Education. Access to education empowers afghan girls and women, 2024. URL `https:// www.seattletimes.com/sponsored/access-to-education-empowers-afghan-girls-and-women/` . 

22 

Cultural Fidelity in Large-Language Models 

- [for Economic Co-operation and Development(2019)] Organisation for Economic Co-operation and Development. Oecd ai principles. Retrieved [Insert Date], 2019. URL `https://oecd.ai/en/ai-principles` . n.d. 

- [Haerpfer et al.(2022)Haerpfer, Inglehart, Moreno, Welzel, Kizilova, Diez-Medrano, Lagos, Norris, Ponarin, and Puranen] C. Haerpfer, R. Inglehart, A. Moreno, C. Welzel, K. Kizilova, J. Diez-Medrano, M. Lagos, P. Norris, E. Ponarin, and B. Puranen, editors. _World Values Survey: Round Seven – Country-Pooled Datafile Version 6.0_ . JD Systems Institute and WVSA Secretariat, Madrid, Spain and Vienna, Austria, 2022. doi: 10.14281/18241.24. 

- [Health(2023)] Jacaranda Health. Jacaranda launches first-in-kind swahili large language model, October 31 2023. URL `https://www.jacarandahealth.org/blog/ jacaranda-launches-first-in-kind-swahili-large-language-model` . 

- [Holten et al.(2021)Holten, de Goeij, and Kleiverda] L. Holten, E. de Goeij, and G. Kleiverda. Permeability of abortion care in the netherlands: a qualitative analysis of women’s experiences, health professional perspectives, and the internet resource of women on web. _Sexual and Reproductive Health Matters_ , 29:1917042, 2021. doi: 10.1080/26410397.2021.1917042. URL `https://www.tandfonline.com/doi/full/10.1080/26410397. 2021.1917042#d1e1040` . 

- [House(2023)] Freedom House. Freedom in the world 2023. `https://freedomhouse.org/countries/ freedom-world/scores` , 2023. Accessed September 10, 2024. 

- [Joshi et al.(2020)Joshi, Santy, Budhiraja, Bali, and Choudhury] Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. The state and fate of linguistic diversity and inclusion in the NLP world. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ , 2020. doi: 10.18653/ v1/2020.acl-main.560. 

- [Jungherr(2023)] A. Jungherr. Artificial intelligence and democracy: A conceptual framework. _Social Media + Society_ , 9, 2023. doi: 10.1177/20563051231186353. 

- [Kaplan et al.(2020)Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, and Amodei] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. URL `https://arxiv.org/abs/2001.08361` . 

- [Keshari et al.(2024)Keshari, Sign, Jain, and Chadha] Shuvam Keshari, Smriti Sign, Vinija Jain, and Aman Chadha. Born with a silver spoon? investigating socioeconomic bias in large language models, 2024. URL `https: //arxiv.org/abs/2403.14633` . 

- [Kharchenko et al.(2024)Kharchenko, Roosta, Chadha, and Shah] J. Kharchenko, T. Roosta, A. Chadha, and C. Shah. How well do llms represent values across cultures? empirical analysis of llm responses based on hofstede cultural dimensions. _arXiv_ , 2024. URL `https://arxiv.org/abs/2406.14805v1` . 

- [Kircher and Kutlu(2023)] Ruth Kircher and Ethan Kutlu. Multilingual realities, monolingual ideologies: Social media representations of spanish as a heritage language in the united states. _Mercator European Research Centre on Multilingualism and Language Learning_ , 2023. URL `https://academic.oup.com/applij/article-abstract/ 44/6/1077/6986832` . 

- [Kour and Saabne(2014)] George Kour and Raid Saabne. Real-time segmentation of on-line handwritten arabic script. In _Frontiers in Handwriting Recognition (ICFHR), 2014 14th International Conference on_ , pages 417–422. IEEE, 2014. 

- [Lee and Ta(2023)] N. T. Lee and R. Ta. How language gaps constrain generative ai development, October 2023. URL `https://www.brookings.edu/articles/ how-language-gaps-constrain-generative-ai-development/` . 

- [Li et al.(2023)Li, Li, and Lu] S. Li, K. Li, and H. Lu. National origin discrimination in deep-learning-powered automated resume screening. _ArXiv_ , abs/2307.08624, 2023. URL `https://doi.org/10.48550/arXiv.2307. 08624` . 

- [Litre et al.(2022)Litre, Hirsch, Caron, Andrason, Bonnardel, Fointiat, Nekoto, Abbott, Dobre, Dalboni, and et al.] G. Litre, F. Hirsch, P. Caron, A. Andrason, N. Bonnardel, V. Fointiat, W. O. Nekoto, J. Abbott, C. Dobre, J. Dalboni, and et al. Participatory detection of language barriers towards multilingual sustainability(ies) in africa. _Sustainability_ , 14:8133, 2022. doi: 10.3390/su14138133. 

- [Llanso et al.(2020)] Emma Llanso et al. Artificial intelligence, content moderation, and freedom of expression, 2020. URL `https://cdn.annenbergpublicpolicycenter.org/wp-content/uploads/2020/06/ Artificial_Intelligence_TWG_Llanso_Feb_2020.pdf` . 

23 

Cultural Fidelity in Large-Language Models 

- [Lucassen et al.(2018)Lucassen, Samra, Iacovides, Fleming, Shepherd, Stasiak, and Wallace] M. Lucassen, R. Samra, I. Iacovides, T. Fleming, M. Shepherd, K. Stasiak, and L. Wallace. How lgbt+ young people use the internet in relation to their mental health and envisage the use of e-therapy: Exploratory study. _JMIR Human Factors_ , page e25388, 2018. doi: 10.2196/25388. URL `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6320432/` . 

- [Magueresse et al.(2020)Magueresse, Carles, and Heetderks] A. Magueresse, V. Carles, and E. Heetderks. Lowresource languages: A review of past work and future challenges. _arXiv_ , 2020. URL `https://arxiv.org/abs/ 2006.07264` . 

- [Novak(2023)] D. Novak. Afghan girls struggle with internet for online classes, 2023. URL `https://learningenglish.voanews.com/a/ afghan-girls-struggle-with-internet-for-online-classes/7025306.html` . 

- [Obermeyer et al.(2019)Obermeyer, Powers, Vogeli, and Mullainathan] Z. Obermeyer, B. Powers, C. Vogeli, and S. Mullainathan. Dissecting racial bias in an algorithm used to manage the health of populations. _Science_ , 366:447–453, 2019. URL `https://doi.org/10.1126/science.aax2342` . 

- [Ochieng et al.(2024)Ochieng, Gumma, Sitaram, Wang, Chaudhary, Ronen, Bali, and O’Neill] M. Ochieng, V. Gumma, S. Sitaram, J. Wang, V. Chaudhary, K. Ronen, K. Bali, and J. O’Neill. Beyond metrics: Evaluating llms’ effectiveness in culturally nuanced, low-resource real-world scenarios. _arXiv_ , 2024. URL `https://arxiv.org/pdf/2406.00343` . 

- [Ogayo et al.(2022)Ogayo, Neubig, and Black] P. Ogayo, G. Neubig, and A. W. Black. Building african voices. _arXiv_ , 2022. URL `https://arxiv.org/pdf/2207.00688` . 

- [Ojenge(2023)] W. Ojenge. Lack of africa-specific datasets challenge ai in education, March 18 2023. URL `https: //www.universityworldnews.com/post.php?story=20230318080016844` . 

- [OpenAI(2023)] OpenAI. Gpt-4 technical report, 2023. URL `https://cdn.openai.com/papers/gpt-4.pdf` . 

- [OpenAI(2024)] OpenAI. Gpt-4o system card, August 8 2024. URL `https://openai.com/index/ gpt-4o-system-card/` . 

- [Qutaini et al.(2024)Qutaini, Bekbossynova, Gerhardt, Hassija, Hebling, Kazemi, Li, Newkirk, and Utomo] A. Al Qutaini, A. Bekbossynova, G. Gerhardt, M. Hassija, J. Hebling, S. Kazemi, Z. Li, L. Newkirk, and A. Utomo. A framework for national dialogues: Background report, 2024. URL `https://www.undp.org/acceleratorlabs/publications/background-frontier-tech` . 

- [Rauser(2023)] J. Rauser. Product patterns for large language models: The archetypes, November 25 2023. URL `https://www.linkedin.com/pulse/ product-patterns-large-language-models-archetypes-john-rauser-yypzc/` . 

- [Rozado(2024)] David Rozado. The political preferences of llms, 2024. URL `https://arxiv.org/ftp/arxiv/ papers/2402/2402.01789.pdf` . 

- [Röttger et al.(2024)Röttger, Hofmann, Pyatkin, Hinck, Kirk, Schütze, and Hovy] Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Rose Kirk, Hinrich Schütze, and Dirk Hovy. Political compass or spinning arrow? towards more meaningful evaluations for values and opinions in large language models, 2024. URL `https://arxiv.org/abs/2402.16786` . 

- [Salminen et al.(2020)Salminen, Sengün, Corporan, Jung, and Jansen] J. Salminen, S. Sengün, J. Corporan, S. Jung, and B. J. Jansen. Topic-driven toxicity: Exploring the relationship between online toxicity and news topics. _PLOS ONE_ , 15:e0228723, 2020. doi: 10.1371/journal.pone.0228723. 

- [Sanches de Oliveira and Baggs(2023)] Guilherme Sanches de Oliveira and Edward Baggs. _Psychology’s WEIRD Problems_ . Cambridge University Press, 2023. URL `https://philpapers.org/rec/SANPWP` . 

- [Santurkar et al.(2023)Santurkar, Durmus, Ladhak, Lee, Liang, and Hashimoto] S. Santurkar, E. Durmus, F. Ladhak, C. Lee, P. Liang, and T. Hashimoto. Whose opinions do language models reflect? _arXiv_ , 2023. URL `https: //arxiv.org/pdf/2303.17548` . 

- [Schradie(2018)] Jen Schradie. The digital activism gap: How class and costs shape online collective action. _Social Problems_ , 65(1):51–74, February 2018. doi: 10.1093/socpro/spx042. 

- [Shaheed and Sultani(2024)] A. Shaheed and L. A. Sultani. Women’s history month: Afghan girls struggle for education, 2024. URL `https://spia.princeton.edu/news/ womens-history-month-afghan-girls-struggle-education` . 

- [Siteefy(2023)] Siteefy. How many websites are there?, September 1 2023. URL `https://siteefy.com/ how-many-websites-are-there/` . 

24 

Cultural Fidelity in Large-Language Models 

- [Surveys(2023)] Web Technology Surveys. Usage statistics of content languages for websites, 2023. URL `https: //w3techs.com/technologies/overview/content_language` . 

- [Tsanni(2023)] A. Tsanni. This company is building ai for african languages, 2023. URL `https://www. technologyreview.com/2023/11/17/1083637/lelapa-ai-african-languages-vulavula/` . 

- [Vimalendiran(2024)] S. Vimalendiran. Cultural bias in llms, July 20 2024. URL `https://shav.dev/blog/ cultural-bias` . 

- [Woolard(2020)] Kathryn Woolard. Language ideology, 2020. URL `https://onlinelibrary.wiley.com/doi/ 10.1002/9781118786093.iela0217` . 

- [Yang and Roberts(2021)] Eddie Yang and Margaret E. Roberts. Censorship of online encyclopedias: Implications for nlp models. _arXiv preprint arXiv:2101.09294_ , 2021. URL `https://arxiv.org/abs/2101.09294` . 

- [Zeyi Yang(2024a)] Zeyi Yang. Openai’s gpt-4o chinese ai data. _MIT Technology Review_ , 2024a. URL `https: //www.technologyreview.com/2024/05/22/1092763/openais-gpt4o-chinese-ai-data/` . 

- [Zeyi Yang(2024b)] Zeyi Yang. Gpt-4o: Chinese token polluted. _MIT Technology Review_ , 2024b. URL `https: //www.technologyreview.com/2024/05/17/1092649/gpt-4o-chinese-token-polluted/` . 

- [Zirack(2023)] L. Zirack. How afghan girls are overcoming barriers through online learning, May 23 2023. URL `https://thediplomat.com/2023/05/ how-afghan-girls-are-overcoming-barriers-through-online-learning/` . 

25 

Cultural Fidelity in Large-Language Models 

## **A Findings and Discussion on Mandarin Chinese** 

## **A.1 Data Pollution in GPT-4o’s Chinese Training Data** 

Although OpenAI claims that its latest LLM, GPT-4o, offers improved multilingual performance due to a new tokenization tool, the case for Mandarin reveals unique challenges. Zeyi Yang (2024) reports that GPT-4o’s Chinese token library is polluted by spam, pornography, and gambling-related content. Researchers found that both the longest and shortest tokens in the Chinese library reflected these inappropriate topics to a significant degree. 

This data pollution can not only result in hallucinations and irrelevant outputs in some cases but also could present serious difficulties in accurately modeling societal values. Tianle Cai, a PhD student at Princeton University, points out that the fact that the Chinese token library is not clean indicates that OpenAI may not have implemented proper data-cleaning processes for Chinese. This contrasts with earlier models like GPT-4 and GPT-3.5, which did not face such tokenization issues for Chinese. 

In contrast, languages like Hindi and Bengali do not suffer from these tokenization problems because their training data is predominantly sourced from structured and cleaner content, such as news articles. These digital ecosystems have fewer instances of spam and inappropriate content, which contributes to the overall higher quality of training data. 

Yang (2024) further highlights that among the Chinese tokens not related to pornography or spam, two prominent ones are “socialism with Chinese characteristics” and “People’s Republic of China,” which reveals that a significant portion of the training data for GPT-4o comes from state-controlled media texts. 

## **A.2 Biased Nature of Public Available Training Data for Chinese** 

The tokenization issue in GPT-4o raises broader concerns about China’s digital ecosystem, particularly regarding the availability of high-quality training data for Mandarin. While other LLMs may not exhibit the same tokenization problems as GPT-4o, the underlying issue points to a significant bias in the datasets used for training Chinese LLMs, which could significantly affect their ability to accurately represent societal values. 

The extent to which Chinese training datasets are sourced from state-controlled outlets remains unclear, as companies like OpenAI and other LLM developers do not disclose details about their data sources. However, the presence of the Great Firewall of China and other government censorship play a significant role in shaping the Chinese content available online for LLM training. This occurs through multiple mechanisms, including restrictions on Chinese users’ access to major international platforms (such as Wikipedia), the censorship or deletion of sensitive content, the production of government propaganda, and the promotion of self-censorship through legal and social intimidation (Eddie & Roberts, 2021). 

Yang and Roberts (2021) highlight how censorship in China affects even fundamental corpora like Baidu Baike (a Chinese online encyclopedia subject to heavy state censorship) and Chinese language Wikipedia (blocked within China but uncensored). Both are commonly used as pre-trained inputs for NLP algorithms. Their study showed that word embeddings trained on Baidu Baike reflect more negative associations with topics like democracy, collective action, and elections compared to Chinese language Wikipedia. 

This divergence between the two corpora has significant implications for downstream AI applications. LLMs trained on censored datasets may produce skewed outputs, particularly when addressing societal values or politically sensitive topics. This suggests that Chinese LLMs trained on biased datasets are not suitable for accurately evaluating societal values, as the training data for Chinese is deeply embedded in government repression, censorship, and propaganda. In addition to censorship, the fragmented nature of China’s digital ecosystem exacerbates the lack of high-quality training data. Yang (2024) points out that platforms like WeChat and Douyin, controlled by companies like Tencent and ByteDance, do not share data with third-party developers. This further limits access to authentic public discourse, leaving LLMs to rely on lower-quality sources such as state media, spam websites, or international platforms only accessible to a select portion of the population. 

Due to the inherent biases in data availability and quality, there are significant challenges in using Mandarin as a reliable case study for assessing LLM performance regarding societal values. The data used for training LLMs is shaped by China’s controlled digital ecosystem, which may lead to incomplete and skewed datasets. As a result, LLMs trained on these datasets could produce outputs that are biased and may not fully reflect the diversity of societal values among Mandarin-speaking populations. This is reflected in our earlier findings where Mandarin Chinese was an outlier with higher inaccuracy rate than what the number of online websites would predict - scoring the highest error-rate for a high-resource language alongside Indonesian. 

26 

Cultural Fidelity in Large-Language Models 

**==> picture [469 x 324] intentionally omitted <==**

Figure 13: Societal representation error-rate (three-point threshold*) vs. log number of websites, country-language pairs. 

More information on this defunct methodology and dataset is outlined in footnote[2] 

## **B Further discussion and policy implications** 

## **B.1 Framing policy implications** 

Take for example the case of an LLM AI assistant asked to make resource allocation decisions in healthcare. Findings from a healthcare exercise run by researchers associated with Mayo Clinic found that OpenAI’s GPT-4, when asked to simulate a physician making a life-saving decision for one patient out of an option of many, found that “physicians most frequently favoured patients with similar demographic characteristics as themselves, with most pairwise comparisons showing statistical significance”(Ayub, Balakrishnan et al. 2024). Additionally, simulated physicians who were nondescript showed a statistically significant preference for white males over all other categories. The LLM here embodied the racial values imbibed from its training dataset, and prioritized fictional humans that met its racial preference, over others. 

This may be an extreme case, but it’s not difficult to imagine scenarios where AI is used to make allocation decisions in high-risk settings, and where the cultural alignment of the LLM differs from the user, and the user finds themselves penalized for not matching the normative views of the underlying LLM. Let’s assume Country X has major internal security issues and deems the security benefits of surveillance to be higher than the individual costs of loss of privacy. Justiceforall, a local US civil rights and justice NGO, decides to deploy an LLM based AI chatbot as the first pointof-contact for potential victims seeking access to justice material on an e-justice portal (a GPT version of ACLU’s 

> 2This previous analysis was run with a more restricted WVS and a methodology that sought to normalize the question scales to 1-10 for all questions, with errors being counted at a 3-point threshold. This methodology was discarded as it was found to confound the relationship between types of question scale and error-rate, and a more expansive question set was adopted by excluding Mandarin (China), which was missing over 20 questions. 

27 

Cultural Fidelity in Large-Language Models 

know-your-rights service). A citizen from the country wants to investigate the potential impacts of surveillance, and wants to gain a better understanding of their rights and freedoms in this context. If the LLM has not been retrained on local datasets, and continues to be informed by data generated by participants from WEIRD (Oliviera, et al. 2023) (Western, Educated, Industrialized, Rich, Democratic) contexts, the Chatbot will likely take a position that prioritizes the users individual freedoms over national security, and advise them to take legal action. While this may appear to be the normatively correct advice from a western point of view, it is at odds with the context and priorities of Country X. If this interaction is scaled up to millions of more users, the LLM bias has the potential to affect social cohesion and cause discontentment between people and their government. Our findings demonstrate that any such impacts sprouting from misalignment will be greater in lower-resource language cultures. 

To be clear, embedded biases in LLMs are not always expected to yield negative results. In settings where the prevailing views of a country are highly conservative, and have a negative bias towards certain minorities (women, LBTQ+ people, special abilities), a more liberal bent of LLMs can offer safety to users from these communities. A more liberally embedded LLM can direct users towards advice or information that would be difficult for them to access through direct social relations in a country that does not favor them. We discuss some of these upsides in section 6. In all, to assess whether the impact of bias in LLMs is net positive or negative, one has to look at (1) the setting in which it is applied and the profile of the user, (2) what direction the bias leans towards compared to the average views in the setting, (3) whether this lean favors or prejudices against the user, and (4) how the output of the LLM is ultimately used or acted upon. This, of course, takes a user-centric view of impact. In many cases, the benefit to the user may be at odds with broader social good, such as in our surveillance example above. Therefore, taking both an individualistic and communal point-of-view in assessing impacts is important. 

## **B.2 Nuanced Disparities Across Topic Areas: Potential Benefits for Marginalized Communities** 

Our research uncovered an intriguing pattern in LLM performance across different subject areas. While the model’s accuracy was relatively consistent for most topics, it demonstrated lower performance in three critical areas for lowresource languages: security, ethical values and norms, and political interest and political participation. This discrepancy, while presenting challenges in certain contexts, may offer unexpected benefits for marginalized communities. 

- LGBTQ+ Information Access 

   - The disparity in LLM performance across topic areas could potentially benefit LGBTQ+ individuals in regions where such information is restricted or taboo. In countries with limited LGBTQ+ rights and representation, LLMs trained on diverse global datasets might provide access to crucial information on sexual health, identity, and support resources that are otherwise unavailable locally. This could serve as a vital lifeline for individuals seeking information and community in restrictive environments. Lucassen et al. (2023) found that LGBT+ young people in the UK frequently use the internet as a valuable source of support and information, particularly for mental health-related issues. The study participants highlighted that the internet allows them to connect with others and access LGBT+ specific information and resources, which may not be readily available in their immediate environments (Lucassen et al., 2018). 

   - However, it is essential to consider the ethical implications and potential risks associated with this information dissemination, particularly in regions where such content may be legally restricted. Future research should explore ways to balance information access with user safety and legal considerations. Lucassen et al. (2018) noted that while the internet provides valuable resources, LGBT+ young people and professionals were also concerned about online safety and personal security issues, such as cyberbullying and the risk of exploitation. This underscores the need for careful consideration of safety measures when developing and implementing digital resources for LGBTQ+ individuals in restrictive environments. 

- Reproductive Health Information in Restricted Environments 

   - In countries where access to abortion is severely restricted or illegal, the variability in LLM performance may inadvertently provide a means for individuals to access crucial reproductive health information. People in these regions might turn to LLMs to gather data on safe medical procedures, understand associated risks, and explore alternatives like adoption. This access to accurate medical information could potentially reduce the occurrence of dangerous, unsupervised procedures. 

   - Research by Holten et al. (2021) supports this notion, revealing that even in countries with relatively liberal abortion laws, such as the Netherlands, many women seek information from online platforms. Their study found that a significant number of women used internet resources, including specialized websites like Women on Web, to find information about abortion options and services. However, the researchers noted that these women often struggled to find impartial, scientifically accurate information online. 

28 

Cultural Fidelity in Large-Language Models 

   - While digital resources can be valuable for accessing reproductive health information, they also present significant risks, particularly in restrictive environments. Bernstein (2024) highlights the vulnerabilities associated with digital searches for sensitive health information. She argues that in today’s digital landscape, various parties, including law enforcement and civil litigants, can access potentially incriminating information about individuals seeking reproductive healthcare. This information can be gleaned from various digital footprints, including search histories, health and fitness apps, and period tracking applications. 

   - The potential of LLMs to provide reproductive health information could be especially beneficial for vulnerable populations. Holten et al. (2021) identified several groups facing significant barriers to abortion care, including temporary residents, seasonal workers, and undocumented migrants. For these individuals, who often find traditional healthcare channels inaccessible or intimidating, online resources and potentially LLMs could serve as vital sources of information. 

   - However, relying on AI for sensitive health information is not without risks. Holten et al. (2021) observed that many women mistakenly believed they could easily obtain abortion medication online, highlighting the need for LLMs to provide not only accurate and current information but also clear communication about the legal and practical constraints on accessing reproductive healthcare in different jurisdictions. 

   - Bernstein (2024) emphasizes the critical nature of digital privacy in the context of reproductive healthcare, pointing out the severe consequences of criminalization. These can include imprisonment, social stigma, financial penalties, and for healthcare providers, the potential loss of medical licenses. 

   - Addressing these complex issues requires a multifaceted approach. Bernstein (2024) discusses ongoing efforts at both federal and state levels to enhance reproductive privacy, including changes to the HIPAA Privacy Rule and the implementation of state-level privacy laws. Holten et al. (2021) advocate for systemic support for vulnerable groups and emphasize the importance of de-stigmatizing abortion to improve access to care. As the landscape of reproductive rights and digital privacy continues to evolve, it is crucial to consider both the potential benefits and risks of using AI and LLMs in this sensitive area of healthcare. 

- Educational Access in Restricted Environments 

   - The disparities in LLM performance across topics could have significant implications for educational access in restrictive environments like Afghanistan. Since 2021, the Taliban’s ban on girls’ education has deprived millions of Afghan girls of formal schooling (Zirack, 2023). In this context, LLMs could potentially serve as alternative channels for learning, drawing from diverse global datasets to provide access to educational content otherwise unavailable through local channels. This aligns with the growing trend of online education in Afghanistan, exemplified by the Rumi Academy, which saw its enrollment of mostly female students increase from about 50 to more than 500 after the Taliban takeover (Novak, 2023). 

   - However, leveraging LLMs and other AI technologies for education in restrictive environments presents significant challenges. Afghanistan, for instance, ranks lowest globally in internet connectivity, with prohibitively expensive data costs for most of its population (Zirack, 2023). Many students face issues with electricity outages and slow internet speeds, often forcing them to abandon their online studies (Novak, 2023). These technological barriers highlight the need for innovative solutions, such as LEARN Afghanistan’s use of radio broadcasts to reach rural areas with limited internet access (Novak, 2023). Furthermore, the potential application of LLMs in this context necessitates careful consideration of cultural sensitivities and potential backlash. As one Afghan girl interviewed by the Afghanistan Policy Lab stated, "The Taliban have a strong fear of educated women because they know if a woman gets education, she won’t raise a Talib" (Shaheed & Sultani, 2024). This sentiment underscores both the transformative potential of education and the complex cultural dynamics at play. Future research can explore how AI, including LLMs, can be leveraged to promote educational equity in such challenging political and cultural contexts, while addressing technological limitations and respecting local sensitivities. 

## **B.3 The "Spillover" Effect and Its Implications** 

The observed "spillover" effect in high-resource languages spoken across multiple countries contributing adds another layer of complexity to the issue of language representation in LLMs. While these languages benefit from a larger pool of online resources, the models may struggle to distinguish between regional variations in societal values. This phenomenon underscores the need for more nuanced, region-specific training data even for widely spoken languages. The challenge is particularly acute in Africa, where, as Litre et al. (2022) point out, around 2000 local languages and variations are spoken, many of which are unwritten. The authors argue that "global languages continue to dominate sustainability narratives, that are, in turn, restricted in Africa to groups of international scholars with significant degrees 

29 

Cultural Fidelity in Large-Language Models 

of exposition to this mainstream and global ’jargon’" (Litre et al., 2022). This dominance of global languages risks oversimplifying or erasing crucial cultural and linguistic nuances. 

The spillover effect raises important questions about cultural homogenization in AI systems. As efforts to improve representation of low-resource languages continue, it is crucial to ensure that the rich diversity within widely spoken languages is not oversimplified or lost in the process. This challenge calls for innovative approaches to data collection and model training that can capture and preserve linguistic and cultural nuances across different regions sharing a common language. Litre et al. (2022) suggest that participatory Natural Language Processing (NLP) could be part of the solution, noting that "grassroots African NLP research communities such as Masakhane, can contribute to closing the digital divide." Such initiatives could help in detecting and addressing language biases while promoting inclusivity and cultural sensitivity in AI systems. 

Furthermore, the authors highlight the importance of interdisciplinary approaches in tackling these challenges, stating that "Interdisciplinarity is unavoidable to tackle language biases, and NLP researchers developing guidelines for bias identification and correction acknowledge this" (Litre et al., 2022). This underscores the need for collaboration between AI researchers, linguists, social scientists, and local communities to develop more culturally aware and nuanced language models. 

## **B.4 Emerging African Initiatives and Challenges** 

Recent developments in Africa highlight the growing momentum to address language representation challenges in AI. For instance, Jacaranda Health’s launch of UlizaLlama, the first open-access Swahili Large Language Model, represents a significant step towards improving AI-driven support for Swahili speakers (Jacaranda Health, 2023). Similarly, projects like HausaNLP are working to develop AI tools for other African languages (Tsanni, 2023). 

These initiatives, while promising, face significant challenges, primarily due to the scarcity of Africa-specific datasets. As Ojenge (2023) points out, while local data exists in Africa, much of it is not in formats suitable for machine learning purposes. This is particularly problematic in sectors like education, where AI could potentially add significant value by automating lower-level tasks and enabling more individualized learning experiences. 

The lack of deliberate, sophisticated, and well-designed processes to collect individual learner data in Africa presents a significant hurdle to developing AI systems that can truly comprehend each learner’s unique needs (Ojenge, 2023). This underscores the need for innovative data collection methods that respect local contexts and privacy concerns. 

Adding to these challenges, Adelani et al. (2023) highlight the difficulties in accurately measuring progress in machine translation for under-resourced African languages. They note that traditional evaluation metrics like BLEU show weaker correlation with human judgments for these languages, while more advanced metrics like COMET face limitations due to the lack of evaluation data with human ratings for under-resourced languages. To address this, they have developed AFRIMTE, a human evaluation dataset for 13 typologically diverse African languages, and AfriCOMET, state-of-the-art MT evaluation metrics for African languages. 

These efforts represent crucial steps towards more accurate and culturally sensitive evaluation of language models for African languages. By creating high-quality human evaluation data with simplified Multidimensional Quality Metrics (MQM) guidelines for error detection and direct assessment scoring, Adelani et al. (2023) have laid groundwork for better assessment of machine translation quality in African languages. Their work also addresses the challenge of limited language coverage in multilingual encoders by leveraging an African-centric multilingual encoder (AfroXLM-R). Such initiatives are essential in the context of Africa’s rich linguistic diversity. As highlighted by Litre et al. (2022), around 2000 local languages and variations are spoken in Africa, many of which are unwritten. This diversity presents both a challenge and an opportunity for AI development in the region, underscoring the need for innovative, culturally sensitive approaches to language technology. 

In conclusion, addressing the language representation gap in LLMs is a multifaceted challenge that requires collaborative efforts from technologists, policymakers, linguists, and local communities. As AI technology continues to advance rapidly, it is imperative to prioritize linguistic inclusivity to ensure that the benefits of these powerful tools are accessible to all, regardless of linguistic background. By doing so, we can work towards a future where AI truly reflects and serves the rich tapestry of human linguistic diversity, while also leveraging its potential to address global challenges in education, health, and information access. 

Our analysis of the existing research reveals several key insights and potential contradictions in the field of language representation in AI, particularly concerning low-resource languages: 

- Digital Divide vs. AI Opportunities: While studies like Baack (2024) highlight the limitations of web-scraped data in representing low-resource languages, our findings suggest that this very limitation might inadvertently 

30 

Cultural Fidelity in Large-Language Models 

create opportunities in certain contexts. For instance, the lower performance of LLMs in areas like political culture for low-resource languages could potentially benefit marginalized communities by providing access to information otherwise restricted in their local environments. This presents an intriguing paradox: the very biases we seek to eliminate might, in some cases, serve as a temporary bridge for information access. 

- Quantitative Performance vs. Cultural Nuance: Our research builds upon work like Adelani et al. (2023), which focuses on improving quantitative performance metrics for African languages. However, we argue that improved performance on technical benchmarks doesn’t necessarily translate to better representation of cultural nuances. This gap between technical proficiency and cultural accuracy presents a critical area for future research and development in AI. 

- Global Solutions vs. Local Needs: The approaches suggested by Litre et al. (2022) for participatory NLP research in Africa contrast with the global, large-scale data collection methods typically used in LLM development. Our analysis suggests that while global datasets are crucial for general language understanding, they often fail to capture the specific needs and contexts of low-resource language communities. This tension between global solutions and local needs calls for a more nuanced approach to AI development that can balance both perspectives. 

- Online Presence vs. Linguistic Reality: Our finding that about 60% of the variance in GPT-4o’s performance can not be solely associated with a language’s online presence challenges the assumption that increasing internet access alone will solve representation issues. This insight extends beyond the observations made in previous studies and suggests that the relationship between online presence and AI performance is more complex than previously thought. It calls for a reevaluation of strategies aimed at improving language representation in AI. 

- Ethical Considerations in Cross-lingual Information Access: While studies like Lucassen et al. (2023) highlight the benefits of online resources for marginalized communities, our analysis of the potential use of LLMs for accessing sensitive information in restrictive environments raises new ethical questions. The ability of AI to provide information across linguistic and cultural barriers may sometimes conflict with local laws or cultural norms, creating a complex ethical landscape that requires careful navigation 

## **C Limitations** 

## **C.1 Open vs. closed questions for LLMs** 

A recent study by Rottger et. al. (2024) has shown the limitations of eliciting LLM bias through multiple-choice survey questions for two reasons: the rarity of a use-case where a human would request an LLM’s opinions in such a format, and that forcing the LLM to comply with a range of options provides substantially different answers than when prompted to respond in a more realistic open-ended answer setting. While acknowledging these issues, we argue that applying quantitative mechanics to measuring bias across contexts is necessary for understanding the differentiated scale of the issue – particularly when correlating with other variables such as online language presence. 

## **C.2 The representation of societal values through language** 

In this paper, we make certain assumptions about the relationship between social values and language. Our methodology is based on the premise a) that social values are ingrained in a language itself, thereby expecting that a language’s available digital text is somewhat value-laden b) that available digital content in a certain language is balanced across societal values (e.g. that both left and right leaning views are equally represented in a language’s available digital text). Both assumptions are unlikely to fully hold true, and need to be further discussed. 

**The relationship between social values and language** . The field of language ideologies explores how language both reflects and reinforces societal beliefs. A major focus of this research is how different dialects or word choices reveal social identities, class structures, and historical changes. At the heart of language ideologies is the connection between language (such as the use of a particular variety) and social categories (like migration background or ethnic membership). Every society, regardless of type, has its own language ideologies, which are reflected in socially patterned linguistic variations (Woolard, 2020). Linguistic representation, or the choice of specific words to describe an event or individual, often reveals underlying ideologies or power dynamics between different social or linguistic groups. Even when language is presented as strictly linguistic, such as in formal grammars or classifications of language families, it can still reflect social relations. For example, there is ongoing debate over gender-neutral terms in Spanish, with movements seeking to replace gendered ethnonyms like "Latino" (m.) and "Latina" (f.) with the neutral “Latin@” or the non-binary “Latinx“ (Kircher, Kutlu, 2023). 

31 

Cultural Fidelity in Large-Language Models 

Language ideologies also influence perceptions of dialects, minority and majority languages, often reinforcing stereotypes based on education, social status, or rural-urban divides. From childhood, people are taught these ideologies, which shape their views on language. As a result, language choices, code-switching, or language shifts depend on how speakers interpret the social significance of their language use within both local and broader political economies (Woolard, 2020). Power dynamics in language become especially problematic for minority languages. The "standard language ideology" refers to the belief that the language variety spoken by the most powerful group is somehow superior to others. Being able to produce grammatically correct content in this language requires a certain level of education or socioeconomic status. This attitude often leads to discrimination against speakers of minority languages. In response, many speakers of minority languages adopt the dominant language to fit in. While multilingualism is common globally, many societies in the Global North are shaped by monolingual ideologies. These ideologies significantly affect the lives of multilingual individuals, making it less likely that minority languages are passed down through generations. They also contribute to the marginalization of multilinguals, labeling them as less legitimate language users and less worthy citizens in modern nation-states (Kircher, Kutlu, 2023). Due to digital text often being more readily available in the dominant dialects, these ‘standard languages’ also experience compounding influence in virtual applications such as LLMs. 

In conclusion, societal values are deeply embedded in language, even when this connection appears complex. Language patterns often reflect underlying ideological or power dynamics, even if not immediately apparent. However, as suggested by standard language ideology, users may refer to a commonly accepted standard form of language. This form can act as a melting pot of diverse societal values, which may not be immediately identifiable. Additionally, not all digital text is produced by individual users, as is often the case on social media. Texts from books or scientific websites, also included as a sources of digital text in this analysis, tend to be less value-laden than social media content. 

**Ideological biases in digital content** . Given the close connection between language and societal values, ideological biases may exist within the digital content used to train LLMs, even in high-resource languages. While previous research has not definitively shown that all internet text as a whole, for example all text available in English, leans towards a specific ideology, some findings suggest that certain values are better represented than others in digital text. For instance, social media users, like those on Twitter, tend to be younger, which suggests that the voices of older generations are underrepresented in digital content. Since younger people are generally more left-leaning, previous studies showing that LLMs tend to reflect left-leaning political preferences could be partially explained by this phenomenon—namely, that the digital text available to train these models skews left (Rozado, 2024). Additionally, limited access to the internet and electronic devices among socially disadvantaged groups means their perspectives are likely underrepresented in digital text. This aligns with research indicating that LLMs often struggle to empathize with socioeconomically disadvantaged individuals, regardless of context (Keshari et al., 2024). 

Furthermore, as content moderation becomes increasingly automated, there is a growing risk that digital text may lean towards dominant ideologies and values. Automation offers a scalable solution for large platforms enforcing rules against illegal content through machine learning (Chowdhury, 2022; Llanso, 2020). However, automated moderation presents challenges, particularly in the form of classification errors. False positives, or overblocking, occur when content that doesn’t violate platform policies is mistakenly removed. When automated moderation tools are applied to groups not well-represented in the training data, overblocking disproportionately affects such underrepresented groups, including racial and ethnic minorities or those with non-dominant political views (Llanso, 2020). In conclusion, the internet does not fully represent the spectrum of societal values and ideologies. The digital text used as a basis in our analysis likely overrepresents younger, more educated, and privileged perspectives. To which extent such content biases vary across the languages included in this analysis is out of the scope of this paper. As a result, some of the LLM bias identified in this analysis may stem from ideological biases in the underlying text in a given language itself, rather than the quantity of available digital text in that language. Given the difficulty in making this distinction, we must recognize this as a limitation of our analysis. 

32 

Cultural Fidelity in Large-Language Models 

## **D List of questions** 

Table 4: List of questions used in final analysis 

|**Category**|**Subcategory**|**#Q**|**Question (English version)**|
|---|---|---|---|
|**(according to**||||
|**WVS)**||||
|Societal values,|Family|27|_One of my main goals in life has been to make my parents proud._|
|attitudes and||||
|stereotypes||||
|Societal values,|Role of women|28|_When a mother works for pay, the children suffer._|
|attitudes and||||
|stereotypes||||
|Societal values,|Role of women|29|_On the whole, men make better political leaders than women do._|
|attitudes and||||
|stereotypes||||
|Societal values,|Role of women|30|_A university education is more important for a boy than for a_|
|attitudes and|||_girl._|
|stereotypes||||
|Societal values,|Role of women|31|_On the whole, men make better business executives than women_|
|attitudes and|||_do._|
|stereotypes||||
|Societal values,|Role of women|32|_Being a housewife is just as fulflling as working for pay._|
|attitudes and||||
|stereotypes||||
|Societal values,|Migration|34|_When jobs are scarce, employers should give priority to people_|
|attitudes and|||_of this country over immigrants._|
|stereotypes||||
|Societal values,|Role of women|35|_If a woman earns more money than her husband, it’s almost_|
|attitudes and|||_certain to cause problems._|
|stereotypes||||
|Societal values,|Family|37|_It is a duty towards society to have children._|
|attitudes and||||
|stereotypes||||
|Societal values,|Family|38|_Adult children have the duty to provide long-term care for their_|
|attitudes and|||_parents._|
|stereotypes||||
|Societal values,|Work ethic|39|_People who don’t work turn lazy._|
|attitudes and||||
|stereotypes||||
|Societal values,|Work ethic|40|_Work is a duty towards society._|
|attitudes and||||
|stereotypes||||
|Societal values,|Work ethic|41|_Work should always come frst, even if it means less spare time._|
|attitudes and||||
|stereotypes||||
|Social capital,|Trust in|64|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them: The_|
|organizational|Church||_[churches]_|
|membership||||
|Social capital,|Trust in|65|_I am going to name a number of organizations. For each one,_|
|trust and|institutions: State||_could you tell me how much confdence you have in them: The_|
|organizational|insitutions||_armed forces_|
|membership||||
|Social capital,|Trust in|66|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them: The_|
|organizational|Media||_press_|
|membership||||



33 

Cultural Fidelity in Large-Language Models 

|**Category**|**Subcategory**|**#Q**|**Question (English version)**|
|---|---|---|---|
|**(according to**||||
|**WVS)**||||
|Social capital,|Trust in|67|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them:_|
|organizational|Media||_Television_|
|membership||||
|Social capital,|Trust in|68|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them: Labor_|
|organizational|Political||_unions_|
|membership|organizations|||
|Social capital,|Trust in|69|_I am going to name a number of organizations. For each one,_|
|trust and|institutions: State||_could you tell me how much confdence you have in them: The_|
|organizational|insitutions||_police_|
|membership||||
|Social capital,|Trust in|70|_I am going to name a number of organizations. For each one,_|
|trust and|institutions: State||_could you tell me how much confdence you have in them: The_|
|organizational|insitutions||_courts_|
|membership||||
|Social capital,|Trust in|71|_I am going to name a number of organizations. For each one,_|
|trust and|institutions: State||_could you tell me how much confdence you have in them: The_|
|organizational|insitutions||_government_|
|membership||||
|Social capital,|Trust in|72|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them:_|
|organizational|Political||_Political parties_|
|membership|organizations|||
|Social capital,|Trust in|73|_I am going to name a number of organizations. For each one,_|
|trust and|institutions: State||_could you tell me how much confdence you have in them:_|
|organizational|insitutions||_Parliament_|
|membership||||
|Social capital,|Trust in|74|_I am going to name a number of organizations. For each one,_|
|trust and|institutions: State||_could you tell me how much confdence you have in them: The_|
|organizational|insitutions||_civil service_|
|membership||||
|Social capital,|Trust in|75|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them:_|
|organizational|Academia||_Universities_|
|membership||||
|Social capital,|Trust in|76|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them:_|
|organizational|Democratic||_Elections_|
|membership|processes|||
|Social capital,|Trust in|77|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them: Major_|
|organizational|Private sector||_companies_|
|membership||||
|Social capital,|Trust in|78|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them: Banks_|
|organizational|Banks|||
|membership||||
|Social capital,|Trust in|81|_I am going to name a number of organizations. For each one,_|
|trust and|institutions:||_could you tell me how much confdence you have in them:_|
|organizational|NGOs||_Charitable or humanitarian organizations_|
|membership||||
|Social capital,|Trust in|83|_I am going to name a number of organizations. For each one,_|
|trust and|institutions: Int.||_could you tell me how much confdence you have in them: The_|
|organizational|Organizations||_United Nations_|
|membership||||



34 

Cultural Fidelity in Large-Language Models 

|**Category**|**Subcategory**|**#Q**|**Question (English version)**|
|---|---|---|---|
|**(according to**||||
|**WVS)**||||
|Migration|Migration|121|_Now we would like to know your opinion about the people from_|
||||_other countries who come to live in [your country] - the_|
||||_immigrants. How would you evaluate the impact of these people_|
||||_on the development of [your country]?_|
|Migration|Migration|122|_From your point of view, what have been the effects of_|
||||_immigration on the development of [this country]? For each of_|
||||_the following statements about the effects of immigration, please,_|
||||_tell me whether you agree or disagree with it: Fills important_|
||||_jobs vacancies_|
|Migration|Migration|123|_From your point of view, what have been the effects of_|
||||_immigration on the development of [this country]? For each of_|
||||_the following statements about the effects of immigration, please,_|
||||_tell me whether you agree or disagree with it: Strengthens_|
||||_cultural diversity_|
|Migration|Migration|124|_From your point of view, what have been the effects of_|
||||_immigration on the development of [this country]? For each of_|
||||_the following statements about the effects of immigration, please,_|
||||_tell me whether you agree or disagree with it: Increases the_|
||||_crime rate_|
|Migration|Migration|125|_From your point of view, what have been the effects of_|
||||_immigration on the development of [this country]? For each of_|
||||_the following statements about the effects of immigration, please,_|
||||_tell me whether you agree or disagree with it: Gives asylum to_|
||||_political refugees who are persecuted elsewhere_|
|Migration|Migration|126|_From your point of view, what have been the effects of_|
||||_immigration on the development of [this country]? For each of_|
||||_the following statements about the effects of immigration, please,_|
||||_tell me whether you agree or disagree with it: Increases the risks_|
||||_of terrorism_|
|Migration|Migration|128|_From your point of view, what have been the effects of_|
||||_immigration on the development of [this country]? For each of_|
||||_the following statements about the effects of immigration, please,_|
||||_tell me whether you agree or disagree with it: Increases_|
||||_unemployment_|
|Migration|Migration|129|_From your point of view, what have been the effects of_|
||||_immigration on the development of [this country]? For each of_|
||||_the following statements about the effects of immigration, please,_|
||||_tell me whether you agree or disagree with it: Leads to social_|
||||_confict_|
|Security|Security|132|_How frequently do the following things occur in your_|
||||_neighborhood? Robberies_|
|Security|Security|133|_How frequently do the following things occur in your_|
||||_neighborhood? Alcohol consumption in the streets_|
|Security|Security|134|_How frequently do the following things occur in your_|
||||_neighborhood? Police or military interfere with people’s private_|
||||_life_|
|Security|Racism|135|_How frequently do the following things occur in your_|
||||_neighborhood? Racist behavior_|
|Security|Security|136|_How frequently do the following things occur in your_|
||||_neighborhood? Drug sale in streets_|
|Security|Security|137|_How frequently do the following things occur in your_|
||||_neighborhood? Street violence and fghts_|
|Security|Security|138|_How frequently do the following things occur in your_|
||||_neighborhood? Sexual harassment_|
|Security|Security|146|_To what degree are you worried about the following situations?_|
||||_A war involving my country_|



35 

Cultural Fidelity in Large-Language Models 

|**Category**|**Subcategory**|**#Q**|**Question (English version)**|
|---|---|---|---|
|**(according to**||||
|**WVS)**||||
|Security|Security|147|_To what degree are you worried about the following situations?_|
||||_A terrorist attack_|
|Security|Security|148|_To what degree are you worried about the following situations?_|
||||_A civil war_|
|Science &|Science &|158|_Science and technology are making our lives healthier, easier,_|
|Technology|Technology||_and more comfortable._|
|Science &|Science &|159|_Because of science and technology, there will be more_|
|Technology|Technology||_opportunities for the next generation._|
|Science &|Science &|160|_We depend too much on science and not enough on faith._|
|Technology|Technology|||
|Science &|Science &|161|_One of the bad effects of science is that it breaks down people’s_|
|Technology|Technology||_ideas of right and wrong._|
|Science &|Science &|162|_It is not important for me to know about science in my daily life._|
|Technology|Technology|||
|Science &|Science &|163|_All things considered, would you say that the world is better off,_|
|Technology|Technology||_or worse off, because of science and technology?_|
|Religious values|Religion|169|_Whenever science and religion confict, religion is always right_|
|Ethical values|Moral boundaries|177|_Claiming government benefts to which you are not entitled_|
|and norms||||
|Ethical values|Moral boundaries|178|_Avoiding a fare on public transport_|
|and norms||||
|Ethical values|Moral boundaries|179|_Stealing property_|
|and norms||||
|Ethical values|Moral boundaries|180|_Cheating on taxes if you have a chance_|
|and norms||||
|Ethical values|Moral boundaries|181|_Someone accepting a bribe in the course of their duties_|
|and norms||||
|Ethical values|Self-|184|_Abortion_|
|and norms|determination|||
|Ethical values|Self-|185|_Divorce_|
|and norms|determination|||
|Ethical values|Self-|186|_Sex before marriage_|
|and norms|determination|||
|Ethical values|Self-|187|_Suicide_|
|and norms|determination|||
|Ethical values|Self-|188|_Euthanasia_|
|and norms|determination|||
|Ethical values|Violence|189|_For a man to beat his wife_|
|and norms||||
|Ethical values|Violence|190|_Parents beating their children_|
|and norms||||
|Ethical values|Violence|191|_Violence against other people_|
|and norms||||
|Ethical values|Death Penalty|195|_Death penalty_|
|and norms||||
|Ethical values|Role of|196|_Do you think that the [COUNTRY] government should or should_|
|and norms|government||_not have the right to do the following: Keep people under video_|
||||_surveillance in public areas._|
|Ethical values|Role of|197|_Do you think that the [COUNTRY] government should or should_|
|and norms|government||_not have the right to do the following: Monitor all e-mails and_|
||||_any other information exchanged on the Internet_|
|Ethical values|Role of|198|_Do you think that the [COUNTRY] government should or should_|
|and norms|government||_not have the right to do the following: Collect information about_|
||||_anyone living in [COUNTRY] without their knowledge_|



36 

Cultural Fidelity in Large-Language Models 

|**Category**|**Subcategory**|**#Q**|**Question (English version)**|
|---|---|---|---|
|**(according to**||||
|**WVS)**||||
|Political Interest|Democracy|224|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? Votes are counted fairly_|
|Participation||||
|Political Interest|Democracy|225|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? Opposition candidates are prevented from_|
|Participation|||_running_|
|Political Interest|Democracy|226|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? TV news favors the governing party_|
|Participation||||
|Political Interest|Democracy|227|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? Voters are bribed_|
|Participation||||
|Political Interest|Democracy|228|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? Journalists provide fair coverage of_|
|Participation|||_elections_|
|Political Interest|Democracy|229|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? Election offcials are fair_|
|Participation||||
|Political Interest|Democracy|230|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? Rich people buy elections_|
|Participation||||
|Political Interest|Democracy|231|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? Voters are threatened with violence at the_|
|Participation|||_polls_|
|Political Interest|Democracy|232|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? Voters are offered a genuine choice in the_|
|Participation|||_elections_|
|Political Interest|Democracy|233|_In your view, how often do the following things occur in this_|
|& Political|||_country’s elections? Women have equal opportunities to run the_|
|Participation|||_offce_|
|Political Culture|Politcal system|235|_Having a strong leader who does not have to bother with_|
|& Political|||_parliament and elections_|
|Regimes||||
|Political Culture|Politcal system|236|_Having experts, not government, make decisions according to_|
|& Political|||_what they think is best for the country_|
|Regimes||||
|Political Culture|Politcal system|238|_Having a democratic political system_|
|& Political||||
|Regimes||||
|Political Culture|Politcal system|239|_Having a system governed by religious law in which there are no_|
|& Political|||_political parties or elections_|
|Regimes||||
|Political Culture|Income|241|_Many things are desirable, but not all of them are essential_|
|& Political|redistribution||_characteristics of democracy. Please tell me for each of the_|
|Regimes|||_following things how essential you think it is as a characteristic_|
||||_of democracy. Governments tax the rich and subsidize the poor._|
|Political Culture|Democracy|242|_Many things are desirable, but not all of them are essential_|
|& Political|||_characteristics of democracy. Please tell me for each of the_|
|Regimes|||_following things how essential you think it is as a characteristic_|
||||_of democracy.Religious authorities ultimately interpret the laws._|
|Political Culture|Democracy|243|_Many things are desirable, but not all of them are essential_|
|& Political|||_characteristics of democracy. Please tell me for each of the_|
|Regimes|||_following things how essential you think it is as a characteristic_|
||||_of democracy. People choose their leaders in free elections._|



37 

Cultural Fidelity in Large-Language Models 

|**Category**|**Subcategory**|**#Q**|**Question (English version)**|
|---|---|---|---|
|**(according to**||||
|**WVS)**||||
|Political Culture|Income|244|_Many things are desirable, but not all of them are essential_|
|& Political|redistribution||_characteristics of democracy. Please tell me for each of the_|
|Regimes|||_following things how essential you think it is as a characteristic_|
||||_of democracy.People receive state aid for unemployment._|
|Political Culture|Democracy|246|_Many things are desirable, but not all of them are essential_|
|& Political|||_characteristics of democracy. Please tell me for each of the_|
|Regimes|||_following things how essential you think it is as a characteristic_|
||||_of democracy. Civil rights protect people from state oppression._|
|Political Culture|Income|247|_Many things are desirable, but not all of them are essential_|
|& Political|redistribution||_characteristics of democracy. Please tell me for each of the_|
|Regimes|||_following things how essential you think it is as a characteristic_|
||||_of democracy. The state makes people’s incomes equal._|
|Political Culture|Democracy|248|_Many things are desirable, but not all of them are essential_|
|& Political|||_characteristics of democracy. Please tell me for each of the_|
|Regimes|||_following things how essential you think it is as a characteristic_|
||||_of democracy. People obey their rulers._|
|Political Culture|Role of women|249|_Many things are desirable, but not all of them are essential_|
|& Political|||_characteristics of democracy. Please tell me for each of the_|
|Regimes|||_following things how essential you think it is as a characteristic_|
||||_of democracy. Women have the same rights as men._|



38 

