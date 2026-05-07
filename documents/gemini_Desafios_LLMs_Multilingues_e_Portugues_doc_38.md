@inproceedings{nina-etal-2026-efficient,
title = "Efficient Fine-Tuning Methods for {P}ortuguese Question Answering: A Comparative Study of {PEFT} on {BERT}imbau and Exploratory Evaluation of Generative {LLM}s",
author = "Nina, Mariela M. and
Costa, Caio Veloso and
Berton, Lilian and
Vega-Oliveros, Didier A.",
editor = "Souza, Marlo and
de-Dios-Flores, Iria and
Santos, Diana and
Freitas, Larissa and
Souza, Jackson Wilke da Cruz and
Ribeiro, Eug{\'e}nio",
booktitle = "Proceedings of the 17th International Conference on Computational Processing of {P}ortuguese ({PROPOR} 2026) - Vol. 1",
month = apr,
year = "2026",
address = "Salvador, Brazil",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/2026.propor-1.91/",
pages = "917--926",
ISBN = "979-8-89176-387-6",
abstract = "Although large language models have transformed natural language processing, their computational costs create accessibility barriers for low-resource languages such as Brazilian Portuguese. This work presents a systematic evaluation of Parameter-Efficient Fine-Tuning (PEFT) and quantization techniques applied to BERTimbau for Question Answering on SQuAD-BR, the Brazilian Portuguese translation of SQuAD v1.We evaluate 40 configurations combining four PEFT methods (LoRA, DoRA, QLoRA, QDoRA) across two model sizes (Base: 110M and Large: 335M parameters). Our findings reveal three critical insights: (1) LoRA achieves 95.8{\%} of baseline performance on BERTimbau-Large while reducing training time by 73.5{\%} (F1 = 81.32 vs. 84.86); (2) higher learning rates (2e-4) substantially improve PEFT performance, with F1 gains of up to +19.71 points compared to standard rates; and (3) larger models show twice the quantization resilience (loss of 4.83 vs. 9.56 F1 points).These results demonstrate that encoder-based models can be efficiently fine-tuned for extractive Brazilian Portuguese question answering with substantially lower computational cost than large generative LLMs, promoting more sustainable approaches aligned with Green AI principles. An exploratory evaluation of Tucano and Sabi{\'a} on the same benchmark shows that although generative models can achieve competitive F1 scores with LoRA fine-tuning, they require up to 4.2 times more GPU memory and three times more training time than BERTimbau-Base, reinforcing the efficiency advantage of smaller encoder-based architectures for this task."
}

<?xml version="1.0" encoding="UTF-8"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
<mods ID="nina-etal-2026-efficient">
<titleInfo>
<title>Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs</title>
</titleInfo>
<name type="personal">
<namePart type="given">Mariela</namePart>
<namePart type="given">M</namePart>
<namePart type="family">Nina</namePart>
<role>
<roleTerm authority="marcrelator" type="text">author</roleTerm>
</role>
</name>
<name type="personal">
<namePart type="given">Caio</namePart>
<namePart type="given">Veloso</namePart>
<namePart type="family">Costa</namePart>
<role>
<roleTerm authority="marcrelator" type="text">author</roleTerm>
</role>
</name>
<name type="personal">
<namePart type="given">Lilian</namePart>
<namePart type="family">Berton</namePart>
<role>
<roleTerm authority="marcrelator" type="text">author</roleTerm>
</role>
</name>
<name type="personal">
<namePart type="given">Didier</namePart>
<namePart type="given">A</namePart>
<namePart type="family">Vega-Oliveros</namePart>
<role>
<roleTerm authority="marcrelator" type="text">author</roleTerm>
</role>
</name>
<originInfo>
<dateIssued>2026-04</dateIssued>
</originInfo>
<typeOfResource>text</typeOfResource>
<relatedItem type="host">
<titleInfo>
<title>Proceedings of the 17th International Conference on Computational Processing of Portuguese (PROPOR 2026) - Vol. 1</title>
</titleInfo>
<name type="personal">
<namePart type="given">Marlo</namePart>
<namePart type="family">Souza</namePart>
<role>
<roleTerm authority="marcrelator" type="text">editor</roleTerm>
</role>
</name>
<name type="personal">
<namePart type="given">Iria</namePart>
<namePart type="family">de-Dios-Flores</namePart>
<role>
<roleTerm authority="marcrelator" type="text">editor</roleTerm>
</role>
</name>
<name type="personal">
<namePart type="given">Diana</namePart>
<namePart type="family">Santos</namePart>
<role>
<roleTerm authority="marcrelator" type="text">editor</roleTerm>
</role>
</name>
<name type="personal">
<namePart type="given">Larissa</namePart>
<namePart type="family">Freitas</namePart>
<role>
<roleTerm authority="marcrelator" type="text">editor</roleTerm>
</role>
</name>
<name type="personal">
<namePart type="given">Jackson</namePart>
<namePart type="given">Wilke</namePart>
<namePart type="given">da</namePart>
<namePart type="given">Cruz</namePart>
<namePart type="family">Souza</namePart>
<role>
<roleTerm authority="marcrelator" type="text">editor</roleTerm>
</role>
</name>
<name type="personal">
<namePart type="given">Eugénio</namePart>
<namePart type="family">Ribeiro</namePart>
<role>
<roleTerm authority="marcrelator" type="text">editor</roleTerm>
</role>
</name>
<originInfo>
<publisher>Association for Computational Linguistics</publisher>
<place>
<placeTerm type="text">Salvador, Brazil</placeTerm>
</place>
</originInfo>
<genre authority="marcgt">conference publication</genre>
<identifier type="isbn">979-8-89176-387-6</identifier>
</relatedItem>
<abstract>Although large language models have transformed natural language processing, their computational costs create accessibility barriers for low-resource languages such as Brazilian Portuguese. This work presents a systematic evaluation of Parameter-Efficient Fine-Tuning (PEFT) and quantization techniques applied to BERTimbau for Question Answering on SQuAD-BR, the Brazilian Portuguese translation of SQuAD v1.We evaluate 40 configurations combining four PEFT methods (LoRA, DoRA, QLoRA, QDoRA) across two model sizes (Base: 110M and Large: 335M parameters). Our findings reveal three critical insights: (1) LoRA achieves 95.8% of baseline performance on BERTimbau-Large while reducing training time by 73.5% (F1 = 81.32 vs. 84.86); (2) higher learning rates (2e-4) substantially improve PEFT performance, with F1 gains of up to +19.71 points compared to standard rates; and (3) larger models show twice the quantization resilience (loss of 4.83 vs. 9.56 F1 points).These results demonstrate that encoder-based models can be efficiently fine-tuned for extractive Brazilian Portuguese question answering with substantially lower computational cost than large generative LLMs, promoting more sustainable approaches aligned with Green AI principles. An exploratory evaluation of Tucano and Sabiá on the same benchmark shows that although generative models can achieve competitive F1 scores with LoRA fine-tuning, they require up to 4.2 times more GPU memory and three times more training time than BERTimbau-Base, reinforcing the efficiency advantage of smaller encoder-based architectures for this task.</abstract>
<identifier type="citekey">nina-etal-2026-efficient</identifier>
<location>
<url>https://aclanthology.org/2026.propor-1.91/</url>
</location>
<part>
<date>2026-04</date>
<extent unit="page">
<start>917</start>
<end>926</end>
</extent>
</part>
</mods>
</modsCollection>

%0 Conference Proceedings
%T Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs
%A Nina, Mariela M.
%A Costa, Caio Veloso
%A Berton, Lilian
%A Vega-Oliveros, Didier A.
%Y Souza, Marlo
%Y de-Dios-Flores, Iria
%Y Santos, Diana
%Y Freitas, Larissa
%Y Souza, Jackson Wilke da Cruz
%Y Ribeiro, Eugénio
%S Proceedings of the 17th International Conference on Computational Processing of Portuguese (PROPOR 2026) - Vol. 1
%D 2026
%8 April
%I Association for Computational Linguistics
%C Salvador, Brazil
%@ 979-8-89176-387-6
%F nina-etal-2026-efficient
%X Although large language models have transformed natural language processing, their computational costs create accessibility barriers for low-resource languages such as Brazilian Portuguese. This work presents a systematic evaluation of Parameter-Efficient Fine-Tuning (PEFT) and quantization techniques applied to BERTimbau for Question Answering on SQuAD-BR, the Brazilian Portuguese translation of SQuAD v1.We evaluate 40 configurations combining four PEFT methods (LoRA, DoRA, QLoRA, QDoRA) across two model sizes (Base: 110M and Large: 335M parameters). Our findings reveal three critical insights: (1) LoRA achieves 95.8% of baseline performance on BERTimbau-Large while reducing training time by 73.5% (F1 = 81.32 vs. 84.86); (2) higher learning rates (2e-4) substantially improve PEFT performance, with F1 gains of up to +19.71 points compared to standard rates; and (3) larger models show twice the quantization resilience (loss of 4.83 vs. 9.56 F1 points).These results demonstrate that encoder-based models can be efficiently fine-tuned for extractive Brazilian Portuguese question answering with substantially lower computational cost than large generative LLMs, promoting more sustainable approaches aligned with Green AI principles. An exploratory evaluation of Tucano and Sabiá on the same benchmark shows that although generative models can achieve competitive F1 scores with LoRA fine-tuning, they require up to 4.2 times more GPU memory and three times more training time than BERTimbau-Base, reinforcing the efficiency advantage of smaller encoder-based architectures for this task.
%U https://aclanthology.org/2026.propor-1.91/
%P 917-926

##### Markdown (Informal)

[Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs](https://aclanthology.org/2026.propor-1.91/) (Nina et al., PROPOR 2026)

##### ACL