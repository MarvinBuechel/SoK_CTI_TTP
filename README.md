# SoK: Automated TTP Extraction from CTI Reports – Are We There Yet?

> [!IMPORTANT]  
> This repository contains the code for reproducing the original paper. However, due to the nature of training language models for classification and generation, which includes some randomness, there may be minor diviations with the tables reported in the original paper. However, on a high level, we believe the conclusions of the paper to still be valid.

This repository contains the code of the paper *"SoK: Automated TTP Extraction from CTI Reports – Are We There Yet?"* [[1]](#references) [PDF](https://thijsvane.de/static/homepage/papers/SoK_Automated_TTP_Extraction_from_CTI_Reports-Are_We_There_Yet.pdf). 

## Introduction
Cyber Threat Intelligence (CTI) plays a critical role in sharing knowledge about new and evolving threats.
With the increased prevalence and sophistication of threat actors, intelligence has expanded from simple indicators of compromise to extensive CTI reports describing high-level attack steps known as Tactics, Techniques and Procedures (TTPs).
Such TTPs, often classified into the ontology of the ATT&CK framework, make CTI significantly more valuable, but also harder to interpret and automatically process.
Natural Language Processing (NLP) makes it possible to automate large parts of the knowledge extraction from CTI reports; over 40 papers discuss approaches, ranging from named entity recognition over embedder models to generative large language models. Unfortunately, existing solutions are largely incomparable as they consider decisively different and constrained settings, rely on custom TTP ontologies, and use a multitude of custom, inaccessible CTI datasets.
We take stock, systematize the knowledge in the field, and empirically evaluate existing approaches in a unified setting for fair comparisons. We gain several fundamental insights, including (1) the finding of a kind of performance limit that existing approaches seemingly cannot overcome as of yet, (2) that traditional NLP approaches (possibly counterintuitively) outperform modern embedder-based and generative approaches in realistic settings, and (3) that further research on understanding inherent ambiguities in TTP ontologies and on the creation of qualitative datasets is key to take a leap in the field.

## Organization
The repository is structured as follows:  
* `NER`: Contains the code related to the NER approaches (Sections 2.2, 4).
* `classification`: Contains related to the Classification approaches (Sections 2.3, 5)
* `generative`: Contains the code related to the Generation approaches (Sections 2.4, 6)
* `scraping`: Contains the code for collecting papers from DBLP and Google Scholar, following the description of Section 2.1
* `ext_tools`: Contains the code of the comparison experiment presented in Appendix A.1, with the implementations of three state-of-the-art approaches.
* `datasets`: Contains the used datasets MITRE TRAM2 with our proposed split, Bosch AnnoCTR dataset, Augmented TRAM2 dataset and the corresponding instruction datasets to train the generative LLM in natural language.

Each folder (NER, classification, gLLM) corresponds to a *sub-project* and, therefore, requires its own setup. To use the code, navigate inside the corresponding folder and follow the instructions listed in the corresponding `README.md`.

> [!TIP]
> Throughout this repository, the Bosch AnnoCTR dataset is often referred as *"bosch"*.


## References
[1] `Marvin Büchel, Tommaso Paladini, Stefano Longari, Michele Carminati, Stefano Zanero, Hodaya Binyamini, Gal Engelberg, Dan Klein, Giancarlo Guizzardi, Marco Caselli, Andrea Continella, Maarten van Steen, Andreas Peter, Thijs van Ede. (2025, August). SoK: Automated TTP Extraction from CTI Reports – Are We There Yet? In 34th USENIX Security Symposium (USENIX). USENIX.`

### Bibtex
```
@inproceedings{buechel2025sok,
  title={{SoK: Automated TTP Extraction from CTI Reports – Are We There Yet?}},
  author={Büchel, Marvin and Paladini, Tommaso and Longari, Stefano and Carminati, Michele and Zanero, Stefano and Binyamini, Hodaya and Engelberg, Gal and Klein, Dan and Guizzardi, Giancarlo and Caselli, Marco and Continella, Andrea and van Steen, Maarten and Peter, Andreas and van Ede, Thijs},
  booktitle={34th USENIX Security Symposium (USENIX)},
  year={2025},
  organization={USENIX}
}
```
