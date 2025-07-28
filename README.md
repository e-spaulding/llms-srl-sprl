# On the Role of Semantic Proto-roles in Semantic Analysis: What do LLMs know about agency?

This repository provides code and instructions to replicate the results in the paper [On the Role of Semantic Proto-roles in Semantic Analysis](https://aclanthology.org/2025.findings-acl.623/), published in ACL Findings 2025.

## Replication

### Data prep

Navigate to the `data-prep` directory and follow the instructions; download the necessary data, set the necessary env variables, and run the preprocessing script `data-prep/preprocess-data.sh`.

### Experiments

Install the required packages in `requirements.txt` (especially if you are preprocessing the data). Find instructions to run the GPT experiments in `scripts/gpt`, and the smaller LM experiments in `scripts/hf`.

## Cite this paper

```
@inproceedings{spaulding-etal-2025-role,
    title = "On the Role of Semantic Proto-roles in Semantic Analysis: What do {LLM}s know about agency?",
    author = "Spaulding, Elizabeth  and
      Ahmed, Shafiuddin Rehan  and
      Martin, James",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.623/",
    pages = "12027--12048",
    ISBN = "979-8-89176-256-5"
}
```
