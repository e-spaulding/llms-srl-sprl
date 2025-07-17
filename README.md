# On the Role of Semantic Proto-roles in Semantic Analysis: What do LLMs know about agency?

This repository provides code and instructions to replicate the results in the paper *On the Role of Semantic Proto-roles in Semantic Analysis: What do LLMs know about agency?*, published in ACL Findings 2025.

## Replication

### Data prep

Navigate to the `data-prep` directory and follow the instructions; download the necessary data, set the necessary env variables, and run the preprocessing script `data-prep/preprocess-data.sh`.

### Experiments

Install the required packages in `requirements.txt` (especially if you are preprocessing the data). Find instructions to run the GPT experiments in `scripts/gpt`, and the smaller LM experiments in `scripts/hf`.

## Cite this paper

```
@inproceedings{spaulding-etal-2025-role,
    title = "On the Role of Semantic Proto-roles in Semantic Analysis: What do LLMs know about agency?",
    author = "Spaulding, Elizabeth  and
      Ahmed, Shafiuddin Ahmed  and
      Martin, James",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = july,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics"
}
```
