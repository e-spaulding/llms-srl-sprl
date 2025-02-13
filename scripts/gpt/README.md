# Experiments via OpenAI API

This directory contains various scripts for running batches of prompts through OpenAI's API.

This creates large batch files and queries the API all in the same run, so I recommend commenting out the code that sends the batch and inspecting the .jsonl file to make sure it's what you want before running the full code.

## Pipeline

A full SRL-SPRL pipeline can be run with

```
python -m scripts.gpt.pipeline path/to/config.yaml
```

This script generates and runs several batches of prompts, waiting for the first batch and then re-generating new prompts based on the output, so please take caution while running it.

## SRL/SPRL with pre-specified predicates and arguments

Run SRL and SRL(+SPRL context) with the following command:

```
python -m scripts.gpt.srl_and_sprl_only run-srl
```

You can also run SPRL-annotate prompts on GPT by running

```
python -m scripts.gpt.srl_and_sprl_only run-sprl-annotate-prompts
```

(You may have to go in to that function and change the paths according to the exact experiment you want to run.)