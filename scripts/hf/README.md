# LMs accessed via HuggingFace

This directory contains a simple ad-hoc script that uses a single GPU to run prompts (SPRL-annotate and One-arg) on various lightweight language models accessed via HuggingFace Transformers.

## How to run

Install the needed libraries in a virtual environment:

```
python3 -m venv /path/to/.venv
source /path/to/.venv/bin/activate
pip install -r requirements.txt
```

Log into the huggingface hub to get the weights:

```
huggingface-cli login
```

Then run the python script!

```
python run_prompts.py
```

The `experiments` directory will be populated with output from the model.