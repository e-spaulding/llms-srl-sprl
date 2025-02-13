#!/bin/bash

# CoNLL2012
echo "Running scripts to preprocess CoNLL 2012..."
cd data-prep/propbank-release/docs/scripts
echo "Running script to build gold conll files from ontonotes. This may take a few minutes. ..."
python map_all_to_conll.py --ontonotes $ONTONOTES_DIR
cd ../../../..
echo "Preprocessing CoNLL2012 data for this repo..."
python -m scripts.preprocessing.preprocess ontonotes-split-maker ./data-prep/propbank-release/docs/evaluation/ ./data-prep/ontonotes-splits.tsv
python -m scripts.preprocessing.preprocess conll12 ./data-prep/propbank-release/data/ontonotes ./corpus/conll12 --split-file-path ./data-prep/ontonotes-splits.tsv

# SPRL
echo "Running scripts to preprocess SPR data..."
python -m scripts.preprocessing.preprocess spr $SPRL_DIR ./corpus/sprl 1
python -m scripts.preprocessing.preprocess spr $SPRL_DIR ./corpus/sprl 2.1

# SRL + SPRL
echo "Running scripts to add SRL annotations to SPR data..."
python -m scripts.preprocessing.add_srl_to_sprl1