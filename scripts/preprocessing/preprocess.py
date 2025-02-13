import json
import pickle
import os
import spacy
import typer
import itertools

from typing import Optional
from spacy.tokens import Doc
from tqdm import tqdm
from glob import glob

from ..data2prompts.items import SpanItem, PredicateItem, RelationItem, SentenceItem
from ..util.util import ensure_output_path_exists, two_way_split
from .conll import get_conll_sentence_lines, get_pred_relations_2005, get_pred_relations_2012

# ---------------- Preliminaries --------------- #
from ..util.macros import SPR1_LABELS, SPR2_LABELS
app = typer.Typer(
    pretty_exceptions_show_locals=False
)

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
            spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)

# ----------- Command line arguments ----------- #

@app.command()
def spr(
    input_path: str,
    output_path: str,
    dataset: str,
    collapse_na_to_false: Optional[bool] = False
):
    input_filename = f'{input_path}/spr{dataset}.json'
    try:
        input_file = open(input_filename)
    except:
        raise ValueError(f"Could not find {input_filename}.")
    output_path = ensure_output_path_exists(output_path)

    spr = json.load(input_file)

    nlp = spacy.blank("en")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab) 

    SPLITS = ["train", "dev", "test"]

    if dataset == '1':
        properties = SPR1_LABELS
    elif dataset == '2' or dataset == '2.1':
        properties = SPR2_LABELS
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    for split in SPLITS:
        sentences = []
        keys_from_split = [key for key in spr.keys() if spr[key]['split'] == split]
        for spr_id in keys_from_split:
            predicates = []
            doc_relations = [] # args
            doc = nlp(spr[spr_id]['sentence'])
            for predicate_key in spr[spr_id]['predicates'].keys():
                predicate_dict = spr[spr_id]['predicates'][predicate_key]
                pred_span = doc[predicate_dict['token'] : predicate_dict['token'] + 1]
                pred_item = PredicateItem(
                    text=pred_span.text,
                    start_char=pred_span.start_char,
                    end_char=pred_span.end_char,
                    roleset_id='', # no roleset ID in SPRL data
                )
                pred_relations = []
                for arg_key in predicate_dict['args'].keys():
                    arg_dict = predicate_dict['args'][arg_key]
                    arg_span = doc[arg_dict['begin_token'] : arg_dict['end_token'] + 1]
                    labels = {label: False for label in properties}
                    for prop in properties:
                        total = 0
                        na = False
                        num_annotations = len(arg_dict['properties'][prop]['annotations'])
                        for annotation in arg_dict['properties'][prop]['annotations']:
                            if annotation['applicable'] == False:
                                na = True
                                break
                            if annotation['response'] is not None:
                                total += annotation['response']
                            else:
                                num_annotations -= 1
                                continue
                        if not na:
                            avg = float(total) / float(num_annotations)
                            if avg >= 4:
                                labels[prop] = True
                        else:
                            if collapse_na_to_false:
                                labels[prop] = False
                            else:
                                labels[prop] = "n/a"
                    
                    protorole_item = RelationItem(
                        arg=SpanItem(
                            text=arg_span.text,
                            start_char=arg_span.start_char,
                            end_char=arg_span.end_char,
                        ),
                        sprl_label=labels
                    )
                    pred_relations.append(protorole_item)
                predicates.append(pred_item)
                doc_relations.append((pred_item, pred_relations))

            sentence_item = SentenceItem(
                sentence_id=spr_id,
                text=doc.text, 
                predicates=predicates, 
                relations=doc_relations
            )

            sentences.append(sentence_item.model_dump())
        
        output_filename = f'{output_path}/{dataset}.{split}{".with-na" if not collapse_na_to_false else ""}.pkl'
        with open(output_filename, 'wb') as output_file:
            pickle.dump(sentences, output_file)

@app.command()
def ontonotes_split_maker(eval_folder: str, output_split_file_path: str):
    print(f'eval folder - {eval_folder}\noutput split file path - {output_split_file_path}')

    splits = ["train", "dev", "test", "conll12-test"]
    split_docs = []
    for split in splits:
        with open(str(eval_folder) + f"/ontonotes-{split}-list.txt") as sf:
            lines = [line.strip() for line in sf.readlines()]
            split_docs.extend([(split, doc) for doc in lines])

    with open(output_split_file_path, "w") as of:
        of.write("\n".join(["\t".join(tup) for tup in split_docs]))
    return

@app.command()
def conll05(
    input_path: str,
    output_path: str
):
    '''
    Preprocess the CoNLL 2005 data into the SentenceItem format

    Args:
    - input_path: str
        The directory for the CoNLL 2005 data
    - output_path: str
        The path to save the output files
    '''
    # check that input_path exists
    if not os.path.exists(input_path):
        raise ValueError(f"Could not find {input_path}.")
    output_path = ensure_output_path_exists(output_path)

    nlp = spacy.blank("en")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    SPLITS = ["train", "dev", "test.brown", "test.wsj"]

    for split in SPLITS:
        conll_path = str(input_path) + f"/{split}-set"
        sentence_cols_lines = get_conll_sentence_lines(conll_path)
        sentences = []
        for i, conll_sentence in enumerate(sentence_cols_lines):
            sentence, rels = get_pred_relations_2005(conll_sentence)
            doc = nlp(sentence)
            id_ =  f'{split}_{i}'
            predicates = []
            relations = []
            for pred, arg_rels in rels:
                pred_span = doc[pred['token_start'] : pred['token_end'] + 1]
                pred_item = PredicateItem(
                    text=pred_span.text,
                    start_char=pred_span.start_char,
                    end_char=pred_span.end_char,
                    roleset_id=pred['sense']
                )
                predicates.append(pred_item)
                pred_relations = []
                for rel in arg_rels:
                    arg_span = doc[rel["token_start"] : rel["token_end"] + 1]
                    label = rel["label"]
                    role_item = RelationItem(
                        arg=SpanItem(
                            text=arg_span.text,
                            start_char=arg_span.start_char,
                            end_char=arg_span.end_char,
                        ),
                        srl_label=label,
                    )
                    pred_relations.append(role_item)
                relations.append((pred_item, pred_relations))
            sentence_item = SentenceItem(
                sentence_id=id_,
                text=doc.text,
                predicates=predicates,
                relations=relations
            )
            sentences.append(sentence_item.model_dump())

        output_filename = f'{output_path}/{split}.pkl'
        with open(output_filename, 'wb') as output_file:
            pickle.dump(sentences, output_file)

@app.command()
def conll12(
    input_path: str,
    output_path: str,
    split_file_path: Optional[str] = None,
    test_p: Optional[float] = 0.1
):
    '''
    Preprocess the CoNLL 2012 data into the SentenceItem format

    Args:
    - input_path: str
        The directory for the CoNLL 2012 data
    - output_path: str
        The path to save the output files
    - split_file_path: str
        The path to the file containing the splits
    - test_p: float
        The proportion of the dataset to be used for testing
    '''
    # check if input_path exists
    if not os.path.exists(input_path):
        raise ValueError(f"Could not find {input_path}.")
    output_path = ensure_output_path_exists(output_path)

    if split_file_path:
        if not os.path.exists(split_file_path):
            raise ValueError(f"Could not find {split_file_path}.")

    nlp = spacy.blank("en")
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    gold_conll_files = list(glob(str(input_path) + "/**/*.gold_conll", recursive=True))

    sentences = []
    relations = []
    doc_ids = []
    for file_path in tqdm(gold_conll_files, desc="processing srl conll12 files"):
        sentence_cols_lines = get_conll_sentence_lines(file_path)
        for conll_sentence in sentence_cols_lines:
            sentence, rels, doc_id = get_pred_relations_2012(conll_sentence)
            doc = nlp(sentence)
            predicates = []
            relations = []
            for pred, arg_rels in rels:
                if len(arg_rels) == 0:
                    # skip predicates without arguments
                    continue
                pred_span = doc[pred['token_start'] : pred['token_end'] + 1]
                pred_item = PredicateItem(
                    text=pred_span.text,
                    start_char=pred_span.start_char,
                    end_char=pred_span.end_char,
                    roleset_id=pred['sense']
                )
                predicates.append(pred_item)
                pred_relations = []
                for rel in arg_rels:
                    arg_span = doc[rel["token_start"] : rel["token_end"] + 1]
                    label = rel["label"]
                    role_item = RelationItem(
                        arg=SpanItem(
                            text=arg_span.text,
                            start_char=arg_span.start_char,
                            end_char=arg_span.end_char,
                        ),
                        srl_label=label,
                    )
                    pred_relations.append(role_item)
                relations.append((pred_item, pred_relations))
            if len(predicates) == 0:
                continue
            sentence_item = SentenceItem(
                sentence_id=doc_id,
                text=doc.text,
                predicates=predicates,
                relations=relations
            )
            sentences.append(sentence_item.model_dump())
            doc_ids.append(doc_id.split('-')[0])

    # Handle splitting    
    doc_id2docs = {
        k: list(g) for k, g in itertools.groupby(sentences, lambda x: x['sentence_id'].split('-')[0])
    }

    if split_file_path:
        with open(split_file_path) as ssf:
            lines = [line.strip().split() for line in ssf]
            split_name2doc_id = {}
            for split, doc_id in lines:
                if split not in split_name2doc_id:
                    split_name2doc_id[split] = []
                split_name2doc_id[split].append(doc_id)

        for split, doc_ids in split_name2doc_id.items():
            split_docs = [
                doc
                for d_id in doc_ids
                if d_id in doc_id2docs
                for doc in doc_id2docs[d_id]
            ]
            output_filename = f'{output_path}/{split}.pkl'
            with open(output_filename, 'wb') as output_file:
                pickle.dump(split_docs, output_file)
    else:
        train_sentences, dev_sentences = two_way_split(
            sentences=sentences, 
            test_p=test_p
            )
        train_output_filename = f'{output_path}/train.pkl'
        dev_output_filename = f'{output_path}/dev.pkl'
        with open(train_output_filename, 'wb') as train_output_file:
            pickle.dump(train_sentences, train_output_file)
        with open(dev_output_filename, 'wb') as dev_output_file:
            pickle.dump(dev_sentences, dev_output_file)

if __name__ == '__main__':
    app()