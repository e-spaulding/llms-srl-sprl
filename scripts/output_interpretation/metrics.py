import logging
import pickle
import json
import os
import re

from typing import List, Union
from collections import defaultdict
from typer import Typer
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from ..data2prompts.items import PredictedSentenceItem, SentenceItem, PredicateItem, SpanItem, RelationItem
from .parsing import merge_gpt_output, get_items_from_dump
from ..util.macros import *

logger = logging.getLogger(__name__)
app = Typer(pretty_exceptions_show_locals=False)


def precision_recall_f1_support(y_true, y_pred):
    '''Wrapper for sklearn.metrics.precision_recall_fscore_support'''
    count = len(y_true)
    prfs = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return prfs[0], prfs[1], prfs[2], count

def fuzzy_role_match(predicted: str, gold: str) -> bool:
    if predicted == gold:
        return True
    if gold == 'ARG0':
        if predicted.lower() == 'agent':
            return True
        if 'agent' in predicted.lower():
            return True
    if gold == 'ARG1':
        if predicted.lower() == 'patient':
            return True
        if 'patient' in predicted.lower():
            return True
    return False

def get_hits(gold_items, predicted_items):
    '''Get the number of hits (true positives) in the predicted items'''
    return sum([1 for item in predicted_items if item in gold_items])

def try_to_find_subset_keys(path):
    subset_keys = None
    for file in os.listdir(path):
        if file == 'subset_keys.txt':
            with open(f'{path}/{file}', 'r') as f:
                subset_keys = f.read().split('\n')
    return subset_keys

def try_to_find_blacklist(path):
    blacklist = None
    try:
        with open(f'{path}/blacklist.txt', 'r') as f:
            blacklist = f.read().split('\n')
    except:
        pass
    return blacklist

def get_spr_one_hots_from_TFN_prompts(path: str, blacklist: List[str] = None, na_ids_to_skip: List[str] = None, sprl_subset: List[str] = SPR1_LABELS):
    f = open(path)
    results_dict = json.load(f)

    one_hots = {
        'gold_per_property': {
            spr_label: [] for spr_label in sprl_subset
        },
        'predicted_per_property': {
            spr_label: [] for spr_label in sprl_subset
        },
        'gold_all_properties': [],
        'predicted_all_properties': []
    }

    for k in results_dict.keys():
        if k in ['config', 'batch_progress', 'total_elapsed_time']:
            continue
        if na_ids_to_skip and k in na_ids_to_skip:
            continue

        id_splits = k.split('|')
        property_name = id_splits[-1]

        if property_name not in sprl_subset:
            continue

        gold = results_dict[k]['prompt']['gold']
        clean_answer = results_dict[k]['clean_output']
        match = re.search('TRUE|FALSE', clean_answer.upper())
        if match and match.group() == 'TRUE':
            predicted = True
        elif match and match.group() == 'FALSE':
            predicted = False
        elif match and match.group() == 'N/A':
            if blacklist:
                blacklist.append(k)
            predicted = not gold
            # raise NotImplementedError('N/A not yet implemented')
        else:
            if blacklist:
                blacklist.append(k)
            predicted = not gold
            # raise ValueError(f'Unknown output for {k} in {path}: {clean_answer}')
        
        if gold:
            one_hots['gold_per_property'][property_name].append(1)
            one_hots['gold_all_properties'].append(1)
        else:
            one_hots['gold_per_property'][property_name].append(0)
            one_hots['gold_all_properties'].append(0)

        if predicted:
            one_hots['predicted_per_property'][property_name].append(1)
            one_hots['predicted_all_properties'].append(1)
        else:
            one_hots['predicted_per_property'][property_name].append(0)
            one_hots['predicted_all_properties'].append(0)

    return one_hots

def get_spr_one_hots_from_sentences(items: dict, pstring: str,
        subset_keys: list = None, skip_na: bool = False)-> dict:
    '''Get the one-hot encoded SPR labels from the gold and predicted sentences'''
    if pstring == 'intersection':
        sprl_subset = LABELS_INTERSECTION
    elif pstring == 'all':
        sprl_subset = list(items.values())[0]['gold']['item'].relations[0][1][0].sprl_label.keys()
    else:
        sprl_subset = pstring.split('-')
    predictions_no_nones = [item for item in items.values() if item['predicted'][f'sprl-{pstring}']['item'] is not None]
    logger.info(f'Num predicted sentences: {len(predictions_no_nones)}')
    # Set up dict for labeled one-hots
    one_hots = {
        'correctly_retrieved_spans': {'gold_per_property': {
                spr_label: [] for spr_label in sprl_subset},
            'predicted_per_property': {spr_label: [] for spr_label in sprl_subset},
            'gold_all_properties': [],
            'predicted_all_properties': []},
        'all_spans': {'gold_per_property': {
                spr_label: [] for spr_label in sprl_subset},
            'predicted_per_property': {spr_label: [] for spr_label in sprl_subset},
            'gold_all_properties': [],
            'predicted_all_properties': []}
    }
    for sentence_id in tqdm(items.keys(), desc='Tallying SPR output', total=len(items.keys())):
        gold_sentence = items[sentence_id]['gold']['item']
        predicted_sentence = items[sentence_id]['predicted'][f'sprl-{pstring}']['item']
        if not predicted_sentence:
            # Add one hots for this sentence
            for pred, rels in gold_sentence.relations:
                for rel in rels:
                    query = f'{gold_sentence.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}' # check if this is a pred-arg pair to be scored
                    if subset_keys is not None:
                        if query not in subset_keys:
                            continue
                    for spr_key in sprl_subset:
                        if skip_na and rel.sprl_label[spr_key] == 'n/a':
                            continue
                        one_hots['all_spans']['gold_per_property'][spr_key].append(rel.sprl_label[spr_key])
                        one_hots['all_spans']['predicted_per_property'][spr_key].append(0)
                        one_hots['all_spans']['gold_all_properties'].append(rel.sprl_label[spr_key])
                        one_hots['all_spans']['predicted_all_properties'].append(0)
            continue
        else: # There is a predicted sentence
            gold_sentence.set_compare_text(False)
            predicted_sentence.set_compare_text(False)
            gold_sentence.set_predicate_compare_roleset_ids(False)
            predicted_sentence.set_predicate_compare_roleset_ids(False)
            for gold_pred, gold_rels in gold_sentence.relations:
                # Find the corresponding predicted predicate
                predicted_pred = next((pred for pred in predicted_sentence.predicates if pred == gold_pred), None) # There should only be one?
                for gold_rel in gold_rels:
                    query = f'{gold_sentence.sentence_id}|{gold_pred.start_char}|{gold_pred.end_char}|{gold_rel.arg.start_char}|{gold_rel.arg.end_char}' # check if this is a pred-arg pair to be scored
                    if subset_keys is not None:
                        if query not in subset_keys:
                            continue
                    if not predicted_pred:
                        # Add one hots for this relation
                        for spr_key in sprl_subset:
                            if skip_na and gold_rel.sprl_label[spr_key] == 'n/a':
                                continue
                            one_hots['all_spans']['gold_per_property'][spr_key].append(gold_rel.sprl_label[spr_key])
                            one_hots['all_spans']['predicted_per_property'][spr_key].append(0)
                            one_hots['all_spans']['gold_all_properties'].append(gold_rel.sprl_label[spr_key])
                            one_hots['all_spans']['predicted_all_properties'].append(0)
                        continue
                    # Find the corresponding predicted relation
                    predicted_rel = next((rel for pred, rels in predicted_sentence.relations for rel in rels if predicted_pred == pred and rel.arg == gold_rel.arg), None)
                    if not predicted_rel or not predicted_rel.sprl_label:
                        # Add one hots for this relation
                        for spr_key in sprl_subset:
                            if skip_na and gold_rel.sprl_label[spr_key] == 'n/a':
                                continue
                            one_hots['all_spans']['gold_per_property'][spr_key].append(gold_rel.sprl_label[spr_key])
                            one_hots['all_spans']['predicted_per_property'][spr_key].append(0)
                            one_hots['all_spans']['gold_all_properties'].append(gold_rel.sprl_label[spr_key])
                            one_hots['all_spans']['predicted_all_properties'].append(0)
                        continue
                    # Add one hots for this relation
                    for spr_key in sprl_subset:
                        if skip_na and gold_rel.sprl_label[spr_key] == 'n/a':
                            continue
                        if skip_na and (predicted_rel.sprl_label[spr_key] != True and predicted_rel.sprl_label[spr_key] != False):
                            # Add one hots for this relation, just record the predicted as the opposite of the gold
                            opposite = 0 if gold_rel.sprl_label[spr_key] == 1 else 1
                            one_hots['correctly_retrieved_spans']['gold_per_property'][spr_key].append(gold_rel.sprl_label[spr_key])
                            one_hots['correctly_retrieved_spans']['predicted_per_property'][spr_key].append(opposite)
                            one_hots['correctly_retrieved_spans']['gold_all_properties'].append(gold_rel.sprl_label[spr_key])
                            one_hots['correctly_retrieved_spans']['predicted_all_properties'].append(opposite)
                            one_hots['all_spans']['gold_per_property'][spr_key].append(gold_rel.sprl_label[spr_key])
                            one_hots['all_spans']['predicted_per_property'][spr_key].append(opposite)
                            one_hots['all_spans']['gold_all_properties'].append(gold_rel.sprl_label[spr_key])
                            one_hots['all_spans']['predicted_all_properties'].append(opposite)
                            continue
                        one_hots['correctly_retrieved_spans']['gold_per_property'][spr_key].append(gold_rel.sprl_label[spr_key])
                        one_hots['correctly_retrieved_spans']['predicted_per_property'][spr_key].append(predicted_rel.sprl_label[spr_key])
                        one_hots['correctly_retrieved_spans']['gold_all_properties'].append(gold_rel.sprl_label[spr_key])
                        one_hots['correctly_retrieved_spans']['predicted_all_properties'].append(predicted_rel.sprl_label[spr_key])
                        one_hots['all_spans']['gold_per_property'][spr_key].append(gold_rel.sprl_label[spr_key])
                        one_hots['all_spans']['predicted_per_property'][spr_key].append(predicted_rel.sprl_label[spr_key])
                        one_hots['all_spans']['gold_all_properties'].append(gold_rel.sprl_label[spr_key])
                        one_hots['all_spans']['predicted_all_properties'].append(predicted_rel.sprl_label[spr_key])
        gold_sentence.reset_compare_to_default()
        predicted_sentence.reset_compare_to_default()

    return one_hots

def get_spr_error_sets_from_sentences(items: dict, pstring: str, skip_na: bool = False):
    if pstring == 'intersection':
        sprl_subset = LABELS_INTERSECTION
    elif pstring == 'all':
        sprl_subset = list(items.values())[0]['gold']['item'].relations[0][1][0].sprl_label.keys()
    else:
        sprl_subset = pstring.split('-')

    error_sets = {}
    error_sets['missing_predictions'] = set()

    for spr_label in sprl_subset:
        error_sets[f'gold_positive_{spr_label}'] = set()
        error_sets[f'gold_negative_{spr_label}'] = set()
        error_sets[f'predicted_positive_{spr_label}'] = set()
        error_sets[f'predicted_negative_{spr_label}'] = set()
        if skip_na:
            error_sets[f'gold_na_{spr_label}'] = set()
            error_sets[f'predicted_na_{spr_label}'] = set()

    for sentence_id in tqdm(items.keys(), desc=f'Generating SPR ({pstring}) error sets', total=len(items.keys())):
        gold_sentence = items[sentence_id]['gold']['item']
        predicted_sentence = items[sentence_id]['predicted'][f'sprl-{pstring}']['item']
        if not predicted_sentence:
            # Add errors to error set
            for pred, rels in gold_sentence.relations:
                for rel in rels:
                    query = f'{sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}'
                    for spr_key in sprl_subset:
                        if rel.sprl_label[spr_key] == True:
                            error_sets[f'gold_positive_{spr_key}'].add(query)
                            error_sets['missing_predictions'].add(query)
                        else:
                            if skip_na and rel.sprl_label[spr_key] == 'n/a':
                                error_sets[f'gold_na_{spr_key}'].add(query)
                                error_sets['missing_predictions'].add(query)
                                continue
                            error_sets[f'gold_negative_{spr_key}'].add(query)
                            error_sets['missing_predictions'].add(query)
            continue
        else: # There is a predicted sentence
            gold_sentence.set_compare_text(False)
            predicted_sentence.set_compare_text(False)
            gold_sentence.set_predicate_compare_roleset_ids(False)
            predicted_sentence.set_predicate_compare_roleset_ids(False)
            for gold_pred, gold_rels in gold_sentence.relations:
                # Find the corresponding predicted predicate
                predicted_pred = next((pred for pred in predicted_sentence.predicates if pred == gold_pred), None) # There should only be one
                for gold_rel in gold_rels:
                    query = f'{sentence_id}|{gold_pred.start_char}|{gold_pred.end_char}|{gold_rel.arg.start_char}|{gold_rel.arg.end_char}'
                    if not predicted_pred:
                        # Add one hots for this relation
                        for spr_key in sprl_subset:
                            if gold_rel.sprl_label[spr_key] == True:
                                error_sets[f'gold_positive_{spr_key}'].add(query)
                                error_sets['missing_predictions'].add(query)
                            else:
                                if skip_na and gold_rel.sprl_label[spr_key] == 'n/a':
                                    error_sets[f'gold_na_{spr_key}'].add(query)
                                    error_sets['missing_predictions'].add(query)
                                    continue
                                error_sets[f'gold_negative_{spr_key}'].add(query)
                                error_sets['missing_predictions'].add(query)
                        continue
                    # Find the corresponding predicted relation
                    predicted_rel = next((rel for pred, rels in predicted_sentence.relations for rel in rels if predicted_pred == pred and rel.arg == gold_rel.arg), None)
                    if not predicted_rel or not predicted_rel.sprl_label:
                        # Add one hots for this relation
                        for spr_key in sprl_subset:
                            if gold_rel.sprl_label[spr_key] == True:
                                error_sets[f'gold_positive_{spr_key}'].add(query)
                                error_sets['missing_predictions'].add(query)
                            else:
                                if skip_na and gold_rel.sprl_label[spr_key] == 'n/a':
                                    error_sets[f'gold_na_{spr_key}'].add(query)
                                    error_sets['missing_predictions'].add(query)
                                    continue
                                error_sets[f'gold_negative_{spr_key}'].add(query)
                                error_sets['missing_predictions'].add(query)
                        continue
                    # Add one hots for this relation
                    for spr_key in sprl_subset:
                        if gold_rel.sprl_label[spr_key] == True:
                            error_sets[f'gold_positive_{spr_key}'].add(query)
                        elif gold_rel.sprl_label[spr_key] == 'n/a':
                            error_sets[f'gold_na_{spr_key}'].add(query)
                        else:
                            error_sets[f'gold_negative_{spr_key}'].add(query)
                        if predicted_rel.sprl_label[spr_key] == True:
                            error_sets[f'predicted_positive_{spr_key}'].add(query)
                        elif predicted_rel.sprl_label[spr_key] == 'n/a':
                            error_sets[f'predicted_na_{spr_key}'].add(query)
                        else:
                            error_sets[f'predicted_negative_{spr_key}'].add(query)
        gold_sentence.reset_compare_to_default()
        predicted_sentence.reset_compare_to_default()

    # Generate intersections
    for spr_label in sprl_subset:
        error_sets[f'gold_positive_{spr_label}_intersection_predicted_positive_{spr_label}'] = error_sets[f'gold_positive_{spr_label}'].intersection(error_sets[f'predicted_positive_{spr_label}'])
        error_sets[f'gold_negative_{spr_label}_intersection_predicted_negative_{spr_label}'] = error_sets[f'gold_negative_{spr_label}'].intersection(error_sets[f'predicted_negative_{spr_label}'])
        error_sets[f'gold_positive_{spr_label}_intersection_predicted_negative_{spr_label}'] = error_sets[f'gold_positive_{spr_label}'].intersection(error_sets[f'predicted_negative_{spr_label}'])
        error_sets[f'gold_negative_{spr_label}_intersection_predicted_positive_{spr_label}'] = error_sets[f'gold_negative_{spr_label}'].intersection(error_sets[f'predicted_positive_{spr_label}'])
        if skip_na:
            error_sets[f'gold_na_{spr_label}_intersection_predicted_na_{spr_label}'] = error_sets[f'gold_na_{spr_label}'].intersection(error_sets[f'predicted_na_{spr_label}'])
            error_sets[f'gold_na_{spr_label}_intersection_predicted_positive_{spr_label}'] = error_sets[f'gold_na_{spr_label}'].intersection(error_sets[f'predicted_positive_{spr_label}'])
            error_sets[f'gold_na_{spr_label}_intersection_predicted_negative_{spr_label}'] = error_sets[f'gold_na_{spr_label}'].intersection(error_sets[f'predicted_negative_{spr_label}'])
            error_sets[f'gold_positive_{spr_label}_intersection_predicted_na_{spr_label}'] = error_sets[f'gold_positive_{spr_label}'].intersection(error_sets[f'predicted_na_{spr_label}'])
            error_sets[f'gold_negative_{spr_label}_intersection_predicted_na_{spr_label}'] = error_sets[f'gold_negative_{spr_label}'].intersection(error_sets[f'predicted_na_{spr_label}'])
    
    return error_sets

def write_error_sets_to_file(error_sets: dict,component: str,
        output_path: str, str_to_append: str = ''):
    # Convert sets to lists for JSON serialization
    for key in error_sets.keys():
        if isinstance(error_sets[key], dict):
            for subkey in error_sets[key].keys():
                error_sets[key][subkey] = list(error_sets[key][subkey])
        else:
            error_sets[key] = list(error_sets[key])

    with open(f'{output_path}/{str_to_append}{component}-error-sets.json', 'w') as f:
        json.dump(error_sets, f)

@app.command()
def score_srl(path: str, disclude_keys_on_blacklist: bool = False, fuzzy_roles: bool = False):
    '''Scores srl (oracle preds and args) from a parsed-responses.pkl dump'''
    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    str_to_append = ''

    if disclude_keys_on_blacklist:
        blacklist = try_to_find_blacklist(path)
        str_to_append += 'subset-'
        if blacklist is None:
            logger.warning('No blacklist found but scoring anyway. Blacklisted items will be scored.')

    # Get items from dump
    items, components_parsed = get_items_from_dump(path)
    scores = {}

    for component in components_parsed:
        hits = 0
        num_gold_arguments = 0
        spans_not_correctly_retrieved = 0
        error_sets = defaultdict(set)
        error_sets_correctly_retrieved = defaultdict(set)
        for i in [0, 1, 2, 3, 4, 5]:
            error_sets[f'gold_ARG{i}'] = set()
            error_sets[f'predicted_ARG{i}'] = set()
            error_sets_correctly_retrieved[f'gold_ARG{i}'] = set()
            error_sets_correctly_retrieved[f'predicted_ARG{i}'] = set()
        for sentence_id in tqdm(items.keys(), desc=f'Scoring {component} for SRL...', total=len(items.keys())):
            gold_item = items[sentence_id]['gold']['item']
            predicted_item = items[sentence_id]['predicted'][component]['item']
            if not predicted_item:
                for pred, rels in gold_item.relations:
                    for rel in rels:
                        query = f'{gold_item.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}' # check if this is a pred-arg pair to be scored
                        if disclude_keys_on_blacklist:
                            if query in blacklist:
                                continue
                        num_gold_arguments += 1
                        spans_not_correctly_retrieved += 1
                        error_sets[f'gold_{rel.srl_label}'].add(query)
                continue
            gold_item.set_predicate_compare_roleset_ids(True)
            gold_item.set_relation_compare_sprl_labels(False)
            gold_item.set_compare_text(False)
            predicted_item.set_predicate_compare_roleset_ids(True)
            predicted_item.set_relation_compare_sprl_labels(False)
            predicted_item.set_compare_text(False)
            for pred, rels in gold_item.relations:
                # Find corresponding predicted predicate
                predicted_pred = next((p for p in predicted_item.predicates if p == pred), None)
                if predicted_pred:
                    predicted_rels = [rel for ppred, rrels in predicted_item.relations for rel in rrels if ppred == predicted_pred]
                    for ppred, predicted_rels in predicted_item.relations:
                        if ppred == predicted_pred:
                            predicted_spans = [prel.arg for prel in predicted_rels]
                            # Find spans not correctly retrieved
                            for rel in rels:
                                query = f'{gold_item.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}' # check if this is a pred-arg pair to be scored
                                if disclude_keys_on_blacklist:
                                    if query in blacklist:
                                        continue
                                num_gold_arguments += 1
                                error_sets[f'gold_{rel.srl_label}'].add(query)
                                if rel.arg not in predicted_spans:
                                    spans_not_correctly_retrieved += 1
                                else:
                                    error_sets_correctly_retrieved[f'gold_{rel.srl_label}'].add(query)
                                    # Find corresponding predicted relation
                                    predicted_rel = next((prel for prel in predicted_rels if prel.arg == rel.arg), None)
                                    if predicted_rel:
                                        if fuzzy_roles:
                                            if fuzzy_role_match(predicted_rel.srl_label, rel.srl_label):
                                                predicted_rel.srl_label = rel.srl_label
                                                hits += 1
                                            error_sets[f'predicted_{predicted_rel.srl_label}'].add(query)
                                        else:
                                            if predicted_rel.srl_label == rel.srl_label:
                                                hits += 1
                                            error_sets[f'predicted_{predicted_rel.srl_label}'].add(query)
                                    else:
                                        raise ValueError('Predicted relation is None but that should be impossible')
                            break
                else:
                    # miss type: predicate miss
                    for rel in rels:
                        query = f'{gold_item.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}' # check if this is a pred-arg pair to be scored
                        if disclude_keys_on_blacklist:
                            if query in blacklist:
                                continue
                        num_gold_arguments += 1
                        spans_not_correctly_retrieved += 1
                        error_sets[f'gold_{rel.srl_label}'].add(query)

        # Get misses/FNs for each argument
        misses = num_gold_arguments - hits
        # Misses for spans not correctly retrieved
        misses_minus_not_correctly_retrieved = misses - spans_not_correctly_retrieved
        correctly_retrieved_spans = num_gold_arguments - spans_not_correctly_retrieved

        for n in [0, 1, 2, 3, 4, 5]:
            for m in [0, 1, 2, 3, 4, 5]:
                error_sets[f'gold_ARG{n}_intersection_predicted_ARG{m}'] = error_sets[f'gold_ARG{n}'].intersection(error_sets[f'predicted_ARG{m}'])
                error_sets_correctly_retrieved[f'gold_ARG{n}_intersection_predicted_ARG{m}'] = error_sets_correctly_retrieved[f'gold_ARG{n}'].intersection(error_sets_correctly_retrieved[f'predicted_ARG{m}'])
            other_prediction_sets = []
            for set_ in error_sets.keys():
                if set_.startswith('predicted_') and set_ not in [f'predicted_ARG{i}' for i in [0, 1, 2, 3, 4, 5]]:
                    other_prediction_sets.append(set_)
            for set_ in other_prediction_sets:
                error_sets[f'gold_ARG{n}_intersection_{set_}'] = error_sets[f'gold_ARG{n}'].intersection(error_sets[set_])
                error_sets_correctly_retrieved[f'gold_ARG{n}_intersection_{set_}'] = error_sets_correctly_retrieved[f'gold_ARG{n}'].intersection(error_sets_correctly_retrieved[set_])

        error_set_stats = {'all_spans': {}, 'correctly_retrieved_spans': {}}

        for n in [0, 1, 2, 3, 4, 5]:
            all_intersections = {}
            for m in [0, 1, 2, 3, 4, 5]:
                all_intersections[f'intersection_predicted_ARG{m}'] = len(error_sets[f'gold_ARG{n}_intersection_predicted_ARG{m}'])
            for set_ in other_prediction_sets:
                all_intersections[f'intersection_{set_}'] = len(error_sets[f'gold_ARG{n}_intersection_{set_}'])
            
            error_set_stats['all_spans'][f'gold_ARG{n}'] = {
                'num': len(error_sets[f'gold_ARG{n}']),
                '%_correct': (len(error_sets[f'gold_ARG{n}_intersection_predicted_ARG{n}']) / len(error_sets[f'gold_ARG{n}']) if len(error_sets[f'gold_ARG{n}']) != 0 else 0),
                **all_intersections
            }
        scores[component] = {
            'all_spans': {
                'tp': hits,
                'fn': misses,
                'accuracy': hits / num_gold_arguments,
                'error_stats': error_set_stats['all_spans']
            },
            'correctly_retrieved_spans': {
                'tp': hits,
                'fn': misses_minus_not_correctly_retrieved,
                'accuracy': (hits / correctly_retrieved_spans) if correctly_retrieved_spans != 0 else 0,
                'error_stats': error_set_stats['correctly_retrieved_spans']
            }
        }
        write_error_sets_to_file(error_sets, component, path, str_to_append)
    # write scores to json
    with open(f'{path}/{str_to_append}scores.json', 'w') as f:
        json.dump(scores, f, indent=4)
    return

@app.command()
def score_sprl_annotate(path: str, spr2: bool = False, na_ids_to_skip: str = None):
    '''Scores sprl from sprl-annotate responses'''
    metrics = [precision_recall_f1_support]

    # Convert spr labels to array-like of shape (n_samples,)
    blacklist = []
    na_ids = None
    if na_ids_to_skip:
        with open(na_ids_to_skip, 'r') as f:
            na_ids = f.read().split('\n')
    if spr2:
        one_hots = get_spr_one_hots_from_TFN_prompts(path=path, blacklist=blacklist, na_ids_to_skip=na_ids, sprl_subset=SPR2_LABELS)
    else:
        one_hots = get_spr_one_hots_from_TFN_prompts(path=path, blacklist=blacklist,  na_ids_to_skip=na_ids)

    all_labels = list(one_hots['gold_per_property'].keys())

    # Prep score dict
    scores = {}
    scores['sprl'] = {'per_property': {spr_label: {} for spr_label in all_labels}, 'all_properties': {}}

    # Compute metrics
    for metric_function in metrics:
        for spr_label in all_labels:
            scores['sprl']['per_property'][spr_label][metric_function.__name__] = metric_function(
                one_hots['gold_per_property'][spr_label], 
                one_hots['predicted_per_property'][spr_label]
            )
        
        # "Micro" average according to SPRL parlance - i.e., all properties across all samples are averaged together
        scores['sprl']['all_properties'][metric_function.__name__] = metric_function(
            one_hots['gold_all_properties'], 
            one_hots['predicted_all_properties']
        )

        # "Macro" average according to SPRL parlance - i.e., properties are averaged first and then average is taken across all samples
        if isinstance(scores['sprl']['all_properties'][metric_function.__name__], float):
            avg = sum([scores['sprl']['per_property'][spr_label][metric_function.__name__] for spr_label in all_labels]) / len(all_labels)
            if 'macro_averages' in scores['sprl']['per_property'].keys():
                scores['sprl']['per_property']['macro_averages'][metric_function.__name__] = avg
            else:
                scores['sprl']['per_property']['macro_averages'] = {
                    metric_function.__name__: avg
                }
        else:
            if 'macro_averages' in scores['sprl']['per_property'].keys():
                scores['sprl']['per_property']['macro_averages'][metric_function.__name__] = []
            else:
                scores['sprl']['per_property']['macro_averages'] = {
                    metric_function.__name__: []
                }

            # Num of scores
            total = len(scores['sprl']['all_properties'][metric_function.__name__])

            for i in range(total):
                scores_to_be_averaged = [scores['sprl']['per_property'][spr_label][metric_function.__name__][i] for spr_label in all_labels]
                if None in scores_to_be_averaged:
                    scores['sprl']['per_property']['macro_averages'][metric_function.__name__].append(None)
                else:
                    scores['sprl']['per_property']['macro_averages'][metric_function.__name__].append(
                        sum([scores['sprl']['per_property'][spr_label][metric_function.__name__][i] for spr_label in all_labels]) / len(all_labels))
                    
    # now do created and destroyed but don't add to average
    for metric_function in metrics:
        for spr_label in ['created', 'destroyed']:
            if spr_label not in scores['sprl']['per_property'].keys():
                scores['sprl']['per_property'][spr_label] = {}
            scores['sprl']['per_property'][spr_label][metric_function.__name__] = metric_function(
                one_hots['created_and_destroyed']['gold'][spr_label], one_hots['created_and_destroyed']['predicted'][spr_label])
            
    # Save scores to json
    # Find the dir to save the scores
    this_dir = os.path.dirname(os.path.realpath(path))
    if na_ids_to_skip:
        output_path = os.path.join(this_dir, 'scores-skip-na.json')
    else:
        output_path = os.path.join(this_dir, 'scores.json')
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=4)

@app.command()
def score_pipeline(path: str, na_ids_to_skip: str = None, subset: bool = False):
    '''Try all metrics on the parsed-responses.pkl file in `path`'''

    metrics = []
    metrics.append(precision_recall_f1_support)
    scores = defaultdict(dict)

    if subset:
        subset_keys = try_to_find_subset_keys(path)
        str_to_append = 'subset-'
    else:
        subset_keys = None
        str_to_append = ''
    na_ids = None
    if na_ids_to_skip:
        with open(na_ids_to_skip, 'r') as f:
            na_ids = f.read().split('\n')

    # Get items from dump
    items, components_parsed = get_items_from_dump(path)

    # Try each component
    for component in components_parsed:
        # Get the gold and predicted labels
        if component == 'predicate':
            hits = 0
            num_gold_predicates = 0
            for sentence_id in tqdm(items.keys(), desc='Scoring predicate (span) output', total=len(items.keys())):
                gold_item = items[sentence_id]['gold']['item']
                predicted_item = items[sentence_id]['predicted'][component]['item']
                num_gold_predicates += len([pred for pred in gold_item.predicates])
                if not predicted_item:
                    continue
                # Compute hits/TPs for each predicate not including the roleset id
                gold_predicates = [pred for pred in gold_item.predicates]
                predicted_predicates = [pred for pred in predicted_item.predicates]
                hits += get_hits(gold_predicates, predicted_predicates)
            # Get misses/FNs for each predicate
            misses = num_gold_predicates - hits
            scores['predicates'] = {'tp': hits, 'fn': misses, 'accuracy': hits / num_gold_predicates}
        elif component == 'roleset_id':
            hits = 0
            num_gold_predicates = 0
            rolesets_missed_because_preds_not_correctly_retrieved = 0
            for sentence_id in tqdm(items.keys(), desc='Scoring predicate (roleset id) output', total=len(items.keys())):
                gold_item = items[sentence_id]['gold']['item']
                num_gold_predicates += len([pred for pred in gold_item.predicates])
                gold_item.set_predicate_compare_roleset_ids(True)
                predicted_item = items[sentence_id]['predicted'][component]['item']
                if not predicted_item:
                    gold_item.reset_compare_to_default()
                    rolesets_missed_because_preds_not_correctly_retrieved += len([pred for pred in gold_item.predicates])
                    continue
                predicted_item.set_predicate_compare_roleset_ids(True)
                # Compute hits/TPs for each predicate including the roleset id
                gold_predicates = [pred for pred in gold_item.predicates]
                predicted_predicates = [pred for pred in predicted_item.predicates]
                hits += get_hits(gold_predicates, predicted_predicates)
                gold_item.reset_compare_to_default()
                predicted_item.reset_compare_to_default()
                # Find rolesets missed because predicates were not correctly retrieved
                for gold_pred in gold_predicates:
                    if gold_pred.roleset_id:
                        predicted_pred = next((pred for pred in predicted_item.predicates if pred == gold_pred), None)
                        if not predicted_pred:
                            rolesets_missed_because_preds_not_correctly_retrieved += 1

            # Get misses/FNs for each roleset_id
            misses = num_gold_predicates - hits
            # Misses for rolesets not correctly retrieved
            misses_minus_not_correctly_retrieved = misses - rolesets_missed_because_preds_not_correctly_retrieved
            correctly_retrieved_rolesets = num_gold_predicates - rolesets_missed_because_preds_not_correctly_retrieved

            scores['predicates_with_roleset_ids'] = {
                'all_predicates': {'tp': hits, 'fn': misses,
                    'accuracy': hits / num_gold_predicates},
                'correctly_retrieved_predicates': {'tp': hits, 'fn': misses_minus_not_correctly_retrieved,
                    'accuracy': hits / correctly_retrieved_rolesets}
            }
        elif component == 'argument':
            # Compute hits/TPs for each argument
            hits = 0
            num_gold_argument_spans = 0
            args_missed_because_preds_not_correctly_retrieved = 0
            for sentence_id in tqdm(items.keys(), desc='Scoring argument (span) output', total=len(items.keys())):
                # Find corresponding predicted sentences
                gold_item = items[sentence_id]['gold']['item']
                predicted_item = items[sentence_id]['predicted'][component]['item']
                if not predicted_item:
                    num_gold_argument_spans += len([rel for pred, rels in gold_item.relations for rel in rels])
                    args_missed_because_preds_not_correctly_retrieved += len([rel for pred, rels in gold_item.relations for rel in rels])
                    continue
                for pred, rels in gold_item.relations:
                    num_gold_argument_spans += len(rels)
                    # Find corresponding predicted predicate
                    predicted_pred = next((p for p in predicted_item.predicates if p == pred), None)
                    if predicted_pred:
                        gold_spans = [rel.arg for rel in rels]
                        predicted_spans = [rel.arg for ppred, rrels in predicted_item.relations for rel in rrels if ppred == predicted_pred]
                        hits += get_hits(gold_spans, predicted_spans)
                    else:
                        # miss type: predicate miss
                        args_missed_because_preds_not_correctly_retrieved += len(rels)

            # Get misses/FNs for each argument
            misses = num_gold_argument_spans - hits
            # Misses for spans not correctly retrieved
            misses_minus_not_correctly_retrieved = misses - args_missed_because_preds_not_correctly_retrieved
            correctly_retrieved_spans = num_gold_argument_spans - args_missed_because_preds_not_correctly_retrieved

            scores['arguments'] = {
                'all_predicates': {'tp': hits, 'fn': misses,
                    'accuracy': hits / num_gold_argument_spans},
                'correctly_retrieved_predicates': {'tp': hits, 'fn': misses_minus_not_correctly_retrieved,
                    'accuracy': hits / correctly_retrieved_spans}
            }
        elif component.startswith('sprl'):
            # see if component includes pstring info
            component_parts = component.split('-')
            if len(component_parts) == 1:
                raise ValueError('SPRL component must include pstring info')
            elif len(component_parts) == 2:
                pstring = component_parts[1]
            else:
                pstring = '-'.join(component_parts[1:])

            sprl_error_sets = get_spr_error_sets_from_sentences(items=items, pstring=pstring)
            write_error_sets_to_file(error_sets=sprl_error_sets, component=component, output_path=path, skip_na=na_ids is not None)
            # Convert spr labels to array-like of shape (n_samples,)
            one_hots = get_spr_one_hots_from_sentences(items=items, pstring=pstring, subset_keys=subset_keys, skip_na=na_ids is not None)
            correctly_retrieved_spans = one_hots['correctly_retrieved_spans']
            all_spans = one_hots['all_spans']
            all_labels = list(correctly_retrieved_spans['gold_per_property'].keys())

            scores['sprl']['correctly_retrieved_spans'] = {'per_property': {spr_label: {} for spr_label in all_labels},
                'all_properties': {}}
            scores['sprl']['all_spans'] = {'per_property': {spr_label: {} for spr_label in all_labels},
                'all_properties': {}}

            # Compute metrics
            for metric_function in metrics:
                for spr_label in all_labels:
                    scores['sprl']['correctly_retrieved_spans']['per_property'][spr_label][metric_function.__name__] = metric_function(
                        correctly_retrieved_spans['gold_per_property'][spr_label], 
                        correctly_retrieved_spans['predicted_per_property'][spr_label])
                    scores['sprl']['all_spans']['per_property'][spr_label][metric_function.__name__] = metric_function(
                        all_spans['gold_per_property'][spr_label], 
                        all_spans['predicted_per_property'][spr_label])
                
                # "Micro" average according to SPRL parlance - i.e., all properties across all samples are averaged together
                scores['sprl']['correctly_retrieved_spans']['all_properties'][metric_function.__name__] = metric_function(
                    correctly_retrieved_spans['gold_all_properties'], 
                    correctly_retrieved_spans['predicted_all_properties'])
                scores['sprl']['all_spans']['all_properties'][metric_function.__name__] = metric_function(
                    all_spans['gold_all_properties'], 
                    all_spans['predicted_all_properties'])

                # "Macro" average according to SPRL parlance - i.e., properties are averaged first and then average is taken across all samples
                if isinstance(scores['sprl']['correctly_retrieved_spans']['all_properties'][metric_function.__name__], float):
                    avg = sum([scores['sprl']['correctly_retrieved_spans']['per_property'][spr_label][metric_function.__name__] for spr_label in all_labels]) / len(all_labels)
                    for which_spans in ['correctly_retrieved_spans', 'all_spans']:
                        if 'macro_averages' in scores['sprl'][which_spans]['per_property'].keys():
                            scores['sprl'][which_spans]['per_property']['macro_averages'][metric_function.__name__] = avg
                        else:
                            scores['sprl'][which_spans]['per_property']['macro_averages'] = {metric_function.__name__: avg}
                else:
                    for which_spans in ['correctly_retrieved_spans', 'all_spans']:
                        if 'macro_averages' in scores['sprl'][which_spans]['per_property'].keys():
                            scores['sprl'][which_spans]['per_property']['macro_averages'][metric_function.__name__] = []
                        else:
                            scores['sprl'][which_spans]['per_property']['macro_averages'] = {metric_function.__name__: []}

                    # Num of scores
                    total = len(scores['sprl']['correctly_retrieved_spans']['all_properties'][metric_function.__name__])

                    for i in range(total):
                        for which_spans in ['correctly_retrieved_spans', 'all_spans']:
                            scores_to_be_averaged = [scores['sprl'][which_spans]['per_property'][spr_label][metric_function.__name__][i] for spr_label in all_labels]
                            if None in scores_to_be_averaged:
                                scores['sprl'][which_spans]['per_property']['macro_averages'][metric_function.__name__].append(None)
                            else:
                                scores['sprl'][which_spans]['per_property']['macro_averages'][metric_function.__name__].append(
                                    sum([scores['sprl'][which_spans]['per_property'][spr_label][metric_function.__name__][i] for spr_label in all_labels]) / len(all_labels))
        elif component.startswith('srl'):
            hits = 0
            num_gold_arguments = 0
            spans_not_correctly_retrieved = 0
            error_sets = defaultdict(set)
            error_sets_correctly_retrieved = defaultdict(set)
            for i in [0, 1, 2, 3, 4, 5]:
                error_sets[f'gold_ARG{i}'] = set()
                error_sets[f'predicted_ARG{i}'] = set()
                error_sets_correctly_retrieved[f'gold_ARG{i}'] = set()
                error_sets_correctly_retrieved[f'predicted_ARG{i}'] = set()
            for sentence_id in tqdm(items.keys(), desc=f'Scoring {component} for SRL...', total=len(items.keys())):
                gold_item = items[sentence_id]['gold']['item']
                predicted_item = items[sentence_id]['predicted'][component]['item']
                if not predicted_item:
                    for pred, rels in gold_item.relations:
                        for rel in rels:
                            query = f'{gold_item.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}' # check if this is a pred-arg pair to be scored
                            if subset:
                                if query not in subset_keys:
                                    continue
                            num_gold_arguments += 1
                            spans_not_correctly_retrieved += 1
                            error_sets[f'gold_{rel.srl_label}'].add(query)
                    continue
                gold_item.set_predicate_compare_roleset_ids(True)
                gold_item.set_relation_compare_sprl_labels(False)
                gold_item.set_compare_text(False)
                predicted_item.set_predicate_compare_roleset_ids(True)
                predicted_item.set_relation_compare_sprl_labels(False)
                predicted_item.set_compare_text(False)
                for pred, rels in gold_item.relations:
                    # Find corresponding predicted predicate
                    predicted_pred = next((p for p in predicted_item.predicates if p == pred), None)
                    if predicted_pred:
                        predicted_rels = [rel for ppred, rrels in predicted_item.relations for rel in rrels if ppred == predicted_pred]
                        for ppred, predicted_rels in predicted_item.relations:
                            if ppred == predicted_pred:
                                predicted_spans = [prel.arg for prel in predicted_rels]
                                # Find spans not correctly retrieved
                                for rel in rels:
                                    query = f'{gold_item.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}' # check if this is a pred-arg pair to be scored
                                    if subset:
                                        if query not in subset_keys:
                                            continue
                                    num_gold_arguments += 1
                                    error_sets[f'gold_{rel.srl_label}'].add(query)
                                    if rel.arg not in predicted_spans:
                                        spans_not_correctly_retrieved += 1
                                    else:
                                        error_sets_correctly_retrieved[f'gold_{rel.srl_label}'].add(query)
                                        # Find corresponding predicted relation
                                        predicted_rel = next((prel for prel in predicted_rels if prel.arg == rel.arg), None)
                                        if predicted_rel:
                                            if predicted_rel.srl_label == rel.srl_label:
                                                hits += 1
                                            error_sets[f'predicted_{predicted_rel.srl_label}'].add(query)
                                break
                    else:
                        # miss type: predicate miss
                        for rel in rels:
                            query = f'{gold_item.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}' # check if this is a pred-arg pair to be scored
                            if subset:
                                if query not in subset_keys:
                                    continue
                            num_gold_arguments += 1
                            spans_not_correctly_retrieved += 1
                            error_sets[f'gold_{rel.srl_label}'].add(query)

            # Get misses/FNs for each argument
            misses = num_gold_arguments - hits
            # Misses for spans not correctly retrieved
            misses_minus_not_correctly_retrieved = misses - spans_not_correctly_retrieved
            correctly_retrieved_spans = num_gold_arguments - spans_not_correctly_retrieved

            for n in [0, 1, 2, 3, 4, 5]:
                for m in [0, 1, 2, 3, 4, 5]:
                    error_sets[f'gold_ARG{n}_intersection_predicted_ARG{m}'] = error_sets[f'gold_ARG{n}'].intersection(error_sets[f'predicted_ARG{m}'])
                    error_sets_correctly_retrieved[f'gold_ARG{n}_intersection_predicted_ARG{m}'] = error_sets_correctly_retrieved[f'gold_ARG{n}'].intersection(error_sets_correctly_retrieved[f'predicted_ARG{m}'])
                other_prediction_sets = []
                for set_ in error_sets.keys():
                    if set_.startswith('predicted_') and set_ not in [f'predicted_ARG{i}' for i in [0, 1, 2, 3, 4, 5]]:
                        other_prediction_sets.append(set_)
                for set_ in other_prediction_sets:
                    error_sets[f'gold_ARG{n}_intersection_{set_}'] = error_sets[f'gold_ARG{n}'].intersection(error_sets[set_])
                    error_sets_correctly_retrieved[f'gold_ARG{n}_intersection_{set_}'] = error_sets_correctly_retrieved[f'gold_ARG{n}'].intersection(error_sets_correctly_retrieved[set_])

            error_set_stats = {'all_spans': {}, 'correctly_retrieved_spans': {}}

            for n in [0, 1, 2, 3, 4, 5]:
                all_intersections = {}
                for m in [0, 1, 2, 3, 4, 5]:
                    all_intersections[f'intersection_predicted_ARG{m}'] = len(error_sets[f'gold_ARG{n}_intersection_predicted_ARG{m}'])
                for set_ in other_prediction_sets:
                    all_intersections[f'intersection_{set_}'] = len(error_sets[f'gold_ARG{n}_intersection_{set_}'])
                
                error_set_stats['all_spans'][f'gold_ARG{n}'] = {
                    'num': len(error_sets[f'gold_ARG{n}']),
                    '%_correct': (len(error_sets[f'gold_ARG{n}_intersection_predicted_ARG{n}']) / len(error_sets[f'gold_ARG{n}']) if len(error_sets[f'gold_ARG{n}']) != 0 else 0),
                    **all_intersections
                }
            scores[component] = {
                'all_spans': {
                    'tp': hits,
                    'fn': misses,
                    'accuracy': hits / num_gold_arguments,
                    'error_stats': error_set_stats['all_spans']
                },
                'correctly_retrieved_spans': {
                    'tp': hits,
                    'fn': misses_minus_not_correctly_retrieved,
                    'accuracy': (hits / correctly_retrieved_spans) if correctly_retrieved_spans != 0 else 0,
                    'error_stats': error_set_stats['correctly_retrieved_spans']
                }
            }
            write_error_sets_to_file(error_sets, component, path, str_to_append)

    # write scores to json
    with open(f'{path}/{str_to_append}scores.json', 'w') as f:
        json.dump(scores, f, indent=4)

if __name__ == '__main__':
    app()