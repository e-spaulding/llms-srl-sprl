import re
import json
import logging
import pickle

from typing import List

from ..data2prompts.items import PredicateItem, SpanItem, RelationItem, PredictedSentenceItem, SentenceItem
from ..data2prompts.pb_lookup import get_roleset_ids
from ..util.macros import SRL_REGEX

logger = logging.getLogger(__name__)

def target_spans(text: str, predicate: PredicateItem, relation: RelationItem = None, tags: str = '<PRED></PRED>')-> str:
    '''Output the text with predicate and relation spans targeted'''
    
    if tags == '[]':
        # Replace 'predicate' with '[predicate]'
        targeted = f"{text[:predicate.start_char]}[{text[predicate.start_char:predicate.end_char]}]{text[predicate.end_char:]}"
        predicate_offset = 2
    else:
        # Replace 'predicate' with '<PRED>predicate</PRED>'
        targeted = f"{text[:predicate.start_char]}<PRED>{text[predicate.start_char:predicate.end_char]}</PRED>{text[predicate.end_char:]}"
        predicate_offset = 13

    if relation:
        if tags == '[]':
            # Replace arg with [arg]
            arg_string = f"[{text[relation.arg.start_char:relation.arg.end_char]}]"
        else:
            # Replace arg with <ARG>arg</ARG>
            arg_string = f"<ARG>{text[relation.arg.start_char:relation.arg.end_char]}</ARG>"
        if relation.arg.start_char > predicate.end_char: # arg comes after pred in the sentence
            targeted = f"{targeted[:relation.arg.start_char+predicate_offset]}{arg_string}{targeted[relation.arg.end_char+predicate_offset:]}"
        else: # arg comes before pred in the sentence
            targeted = f"{targeted[:relation.arg.start_char]}{arg_string}{targeted[relation.arg.end_char:]}"

    return targeted

def target_all_relations(text: str, predicate: PredicateItem, relations: List[RelationItem]):
    '''Output the text with the predicate span targeted with <PRED></PRED> tags and all relation spans targeted with <ARG></ARG> tags'''
    # Sort relations by start char
    relations.sort(key=lambda x: x.arg.start_char)
    targeted = target_spans(text, predicate)
    predicate_offset = 13
    offset = 0
    for rel in relations:
        arg_string = f"<ARG>{rel.arg.text}</ARG>"
        if rel.arg.start_char > predicate.end_char: # arg comes after pred in the sentence
            targeted = f"{targeted[:rel.arg.start_char+predicate_offset+offset]}{arg_string}{targeted[rel.arg.end_char+predicate_offset+offset:]}"
        else: # arg comes before pred in the sentence
            targeted = f"{targeted[:rel.arg.start_char+offset]}{arg_string}{targeted[rel.arg.end_char+offset:]}"
        offset += 11
    return targeted

def normalize_srl_label(srl_label_candidate: str) -> str:
    '''Normalize SRL label to the standard format, e.g. ARG-0 -> ARG0'''
    # Try to match using regex
    match = re.match(SRL_REGEX, srl_label_candidate)
    if match:
        arg_number = match.group('arg_number')
        function_tag = match.group('function_tag')
        if function_tag:
            return f'ARG{arg_number}-{function_tag}'
        else:
            return f'ARG{arg_number}'
    else:
        return srl_label_candidate

def check_srl_label(srl_label_candidate: str) -> bool:
    '''Check if label complies with options given to llm'''
    return srl_label_candidate in ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5']

def try_json_at_every_char(str_to_try: str):
    '''given a string, try to parse json at every character index'''
    for i in range(len(str_to_try)):
        try:
            d = json.loads(str_to_try[:i])
            return d
        except:
            continue
    return None

def merge_predictions(base: PredictedSentenceItem, new: PredictedSentenceItem) -> PredictedSentenceItem:
    '''Merge new predictions into base predictions'''
    if new is None:
        return base.model_copy(deep=True)
    base_predictions = base.model_copy(deep=True)
    new_predictions = new.model_copy(deep=True)
    # Merge predicates
    new_predictions.set_predicate_compare_roleset_ids(False)
    base_predictions.set_predicate_compare_roleset_ids(False)
    new_predictions.set_compare_text(False)
    base_predictions.set_compare_text(False)
    if base_predictions.text != new_predictions.text:
        sentence_txt = base_predictions.text if len(base_predictions.text) > 0 else new_predictions.text
        base_predictions.text = sentence_txt
        new_predictions.text = sentence_txt
    for pred in new_predictions.predicates:
        if pred not in base_predictions.predicates:
            base_predictions.predicates.append(pred)
            base_predictions.relations.append((pred, []))
        else:
            # Now, check if the roleset ID and text is different
            for base_pred in base_predictions.predicates:
                if base_pred == pred:
                    if base_pred.roleset_id != pred.roleset_id:
                        r_id = base_pred.roleset_id if len(base_pred.roleset_id) > 0 else pred.roleset_id
                        base_pred.roleset_id = r_id
                        pred.roleset_id = r_id
                        # Now do the same thing for the predicate within the relations, if any
                        for pred_, _ in base_predictions.relations:
                            if pred == pred_:
                                # Roleset ID merging
                                r_id = pred_.roleset_id if len(pred_.roleset_id) > 0 else pred.roleset_id
                                pred_.roleset_id = pred.roleset_id
                                pred.roleset_id = pred.roleset_id
                                break
                    if base_pred.text != pred.text:
                        txt = base_pred.text if len(base_pred.text) > 0 else pred.text
                        base_pred.text = txt
                        pred.text = txt
                        # Now do the same thing for the predicate within the relations, if any
                        for pred_, _ in base_predictions.relations:
                            if pred == pred_:
                                # Text merging
                                txt = pred_.text if len(pred_.text) > 0 else pred.text
                                pred_.text = txt
                                pred.text = txt
                                break
                    break
        preds_in_relations = [pred_ for pred_, _ in base_predictions.relations]
        if pred not in preds_in_relations:
            base_predictions.relations.append((pred, []))
        else:
            # Now, check if the roleset ID is different
            for pred_, _ in base_predictions.relations:
                if pred_ == pred:
                    if pred_.roleset_id != pred.roleset_id:
                        r_id = pred_.roleset_id if len(pred_.roleset_id) > 0 else pred.roleset_id
                        pred_.roleset_id = pred.roleset_id
                    if pred_.text != pred.text:
                        txt = pred_.text if len(pred_.text) > 0 else pred.text
                        pred_.text = txt
                    break

    # Merge relations
    new_predictions.set_relation_compare_sprl_labels(False)
    base_predictions.set_relation_compare_sprl_labels(False)
    for npred, nrels in new_predictions.relations:
        for bpred, brels in base_predictions.relations:
            if npred == bpred:
                for nrel in nrels:
                    found_nrel_in_base = False

                    # First check if there is a matching arg to be updated in the base item
                    for brel in brels:
                        # check matching args
                        if nrel.arg == brel.arg:
                            found_nrel_in_base = True
                            # merge labels
                            for key in nrel.sprl_label:
                                if key not in brel.sprl_label:
                                    brel.sprl_label[key] = nrel.sprl_label[key]
                            # merge srl labels
                            if nrel.has_srl_label() and not brel.has_srl_label():
                                brel.srl_label = nrel.srl_label
                            # merge text
                            if nrel.arg.text != brel.arg.text:
                                txt = brel.arg.text if len(brel.arg.text) > 0 else nrel.arg.text
                                brel.arg.text = txt
                            break

                    # If no matching arg was found, add the new arg
                    if not found_nrel_in_base:
                        brels.append(nrel)


    base_predictions.reset_compare_to_default()
    if type(base_predictions) == PredictedSentenceItem:
        base_predictions.prediction_id = new_predictions.prediction_id

    return base_predictions

def fuzzy_text_match(text1: str, text2: str) -> bool:
    # Remove whitespace and punctuation and convert to lowercase
    text1 = re.sub(r'\W+', '', text1)
    text2 = re.sub(r'\W+', '', text2)
    text1 = text1.lower()
    text2 = text2.lower()
    # Compare texts
    return text1 == text2

def fuzzy_span_match(span1: SpanItem, span2: SpanItem, char_threshold: int = 5,
        len_threshold: int = 8) -> bool:
    similar_length_by_chars = abs((span1.end_char - span1.start_char) - (span2.end_char - span2.start_char)) <= len_threshold
    similar_start_char = abs(span1.start_char - span2.start_char) <= char_threshold
    similar_end_char = abs(span1.end_char - span2.end_char) <= char_threshold
    similar_text = fuzzy_text_match(span1.text, span2.text)

    return similar_length_by_chars and similar_start_char and similar_end_char and similar_text

def merge_predictions_fuzzy(base: PredictedSentenceItem, new: PredictedSentenceItem) -> PredictedSentenceItem:
    base_predictions = base.model_copy(deep=True)
    new_predictions = new.model_copy(deep=True)
    # Merge predicates
    new_predictions.set_predicate_compare_roleset_ids(False)
    base_predictions.set_predicate_compare_roleset_ids(False)
    new_predictions.set_compare_text(False)
    base_predictions.set_compare_text(False)
    for pred in new_predictions.predicates:
        if pred not in base_predictions.predicates:
            base_predictions.predicates.append(pred)
            base_predictions.relations.append((pred, []))
        else:
            # Now, check if the roleset ID and text is different
            for base_pred in base_predictions.predicates:
                if base_pred == pred:
                    if base_pred.roleset_id != pred.roleset_id:
                        r_id = base_pred.roleset_id if len(base_pred.roleset_id) > 0 else pred.roleset_id
                        base_pred.roleset_id = r_id
                        pred.roleset_id = r_id
                        # Now do the same thing for the predicate within the relations, if any
                        for pred_, _ in base_predictions.relations:
                            if pred == pred_:
                                # Roleset ID merging
                                r_id = pred_.roleset_id if len(pred_.roleset_id) > 0 else pred.roleset_id
                                pred_.roleset_id = pred.roleset_id
                                pred.roleset_id = pred.roleset_id
                                break
                    if base_pred.text != pred.text:
                        txt = base_pred.text if len(base_pred.text) > 0 else pred.text
                        base_pred.text = txt
                        pred.text = txt
                        # Now do the same thing for the predicate within the relations, if any
                        for pred_, _ in base_predictions.relations:
                            if pred == pred_:
                                # Text merging
                                txt = pred_.text if len(pred_.text) > 0 else pred.text
                                pred_.text = txt
                                pred.text = txt
                                break
                    break
        preds_in_relations = [pred_ for pred_, _ in base_predictions.relations]
        if pred not in preds_in_relations:
            base_predictions.relations.append((pred, []))
        else:
            # Now, check if the roleset ID is different
            for pred_, _ in base_predictions.relations:
                if pred_ == pred:
                    if pred_.roleset_id != pred.roleset_id:
                        r_id = pred_.roleset_id if len(pred_.roleset_id) > 0 else pred.roleset_id
                        pred_.roleset_id = pred.roleset_id
                    if pred_.text != pred.text:
                        txt = pred_.text if len(pred_.text) > 0 else pred.text
                        pred_.text = txt
                    break

    # Merge relations
    new_predictions.set_relation_compare_sprl_labels(False)
    base_predictions.set_relation_compare_sprl_labels(False)
    for npred, nrels in new_predictions.relations:
        for bpred, brels in base_predictions.relations:
            if npred == bpred:
                for nrel in nrels:
                    found_nrel_in_base = False

                    # First check if there is a matching arg to be updated in the base item
                    for brel in brels:
                        # check matching args
                        if nrel.arg == brel.arg or fuzzy_span_match(nrel.arg, brel.arg):
                            found_nrel_in_base = True
                            # merge labels
                            for key in nrel.sprl_label:
                                if key not in brel.sprl_label:
                                    brel.sprl_label[key] = nrel.sprl_label[key]
                            # merge srl labels
                            if nrel.has_srl_label() and not brel.has_srl_label():
                                brel.srl_label = nrel.srl_label
                            # merge text
                            if nrel.arg.text != brel.arg.text:
                                txt = brel.arg.text if len(brel.arg.text) > 0 else nrel.arg.text
                                brel.arg.text = txt
                            break

                    # If no matching arg was found, add the new arg
                    if not found_nrel_in_base:
                        brels.append(nrel)
    base_predictions.reset_compare_to_default()
    if type(base_predictions) == PredictedSentenceItem:
        base_predictions.prediction_id = new_predictions.prediction_id
    return base_predictions

def roleset_id_auto_add(items: dict) -> dict:
    '''Automatically add roleset IDs to items that have predicates with only one roleset option (this function is slow)'''
    for sentence_id, item in items.items():
        # gold_item = item['gold']['item']
        if not item['predicted']['roleset_id']['item'] and item['predicted']['predicate']['item']:
            predicted_item = item['predicted']['predicate']['item']
        elif item['predicted']['roleset_id']['item']:
            predicted_item = item['predicted']['roleset_id']['item']
        else:
            continue
        for pred in predicted_item.predicates:
            if pred.roleset_id:
                continue
            roleset_info = get_roleset_ids(predicted_item.text, pred.start_char, pred.end_char)
            if len(roleset_info) == 1:
                pred.roleset_id = roleset_info[0]['id']
                item['predicted']['roleset_id']['prompts'].append({
                    'prediction_id': f'{sentence_id}|{pred.start_char}|{pred.end_char}',
                    'prompt': '',
                    'raw_output': roleset_info[0]['id']
                })
            else:
                logger.info(f'Roleset ID not found for {pred} in sentence {sentence_id}.')
                pass
        # Merge items into the main dictionary
        items[sentence_id]['predicted']['roleset_id']['item'] = predicted_item

    return items

def prune(gold_item, predicted_item, component):
    if component not in ['predicate', 'argument']:
        raise ValueError(f'Component {component} not recognized for pruning.')
    gold_item.set_compare_text(False) # comparing just the spans at this point
    predicted_item.set_compare_text(False) # Comparing just the spans at this point
    if component == 'predicate':
        predicates_to_delete = []
        for predicted_predicate in predicted_item.predicates:
            if predicted_predicate not in gold_item.predicates:
                predicates_to_delete.append(predicted_predicate)
        for predicate_to_delete in predicates_to_delete:
            predicted_item.delete_predicate(predicate_to_delete)
    else:
        predicates_to_delete = []
        for predicted_predicate, predicted_relations in predicted_item.relations:
            if predicted_predicate not in gold_item.predicates:
                predicates_to_delete.append(predicted_predicate)
                continue # Skip relations check if predicate is not in gold data - relations will be deleted with predicate
            relations_to_delete = []
            for predicted_relation in predicted_relations:
                predicted_span = predicted_relation.arg
                gold_spans = [rel.arg for pred, rels in gold_item.relations for rel in rels if predicted_predicate == pred]
                if predicted_span not in gold_spans:
                    relations_to_delete.append(predicted_relation)
            for relation_to_delete in relations_to_delete:
                predicted_item.delete_relation(predicted_predicate, relation_to_delete)
        for predicate_to_delete in predicates_to_delete:
            predicted_item.delete_predicate(predicate_to_delete) # This also deletes all the relations for that predicate
    predicted_item.reset_compare_to_default() # Pruning is done, so reset comparison settings
    gold_item.reset_compare_to_default()
    return predicted_item

def split_output_per_setting(system_output):
    ids = list(system_output.keys())
    per_setting = {}
    for id_ in ids:
        if id_ == 'config' or id_ == 'batch_progress' or id_ == 'total_elapsed_time':
            continue
        new_id_splits = id_.split('-')
        prediction_id = new_id_splits[0]
        if len(new_id_splits) == 2:
            pstring = new_id_splits[1]
            component = 'srl+sprl-' + pstring
            if component not in per_setting.keys():
                per_setting[component] = {prediction_id: system_output[id_]}
            else:
                per_setting[component][prediction_id] = system_output[id_]
        elif len(new_id_splits) == 1:
            if 'srl' not in per_setting.keys():
                per_setting['srl'] = {prediction_id: system_output[id_]}
            else:
                per_setting['srl'][prediction_id] = system_output[id_]
        else:
            pstring = ''
            for p_ in new_id_splits[1:]:
                pstring += p_ + '-'
            pstring = pstring[:-1]

            component = 'srl+sprl-' + pstring
            if component not in per_setting.keys():
                per_setting[component] = {prediction_id: system_output[id_]}
            else:
                per_setting[component][prediction_id] = system_output[id_]
    return per_setting

def dump_hf_responses(items: dict, output_directory: str, include_prompts: bool = True, 
        exclusions: List[str] = ['compare_text', 'compare_roleset_ids', 'compare_sprl_labels']):
    '''
    Dumps the responses from the items dict to a .json file and a .pkl file
    '''
    # First, convert the SentenceItem or PredictedSentenceItem objects to dicts
    items_to_dump = {}
    for sentence_id in items.keys():
        items_to_dump[sentence_id] = {}
        for key in items[sentence_id].keys():
            if key == 'gold':
                items_to_dump[sentence_id][key] = {}
                items_to_dump[sentence_id][key]['item'] = items[sentence_id][key]['item'].model_dump(
                    exclude_defaults=True,
                    exclude=exclusions
                )
            else: # key == 'predicted'
                items_to_dump[sentence_id][key] = {}
                for component in items[sentence_id][key].keys():
                    items_to_dump[sentence_id][key][component] = {}
                    items_to_dump[sentence_id][key][component] = {}
                    if items[sentence_id][key][component]['item']:
                        items_to_dump[sentence_id][key][component]['item'] = items[sentence_id][key][component]['item'].model_dump(
                            exclude_defaults=True,
                            exclude=exclusions
                        )
                    else:
                        items_to_dump[sentence_id][key][component]['item'] = None
                    if include_prompts:
                        items_to_dump[sentence_id][key][component]['prompts'] = items[sentence_id][key][component]['prompts']
    
    logger.info(f'Dumping parsed responses to {output_directory}...')

    # Dump to .json file  
    with open(f'{output_directory}/parsed-responses.json', 'w') as f:
        json.dump(items_to_dump, f, indent=2)
        f.close()

    # Dump to .pkl file
    with open(f'{output_directory}/parsed-responses.pkl', 'wb') as f:
        pickle.dump(items_to_dump, f)
        f.close()

def get_items_from_dump(output_path: str):
    logger.info(f'Loading parsed responses from {output_path}...')
    with open(f'{output_path}/parsed-responses.pkl', 'rb') as f:
        items = pickle.load(f)

    components_parsed = set()
    # Re-instantiate the model dumps into SentenceItem and PredictedSentenceItem objects
    for sentence_id in items.keys():
        items[sentence_id]['gold']['item'] = SentenceItem(**items[sentence_id]['gold']['item'])
        for component in items[sentence_id]['predicted'].keys():
            if component in ['sprl', 'srl+sprl', 'srl+sprl-o', 'srl+sprl-p']:
                for pstring in items[sentence_id]['predicted'][component].keys():
                    if items[sentence_id]['predicted'][component][pstring]['item']:
                        components_parsed.add(f'{component}-{pstring}')
                        items[sentence_id]['predicted'][component][pstring]['item'] = PredictedSentenceItem(**items[sentence_id]['predicted'][component][pstring]['item'])
            else:
                if items[sentence_id]['predicted'][component]['item']:
                    components_parsed.add(component)
                    items[sentence_id]['predicted'][component]['item'] = PredictedSentenceItem(**items[sentence_id]['predicted'][component]['item'])
    
    logger.info(f'Parsed responses from components {components_parsed}.')
    return items, list(components_parsed)