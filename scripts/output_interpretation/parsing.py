import re
import json
import logging
import pickle
import os

from typing import List
from tqdm import tqdm
from typer import Typer

from ..data2prompts.items import PredicateItem, SpanItem, RelationItem, PredictedSentenceItem, SentenceItem
from .utils import *
from ..util.macros import PREDICATE_REGEX, ARGUMENT_REGEX

logger = logging.getLogger(__name__)
app = Typer(pretty_exceptions_show_locals=False)

'''
Parsing functions for different prompt variants
'''

def parse_json_with_one_arg(prediction_id: str, text: str, component: str = '') -> PredictedSentenceItem:
    '''Parse json response with only one argument for a single predicate'''
    
    sentence_id, predicate_start_char, predicate_end_char, argument_start_char, argument_end_char = prediction_id.split('|')

    # We can create the sentence item with just the ID info (assuming it was captured in a previous component)
    predicate = PredicateItem(text='', start_char=int(predicate_start_char), end_char=int(predicate_end_char))
    relation = RelationItem(arg=SpanItem(text='', start_char=int(argument_start_char), end_char=int(argument_end_char)))

    # Remove code block tags
    text = text.replace('```json\n', '')
    text = text.replace('\n```', '')
    
    try:
        # Try to open the JSON
        out_dict = json.loads(text)
    except:
        # If it fails, return relation item without labels (but with span info)
        logger.warning(f'Failed to parse JSON for prediction {prediction_id}.')
        item = PredictedSentenceItem(sentence_id=sentence_id,
            text='', predicates=[predicate], relations=[(predicate, [relation])], prediction_id=prediction_id)
        return item
    
    keys = list(out_dict.keys())
    argument_key = keys[0]
    for label in out_dict[argument_key].keys():
        if label != 'role':
            relation.sprl_label[label] = out_dict[argument_key][label]
        else:
            relation.srl_label = out_dict[argument_key][label]

    item = PredictedSentenceItem(sentence_id=sentence_id,
        text='',
        predicates=[predicate],
        relations=[(predicate, [relation])],
        prediction_id=prediction_id)

    return item

def parse_json_with_all_args(prediction_id: str, text: str, component: str = '', blacklist: List[str] = None) -> PredictedSentenceItem:
    '''Parse json response expecting all args for a single predicate'''
    
    sentence_id, predicate_start_char, predicate_end_char = prediction_id.split('|')

    # We can create the sentence item with just the ID info (assuming it was captured in a previous component)
    predicate = PredicateItem(
        text='', start_char=int(predicate_start_char), end_char=int(predicate_end_char)
    )

    # Remove code block tags
    text = text.replace('```json\n', '')
    text = text.replace('\n```', '')
    text = text.replace('```', '')
    text = text.replace('\\n', '')
    
    try:
        # Try to open the JSON
        out_dict = json.loads(text)
    except:
        out_dict = try_json_at_every_char(text)
        if out_dict is None:
            # If it fails, return relation item without labels (but with span info)
            logger.warning(f'Failed to parse JSON for prediction {prediction_id}.')
            if blacklist:
                blacklist.append(prediction_id)
            item = PredictedSentenceItem(sentence_id=sentence_id,
                text='',
                predicates=[predicate],
                relations=[(predicate, [])],
                prediction_id=prediction_id)
            return item
    
    if type(out_dict) != dict:
        # could be a list
        if type(out_dict) == list:
            out_dict = out_dict[0]
        else:
            # beats me
            logger.warning(f'Failed to parse JSON for prediction {prediction_id}.')
            if blacklist:
                blacklist.append(prediction_id)
            item = PredictedSentenceItem(sentence_id=sentence_id,
                text='',
                predicates=[predicate],
                relations=[(predicate, [])],
                prediction_id=prediction_id)
            return item

    sentence_text_from_output = out_dict.get('text', '')
    
    predicate_dict_from_output = out_dict.get('predicates', {})
    if len(predicate_dict_from_output) == 0 or type(predicate_dict_from_output) != dict:
        # If no predicates are found, return an empty item
        logger.warning(f'Failed to parse JSON for prediction {prediction_id}.')
        if blacklist:
            blacklist.append(prediction_id)
        item = PredictedSentenceItem(sentence_id=sentence_id,
            text='',
            predicates=[predicate],
            relations=[(predicate, [])],
            prediction_id=prediction_id)
        return item
    predicate_text_from_output = list(predicate_dict_from_output.keys())[0]

    # Update predicate item
    predicate.text = predicate_text_from_output

    # Create relation items
    relations = []
    arg_dict_from_output = predicate_dict_from_output[predicate_text_from_output].get('arguments', {})
    if len(arg_dict_from_output) == 0 or type(arg_dict_from_output) != dict:
        # If no arguments are found, return an empty item
        if blacklist:
            blacklist.append(prediction_id)
        logger.warning(f'Failed to parse JSON for prediction {prediction_id}.')
        item = PredictedSentenceItem(sentence_id=sentence_id,
            text='',
            predicates=[predicate],
            relations=[(predicate, [])],
            prediction_id=prediction_id)
        return item
    for arg_key_from_output in arg_dict_from_output:
        if type(arg_dict_from_output[arg_key_from_output]) != dict:
            # If the argument is not a dictionary, skip it
            continue
        arg_start_char = arg_dict_from_output[arg_key_from_output].get('start_char', sentence_text_from_output.find(arg_key_from_output))
        arg_end_char = arg_start_char + len(arg_key_from_output)
        arg = SpanItem(text=arg_key_from_output, start_char=arg_start_char, end_char=arg_end_char)
        
        # get labels
        sprl_label = {}
        srl_label = ''
        for key in arg_dict_from_output[arg_key_from_output].keys():
            if key != 'role' and key != 'start_char':
                sprl_label[key] = arg_dict_from_output[arg_key_from_output][key]
            elif key == 'role':
                srl_label = arg_dict_from_output[arg_key_from_output][key]
                # ensure srl_label is a string and normalize it
                if type(srl_label) == str:
                    srl_label = normalize_srl_label(srl_label)
                else:
                    srl_label = ''

        relation = RelationItem(arg=arg, sprl_label=sprl_label, srl_label=srl_label)
        relations.append(relation)

    item = PredictedSentenceItem(sentence_id=sentence_id,
        text=sentence_text_from_output,
        predicates=[predicate],
        relations=[(predicate, relations)],
        prediction_id=prediction_id)

    return item

def parse_tagged_text(prediction_id: str, text: str) -> PredictedSentenceItem:
    '''
    Parse char spans from response with <PRED></PRED><ARG></ARG> tags
    '''
    predicate_items = []
    relations = []
    r_items = []
    recovered_text = f'{text}'
    id_parts = prediction_id.split('|')
    if len(id_parts) == 1:
        sentence_id = id_parts[0]
    elif len(id_parts) == 3:
        sentence_id, predicate_start_char, predicate_end_char = id_parts
    else:
        raise ValueError(f'Invalid prediction ID for tagged text: {prediction_id}')
    # Get char spans for each predicate and argument and remove all special tags from recovered_text
    while '<PRED>' in recovered_text or '<ARG>' in recovered_text:
        any_tag_match = re.search('<PRED>|</PRED>|<ARG>|</ARG>', recovered_text)
        if not any_tag_match:
            break
        elif any_tag_match.group() == '<PRED>':
            # handle predicate
            match = re.search(PREDICATE_REGEX, recovered_text)
            if match:
                tagged_begin, tagged_end = match.span('tagged_pred') # <PRED>sat</PRED>
                pred_text = match.group('pred') # sat
                new_end_char = len(recovered_text[:tagged_begin]) + len(pred_text)
                # Delete the tags
                recovered_text = recovered_text[:tagged_begin] + pred_text + recovered_text[tagged_end:]
                predicate_item = PredicateItem(text=pred_text, start_char=tagged_begin, end_char=new_end_char)
                predicate_items.append(predicate_item)
            else:
                # logger.debug(f'No match for predicate within <PRED></PRED> tags in {prediction_id}. Deleting tags and continuing.')
                recovered_text = recovered_text[:any_tag_match.start()] + recovered_text[any_tag_match.end():]
                continue
            
        elif any_tag_match.group() == '<ARG>':
            # handle argument
            match = re.search(ARGUMENT_REGEX, recovered_text)
            if match:
                tagged_begin, tagged_end = match.span('tagged_arg') # <ARG>mat</ARG>
                arg_text = match.group('arg') # mat
                new_end_char = len(recovered_text[:tagged_begin]) + len(arg_text)
                # Delete the tags
                recovered_text = recovered_text[:tagged_begin] + arg_text + recovered_text[tagged_end:]
                arg_item = SpanItem(text=arg_text, start_char=tagged_begin, end_char=new_end_char)

                r_item = RelationItem(arg=arg_item)
                r_items.append(r_item)
            else:
                # logger.debug(f'No match for argument within <ARG></ARG> tags in {prediction_id}. Deleting tags and continuing.')
                recovered_text = recovered_text[:any_tag_match.start()] + recovered_text[any_tag_match.end():]
                continue
        elif any_tag_match.group() == '<PROTO>':
            # Because this is handled in the ARG tag, this code should be unreachable - raise an error
            raise ValueError(f"PROTO tag found before ARG tag: {recovered_text}")
        else:
            if any_tag_match.group() == '</ARG>':
                # delete
                recovered_text = recovered_text[:any_tag_match.start()] + recovered_text[any_tag_match.end():]
                continue
            elif any_tag_match.group() == '</PRED>':
                # delete
                recovered_text = recovered_text[:any_tag_match.start()] + recovered_text[any_tag_match.end():]
                continue
            else:
                raise ValueError(f"Invalid tag match: {any_tag_match.group()}")
    
    # Check if there is more than one predicate per set of args and handle
    if len(predicate_items) > 1 and len(r_items) > 0:
        predicted_sentence_item = PredictedSentenceItem(sentence_id=sentence_id, text='', predicates=[predicate_items[-1]],
            relations=[(predicate_items[-1], r_items)], prediction_id=prediction_id)
        return predicted_sentence_item
    elif len(predicate_items) == 1 and len(r_items) > 0:
        relations.append((predicate_items[0], r_items))

    predicted_sentence_item = PredictedSentenceItem(sentence_id=sentence_id, text=recovered_text, predicates=predicate_items,
        relations=relations, prediction_id=prediction_id)

    return predicted_sentence_item

def parse_single_response(prediction_id: str, text: str,
    component: str, blacklist: List[str] = None) -> PredictedSentenceItem:
    if component not in ['srl', 'roleset_id', 'srl+sprl', 'srl+sprl-o', 'srl+sprl-p'] and (not component.startswith('srl') and not component.startswith('sprl')):
        raise ValueError(f'Component {component} not recognized for this parsing function.')
    id_parts = prediction_id.split('|')
    if len(id_parts) == 1:
        raise ValueError(f'Invalid prediction ID for {component}: {prediction_id}')
    elif len(id_parts) == 3:
        sentence_id, predicate_start_char, predicate_end_char = id_parts
        assert component == 'roleset_id', f'Prediction ID {prediction_id} not compatible with component {component} in this parsing function.'
    elif len(id_parts) == 5:
        sentence_id, predicate_start_char, predicate_end_char, argument_start_char, argument_end_char = id_parts
        assert component.startswith('srl'), f'Prediction ID {prediction_id} not compatible with component {component} in this parsing function.'
    else:
        raise ValueError(f'Invalid prediction ID for {component}: {prediction_id}')
    
    # Create the sentence item with just the ID info (assuming it was captured in a previous component)
    predicate = PredicateItem(text='', start_char=int(predicate_start_char), end_char=int(predicate_end_char))

    if component == 'roleset_id':
        # Add parsed roleset to PredicateItem
        predicate.roleset_id = text
        return PredictedSentenceItem(sentence_id=sentence_id,
            text='',
            predicates=[predicate],
            relations=[],
            prediction_id=prediction_id)
    elif component.startswith('srl'):
        # Create the RelationItem with the ID info and add parsed role to RelationItem
        srl_label = normalize_srl_label(text)
        if blacklist and not check_srl_label(srl_label):
            blacklist.append(prediction_id)
        span = SpanItem(text='', start_char=int(argument_start_char), end_char=int(argument_end_char))
        relation = RelationItem(arg=span, srl_label=srl_label)

        return PredictedSentenceItem(sentence_id=sentence_id,
            text='',
            predicates=[predicate],
            relations=[(predicate, [relation])],
            prediction_id=prediction_id)

'''
Functions for parsing entire batches of output and combining separate predictions 
from the same sentence into a single item for each sentence
'''

def check_previous_components_and_merge(items, sentence_id, predicted_item, current_component, fuzzy_merge = False):
    if fuzzy_merge:
        merge_function = merge_predictions_fuzzy
    else:
        merge_function = merge_predictions
    if current_component.startswith('sprl') or current_component.startswith('srl+sprl'):
        splits = current_component.split('-')
        if len(splits) == 2:
            pstring = splits[1]
        elif len(splits) > 2:
            pstring = '-'.join(splits[1:])
        component_pipeline = ['predicate', 'roleset_id', 'argument', f'sprl-{pstring}', f'srl+sprl-{pstring}']
    else:
        component_pipeline = ['predicate', 'roleset_id', 'argument', 'srl']
    
    index = component_pipeline.index(current_component)
    # Then, check the previous components, starting backwards from the current component
    for i in range(index, -1, -1):
        component = component_pipeline[i]
        if component not in items[sentence_id]['predicted']:
            continue
        # Check for an existing parse
        if items[sentence_id]['predicted'][component]['item']:
            predicted_item = merge_function(items[sentence_id]['predicted'][component]['item'], predicted_item)
    # If no previous components have been parsed, return the current component
    return predicted_item

def merge_gpt_output(items: dict, output_directory: str, component: str,
    json_all_args: bool = False, oracle_pred_arg: bool = False, prune_items: bool = False,
    pstring: str = None) -> dict:
    '''build up dict of srl+sprl items from gpt output; meant to be flexible across prompt variants'''

    # Attempt to open file with standard naming convention
    output_file = f'{output_directory}/{component}-batch-output.jsonl'
    with open(output_file) as f:
        jsonl_lines = f.readlines()
    requests_file = f'{output_directory}/{component}-requests.jsonl'
    with open(requests_file) as f:
        requests = f.readlines()

    if pstring is None and (component.startswith('sprl') or component.startswith('srl+sprl')):
        raise ValueError(f'pstring must be provided for component {component}.')
    
    if (component.startswith('srl') or component.startswith('sprl')) and pstring and pstring not in component:
        component = f'{component}-{pstring}'

    # Collect request and response data by ID
    requests_by_id = {}
    responses_by_id = {}
    for line in requests:
        request_dict = json.loads(line)
        requests_by_id[request_dict['custom_id']] = request_dict['body']['messages'][1]['content']
    for line in jsonl_lines:
        response_dict = json.loads(line)
        responses_by_id[response_dict['custom_id']] = response_dict['response']['body']['choices'][0]['message']['content']

    # Find missing IDs
    request_key_set = set(requests_by_id.keys())
    response_key_set = set(responses_by_id.keys())
    missing_ids = request_key_set - response_key_set
    if len(missing_ids) > 0:
        logger.debug(f'Missing IDs from batch output in component {component}: {missing_ids}')
        missing_id_file = open(f'{output_directory}/{component}-missing-ids.txt', 'w')
        for missing_id in missing_ids:
            missing_id_file.write(f'{missing_id}\n')
        missing_id_file.close()
    
    # Make oracle items if necessary
    if component not in items[list(items.keys())[0]]['predicted'].keys():
        # Add the component to the items dictionary
        for sentence_id in items.keys():
            if oracle_pred_arg:
                # make oracle item
                oracle_item = items[sentence_id]['gold']['item'].model_copy(deep=True)
                for predicate, relations in oracle_item.relations:
                    for relation in relations:
                        relation.srl_label = ''
                        relation.sprl_label = {}
                items[sentence_id]['predicted'][component] = {'item': oracle_item, 'prompts': []}
            else:
                items[sentence_id]['predicted'][component] = {'item': None, 'prompts': []}

    # Process the output
    for prediction_id in tqdm(responses_by_id.keys(), total=len(responses_by_id.keys()), desc=f'Processing batch output for component {component}'):
        # ID handling
        id_parts = prediction_id.split('|')
        if len(id_parts) == 1:
            sentence_id = id_parts[0]
        elif len(id_parts) == 3:
            sentence_id, predicate_start_char, predicate_end_char = id_parts
        elif len(id_parts) == 5:
            sentence_id, predicate_start_char, predicate_end_char, arg_start_char, arg_end_char = id_parts
        else:
            raise ValueError(f'Custom ID {prediction_id} not recognized.')

        if sentence_id not in items.keys():
            # logger.warning(f'Sentence ID {sentence_id} for Prediction {prediction_id} not found in items dictionary.')
            continue

        # Get content of response
        content = responses_by_id[prediction_id]
        # Get prompt from request
        try:
            prompt = requests_by_id[prediction_id]
        except KeyError:
            continue

        # Parse content
        if component == 'predicate' or component == 'argument':
            predicted_item = parse_tagged_text(prediction_id, content)
            
            if prune_items:
                gold_item = items[sentence_id]['gold']['item']
                predicted_item = prune(gold_item, predicted_item, component)
            
            # Now finally integrate the predicted item into the items dictionary, merging as needed
            items[sentence_id]['predicted'][component]['item'] = check_previous_components_and_merge(items, sentence_id, predicted_item, component)
            items[sentence_id]['predicted'][component]['prompts'].append({'prediction_id': prediction_id,
                'prompt': prompt,
                'raw_output': content})
        else:
            if (component == 'roleset_id' or component == 'srl' or component.startswith('srl+sprl')) and not json_all_args:
                parsing_fn = parse_single_response
            elif component.startswith('sprl') and not json_all_args:
                parsing_fn = parse_json_with_one_arg
            elif (component.startswith('srl+sprl') or component.startswith('sprl')) and json_all_args:
                parsing_fn = parse_json_with_all_args
            else:
                raise ValueError(f'Component {component} not recognized for this parsing function.')
            predicted_item = parsing_fn(prediction_id, content, component)
            items[sentence_id]['predicted'][component]['item'] = check_previous_components_and_merge(items, sentence_id, predicted_item, component)
            items[sentence_id]['predicted'][component]['prompts'].append({'prediction_id': prediction_id,
                'prompt': prompt,
                'raw_output': content})
        
    return items

def merge_hf_llm_output(items: dict, system_output: dict,
    component: str, json_all_args: bool = True) -> dict:
    '''build up dict of srl+sprl items given previous output (or oracle)'''

    if json_all_args:
        parsing_fn = parse_json_with_all_args
    else:
        parsing_fn = parse_single_response
    
    blacklist = []

    # Process the output
    for prediction_id in tqdm(system_output.keys(), total=len(system_output.keys()), desc=f'Processing system output for component {component}'):
        # ID handling
        id_parts = prediction_id.split('|')
        if len(id_parts) == 1:
            sentence_id = id_parts[0]
        elif len(id_parts) == 3:
            sentence_id, predicate_start_char, predicate_end_char = id_parts
        elif len(id_parts) == 5:
            sentence_id, predicate_start_char, predicate_end_char, arg_start_char, arg_end_char = id_parts
        else:
            raise ValueError(f'Custom ID {prediction_id} not recognized.')

        # Get content of response
        content = system_output[prediction_id]['clean_output']
        # Get prompt from request
        prompt = system_output[prediction_id]['prompt']

        # Parse content
        predicted_item = parsing_fn(prediction_id, content, component, blacklist=blacklist)
        items[sentence_id]['predicted'][component]['item'] = check_previous_components_and_merge(items, sentence_id, predicted_item, component, fuzzy_merge=True)
        items[sentence_id]['predicted'][component]['prompts'].append({'prediction_id': prediction_id,
            'prompt': prompt,
            'raw_output': content})
        
    return items, blacklist

@app.command()
def process_hf_system_output(system_output_path: str, gold_path: str):
    '''Create parsed dump of srl+sprl items for models accessed through huggingface
    (pythia, llama, olmo, tulu, qwen, etc.)'''

    log_level = logging.INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level) 

    logging.basicConfig(
        level=log_level,
        handlers=[console_handler]
    )
    
    # Load system output
    with open(system_output_path, 'r') as f:
        system_output = json.load(f)

    # Split output per setting
    system_output_per_setting = split_output_per_setting(system_output)

    # Get parent directory
    out_dir = os.path.dirname(system_output_path)

    # Load gold data
    f = open(gold_path, 'rb')
    gold_data = pickle.load(f)
    sentence_items = [SentenceItem(**item) for item in gold_data]
    items = {item.sentence_id: {'gold': {'item': item}, 'predicted': {}} for item in sentence_items}
    
    # initialize component dicts if necessary
    for sentence_id in items.keys():
        oracle_item = items[sentence_id]['gold']['item'].model_copy(deep=True)
        for predicate, relations in oracle_item.relations:
            for relation in relations:
                relation.srl_label = ''
                relation.sprl_label = {}
        for component in system_output_per_setting.keys():
            if component not in items[sentence_id]['predicted'].keys():
                items[sentence_id]['predicted'][component] = {'item': oracle_item, 'prompts': []}

    # Now parse the data
    logger.info(f'Parsing {system_output_path}...')
    blacklist = []
    for component in system_output_per_setting.keys():
        items, blacklist_ = merge_hf_llm_output(items, system_output_per_setting[component], component, json_all_args=False, prune_items=True)
        blacklist += blacklist_

    dump_hf_responses(items, out_dir)
    blacklist = list(set(blacklist))
    # Save blacklist to txt file
    with open(f'{out_dir}/blacklist.txt', 'w') as f:
        for item in blacklist:
            f.write(f'{item}\n')

if __name__ == '__main__':
    app()