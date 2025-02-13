import pickle
import logging

from typing import List
from openai import OpenAI
from tqdm import tqdm
from typer import Typer

from .batches import aggregate_requests, send_batch, wait_for_batch, get_batch_output
from ..util.util import ensure_output_path_exists
from ..util.macros import *
from ..data2prompts.items import SentenceItem
from ..output_interpretation.parsing import merge_gpt_output
from ..data2prompts.items2prompts import get_chain_of_thought_prompts_from_item, get_all_args_prompts_from_item, load_llama_sprl_prompts

app = Typer(
    pretty_exceptions_show_locals=False
)
logger = logging.getLogger(__name__)

def get_pb_guidelines_sys_prompt():
    pb_guidelines_sys_prompt = 'You\'re tasked with using Semantic Role Labeling (SRL) to identify Predicates, Arguments, and their Semantic Roles from text. Below are some guidelines for PropBank annotators that may be useful when choosing how to label Arguments for their Semantic Roles: \n\n'
    pb_g_file = open(PATH_TO_PB_GUIDELINES, 'r')
    pb_guidelines = pb_g_file.read()
    pb_g_file.close()
    pb_guidelines_sys_prompt += pb_guidelines
    return pb_guidelines_sys_prompt

def unpickle_gold_items(path_to_data: str):
    f = open(path_to_data, 'rb')
    data = pickle.load(f)
    sentence_items = [SentenceItem(**item) for item in data]
    items = {item.sentence_id: {'gold': {'item': item}} for item in sentence_items}
    return items

def add_oracle_items_to_dict(items: dict, components_to_add: List[str], pstrings: List[str]):
    for component in components_to_add:
        if component not in ['srl', 'sprl', 'srl+sprl']:
            raise ValueError(f'Invalid component: {component}')
    for sentence_id in items.keys():
        oracle_item = items[sentence_id]['gold']['item'].model_copy(deep=True)
        for predicate, relations in oracle_item.relations:
            for relation in relations:
                relation.srl_label = ''
                relation.sprl_label = {}
        for component in components_to_add:
            if 'predicted' not in items[sentence_id]:
                items[sentence_id]['predicted'] = {}
            if 'sprl' in component:
                for pstring in pstrings:
                    items[sentence_id]['predicted'][f'{component}-{pstring}'] = {'item': oracle_item.model_copy(deep=True), 'prompts': []}
                else:
                    items[sentence_id]['predicted'] = {component: {'item': oracle_item.model_copy(deep=True), 'prompts': []}}
    return items

@app.command()
def run_srl():

    # Logging
    log_level = logging.INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level) 
    logging.basicConfig(level=log_level,
        handlers=[console_handler])

    # Paths
    path_to_data = f'corpus/srl+sprl-with-na/1.test.core-roles-only.pkl'
    path_to_experiment = f'experiments/'
    path_to_experiment = ensure_output_path_exists(path_to_experiment)

    # Load items from corpus and initialize the items dict for the first time
    items = unpickle_gold_items(path_to_data)
    pstrings = ['intersection',
        'all',
        'volition-change_of_state',
        'instigation-change_of_location']
    components_to_add = ['sprl', 'srl+sprl', 'srl']
    items = add_oracle_items_to_dict(items, components_to_add, pstrings)
    
    # Generate requests for SRL alone
    component = 'srl'
    prompts = {}
    for sentence_id in tqdm(items.keys(), total=len(items), desc=f'Generating {component} requests'):
        template = f'templates/all-args.jinja'
        template_options = {'srl': True,
            'prepend_previous_prompt': False,
            'include_sprl': False}
        item_dict = items[sentence_id]
        new_prompts, new_item_dict = get_all_args_prompts_from_item(item_dict_with_gold_and_predicted=item_dict,
            template=template,
            **template_options)
        prompts.update(new_prompts)
    request_path = f'{path_to_experiment}/{component}-requests.jsonl' 
    aggregate_requests(prompts=prompts,
        model_ver='gpt-4o-2024-05-13',
        system_message='You\'re tasked with using Semantic Role Labeling (SRL) to identify Predicates, Arguments, and their Semantic Roles from text.',
        output_path=request_path)
    # send the batch
    config = {'api_key': API_KEY}
    batch_id = send_batch(request_path, component, path_to_experiment, config)
    # wait for batch
    client = OpenAI(api_key='')
    completed_batch = wait_for_batch(batch_id, client)
    if completed_batch:
        output_filename = get_batch_output(batch_id, client, component, path_to_experiment)
        logger.info(f'Batch job {batch_id} complete. Output saved to {output_filename}.')
    else:
        logger.error(f'Batch job {batch_id} failed.') # abort just in case
        return

    # Now generate requests for SRL+SPRL
    for pstring in pstrings:
        prompts = {}
        component = f'srl+sprl-{pstring}'
        for sentence_id in tqdm(items.keys(), total=len(items), desc=f'Generating {component} requests'):
            template = f'templates/all-args-na.jinja'
            template_options = {'srl': True,
                'prepend_previous_prompt': False,
                'include_sprl': True,
                'sprl_subset': pstring.split('-') if pstring != 'intersection' else LABELS_INTERSECTION,
                'system_message': 'You\'re tasked with using Semantic Role Labeling (SRL) to identify Predicates, Arguments, and their Semantic Roles from text.'}
            item_dict = items[sentence_id]
            new_prompts, new_item_dict = get_all_args_prompts_from_item(item_dict_with_gold_and_predicted=item_dict,
                template=template,
                **template_options)
            prompts.update(new_prompts)
        request_path = f'{path_to_experiment}/{component}-requests.jsonl'
        aggregate_requests(prompts=prompts,
            model_ver='gpt-4o-2024-05-13',
            system_message='You\'re tasked with using Semantic Role Labeling (SRL) to identify Predicates, Arguments, and their Semantic Roles from text.',
            output_path=request_path)

        # send batch
        config = {'api_key': API_KEY}
        batch_id = send_batch(request_path, component, path_to_experiment, config) # this checks if there's already a batch output file
        # wait for batch
        client = OpenAI(api_key='')
        completed_batch = wait_for_batch(batch_id, client)
        if completed_batch:
            output_filename = get_batch_output(batch_id, client, component, path_to_experiment)
            logger.info(f'Batch job {batch_id} complete. Output saved to {output_filename}.')
        else:
            logger.error(f'Batch job {batch_id} failed.')
            return
    return

@app.command()
def run_sprl_annotate_prompts():
    PROMPT_STR = 'Response: '
    SYS_MSG = ''
    path_to_experiment = f'{PATH_TO_THIS_REPO}/experiments/gpt-4o-2024-05-13/spr2/sprl-annotate'
    component = 'sprl'

    prompt_contexts_by_id = load_llama_sprl_prompts(
        path_to_data = f'{PATH_TO_THIS_REPO}/corpus/sprl/2.1.test.all.pkl',
        path_to_template_directory = f'{PATH_TO_THIS_REPO}/templates/sprl-annotate',
    )
    prompts = {}
    for id_ in prompt_contexts_by_id.keys():
        prompts[id_] = prompt_contexts_by_id[id_]['prompt'] + f'\n{PROMPT_STR}'

    request_path = f'{path_to_experiment}/{component}-requests.jsonl'
    aggregate_requests(prompts=prompts,
        model_ver='gpt-4o-2024-05-13',
        system_message=SYS_MSG,
        output_path=request_path)
    # send batch
    config = {'api_key': API_KEY}
    batch_id = send_batch(request_path, component, path_to_experiment, config)
    # wait for batch
    client = OpenAI(api_key=API_KEY)
    completed_batch = wait_for_batch(batch_id, client)
    if completed_batch:
        output_filename = get_batch_output(batch_id, client, component, path_to_experiment)
        logger.info(f'Batch job {batch_id} complete. Output saved to {output_filename}.')
    else:
        logger.error(f'Batch job {batch_id} failed.')
        return

if __name__ == '__main__':
    app()