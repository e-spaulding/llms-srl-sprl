import os
import pickle
import logging

from yaml import load, Loader, dump
from typing import List
from openai import OpenAI
from tqdm import tqdm
from typer import Typer

from .batches import aggregate_requests, send_batch, wait_for_batch, get_batch_output
from ..util.macros import *
from ..data2prompts.items import SentenceItem
from ..output_interpretation.parsing import merge_gpt_output, get_items_from_dump
from ..util.util import prep_out_dir_for_experiments
from ..data2prompts.items2prompts import get_chain_of_thought_prompts_from_item

app = Typer(
    pretty_exceptions_show_locals=False
)
logger = logging.getLogger(__name__)


def run_pipeline(config: dict, config_path: str, out_dir: str, pstring: str,
        path_to_data: str, task_in_path: str, sprl_subset: List[str], client: OpenAI):
    '''Run all components of the pipeline'''
    
    # Generate and send requests for each component
    components = ['predicate', 'roleset_id', 'argument', 'srl', f'sprl-{pstring}', f'srl+sprl-{pstring}']
    
    # initialize the items dict
    if config['check_for_existing_parses']:
        items, components_parsed = get_items_from_dump(out_dir)
        # Initialize the rest of the items dict
        for sentence_id in items.keys():
            for c in components:
                if 'sprl' in c:
                    if c not in items[sentence_id]['predicted']:
                        items[sentence_id]['predicted'][c] = {}
    else:
        # Load items from corpus and initialize the items dict for the first time
        f = open(path_to_data, 'rb')
        data = pickle.load(f)
        sentence_items = [SentenceItem(**item) for item in data]
        items = {item.sentence_id: {'gold': {'item': item}} for item in sentence_items}
        components_parsed = []
        for sentence_id in items.keys():
            items[sentence_id]['predicted'] = {}
            for c in components:
                items[sentence_id]['predicted'][c] = {'item': None, 'prompts': []}
    
    if 'srl' in task_in_path:
        system_message = f'You\'re tasked with using Semantic Role Labeling (SRL) to identify Predicates, Arguments, and their Semantic Roles from text.'
    else:
        system_message = f'"You\'re tasked with using Semantic Proto-role Labeling (SPRL) to identify Predicates, Arguments, and their SPRL Properties from text.'
    
    previous_component = None
    for component in components:

        # In case this component has already been run, check for the output file:
        if os.path.exists(os.path.join(out_dir, f'{component}-batch-output.jsonl')):
            logger.warning(f'Batch output file {component}-batch-output.jsonl already exists. Using the already-existing output as input for the next component and skipping this component.')
            
            # Update our items dict
            if component not in components_parsed:
                items = merge_gpt_output(
                    items=items,
                    output_directory=out_dir,
                    component=component,
                    prune_items=True if component in ['predicate', 'argument'] else False,
                    pstring=pstring if component != 'srl' else None
                )
                components_parsed.append(component)
                previous_component = component
            continue # skip this component; go to next

        # Find output from the previous component
        if previous_component and previous_component not in components_parsed:
            items = merge_gpt_output(
                items=items,
                output_directory=out_dir,
                component=previous_component,
                prune_items=True if previous_component in ['predicate', 'argument'] else False,
                pstring=pstring if previous_component != 'srl' else None
            )
            components_parsed.append(previous_component)
                        
        # Generate requests
        prompts = {}
        for sentence_id in tqdm(items.keys(), total=len(items.keys()), desc=f'Generating {component} requests'):
            if component.startswith('srl+sprl'):
                template = f'{config["template_path"]}/pipeline-srl.jinja'
            elif component.startswith('sprl'):
                template = f'{config["template_path"]}/pipeline-sprl.jinja'
            else:
                template = f'{config["template_path"]}/pipeline-{component}.jinja'

            template_options = {'srl': 'srl' in task_in_path,
                'prepend_previous_prompt': True if previous_component else None,
                'sprl_subset': sprl_subset}

            if previous_component:
                if component.startswith('sprl'):
                    item_dict = items[sentence_id]['predicted']['argument']
                else:
                    item_dict = items[sentence_id]['predicted'][previous_component]
            else:
                item_dict = items[sentence_id]['gold']
            new_prompts, new_item_dict = get_chain_of_thought_prompts_from_item(item_dict=item_dict, component=component,
                template=template, model_ver=config['model_ver'], **template_options)
            prompts.update(new_prompts)
            if component == 'roleset_id':
                items[sentence_id]['predicted']['roleset_id'] = new_item_dict # update the items dict with the new roleset_id
        request_path = f'{out_dir}/{component}-requests.jsonl'
        aggregate_requests(prompts=prompts, model_ver=config['model_ver'], system_message=system_message, output_path=request_path)
        # batch_id = send_batch(f'{component}-requests.jsonl', component, out_dir, config) # uncomment to send batch to api
        # completed_batch = wait_for_batch(batch_id, client) # uncomment to sent batch to api
        previous_component = component

        # if completed_batch: # uncomment to send batch to api
        #     output_filename = get_batch_output(batch_id, client, component, out_dir)
        #     logger.info(f'Batch job {batch_id} complete. Output saved to {output_filename}.')
        # else:
        #     logger.error(f'Batch job {batch_id} failed.')
        #     return
    
    logger.info(f'Pipeline run for properties {pstring} from {config_path} complete.')
    return

@app.command()
def run_experiment(config_path: str):
    '''Send a batch of OpenAI requests to the API and wait for completion
    before sending again as per the configuration file at config_path.'''
    # Load the configuration file
    with open(config_path) as f:
        config = load(f, Loader)

    # Paths and such
    pstrings = config['property_strings']
    if 'timestring_override' in config.keys():
        tstring = config['timestring_override']
    else:
        tstring = None

    path_to_data = config['path_to_data']
    
    # Get out_dir from config_path
    out_dir = config['out_dir']
    out_dir, tstring = prep_out_dir_for_experiments(out_dir=out_dir, tstring_override=tstring)
    # Save config file to out_dir
    with open(f'{out_dir}/config.yaml', 'w') as f:
        dump(config, f)
        f.close()
    # Logging
    if 'log_level_override' in config.keys():
        log_level = config['log_level_override']
    else:
        log_level = logging.INFO
    file_handler = logging.FileHandler(f'{out_dir}/run.log')
    file_handler.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level) 

    logging.basicConfig(level=log_level, handlers=[console_handler, file_handler])

    # Initialize the OpenAI client
    client = OpenAI(api_key=config['api_key'])

    for pstring in pstrings:
        if pstring == 'intersection':
            sprl_subset = LABELS_INTERSECTION
        elif pstring == 'all':
            sprl_subset = list(SPR_DEFINITIONS.keys())
        else:
            sprl_subset = pstring.split('-')
        
        out_dir, tstring = prep_out_dir_for_experiments(out_dir=config['out_dir'], tstring_override=tstring)
        task_in_path = f'srl+sprl-{pstring}'

        run_pipeline(config=config,
            config_path=config_path,
            out_dir=out_dir,
            pstring=pstring,
            path_to_data=path_to_data,
            task_in_path=task_in_path,
            sprl_subset=sprl_subset,
            client=client)

    logger.info(f'Pipeline run for all property configurations from {config_path} complete.')
    return

if __name__ == '__main__':
    app()