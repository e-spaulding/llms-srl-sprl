import os
import pickle
import jinja2
import logging

from typing import Tuple
from tqdm import tqdm
from typer import Typer

from .items import SentenceItem
from ..output_interpretation.parsing import target_spans, target_all_relations
from .pb_lookup import roleset_options_to_string, get_roles_and_descs
from ..util.macros import SPR_DEFINITIONS, NEGATED_DEFINITIONS, PATH_TO_THIS_REPO

app = Typer(
    pretty_exceptions_show_locals=False
)
logger = logging.getLogger(__name__)


def get_chain_of_thought_prompts_from_item(
        item_dict: dict,
        component: str,
        template: str = "templates/template.jinja",
        **template_options) -> Tuple[dict, dict]:
    '''Turns an item object into a single string pipeline prompt
    given a .jinja template for the prompt'''

    environment = jinja2.Environment()
    jinja_template = environment.from_string(open(template).read())

    # Handle template options
    examples = template_options.get('examples', None)
    prepend_previous_prompt = template_options.get('prepend_previous_prompt', False)
    system_message = template_options.get('system_message', None)
    prepend_system_message_to_predicate_prompt = template_options.get('prepend_system_message_to_predicate_prompt', False)
    sprl_subset = template_options.get('sprl_subset', None)
    srl = template_options.get('srl', False) # Whether or not SRL is the end of the pipeline; default is False (i.e. SPRL is the end instead)

    sentence_item = item_dict['item']
    prompts = {}

    if not sentence_item:
        # logger.debug('No SentenceItem found in item_dict for this component. Skipping.')
        return [], item_dict

    if not system_message:
        if srl:
            system_message = f'You\'re tasked with using Semantic Role Labeling (SRL) to identify Predicates, Arguments, and their Semantic Roles from text.'
        else:
            system_message = f'"You\'re tasked with using Semantic Proto-role Labeling (SPRL) to identify Predicates, Arguments, and their SPRL Properties from text.'

    if component == 'predicate':
        prompt = jinja_template.render(
            sentence=sentence_item.model_dump(), # recursively turns the SentenceItem into a dict
            examples=examples
        )
        if prepend_system_message_to_predicate_prompt:
            prompt = f'{system_message}\n{prompt}'
        prompts[sentence_item.sentence_id] = prompt

    elif component == 'roleset_id':
        # One prompt per predicate

        for pred in sentence_item.predicates:
            prediction_id = f'{sentence_item.sentence_id}|{pred.start_char}|{pred.end_char}'
            if prediction_id in prompts.keys():
                # logger.debug(f'Prediction ID {prediction_id} already exists. Skipping.')
                continue
            roleset_options, singular_roleset_option = roleset_options_to_string(sentence_item.text, pred.start_char, pred.end_char)
            if roleset_options is None:
                # logger.debug(f'No roleset options found for predicate {pred.text} in sentence {sentence_item.sentence_id}. Skipping.')
                continue
            
            text_with_pred_targeted = target_spans(sentence_item.text, pred)
            prompt = jinja_template.render(
                predicate=pred.model_dump(),
                roleset_options=roleset_options,
                text_with_targets=text_with_pred_targeted
            )
            if prepend_previous_prompt:
                # Find previous prompt for this predicate
                found = False
                for prompt_dict in item_dict['prompts']:
                    if prompt_dict['prediction_id'] == sentence_item.sentence_id:
                        previous_prompt = prompt_dict['prompt']
                        raw_output = prompt_dict['raw_output']
                        found = True
                        break
                if not found:
                    logger.warning(f'Previous prompt not found for {prediction_id}. Skipping.')
                    continue
                prompt = f'{previous_prompt}\n{raw_output}\n{prompt}'
            if singular_roleset_option: # do not ask the LLM. Just choose the one option and move on
                pred.roleset_id = singular_roleset_option
                item_dict['item'] = sentence_item
                if 'prompts' not in item_dict.keys():
                    item_dict['prompts'] = []
                item_dict['prompts'].append({
                    'prediction_id': prediction_id,
                    'prompt': prompt,
                    'raw_output': singular_roleset_option
                })
                continue
            prompts[prediction_id] = prompt
    elif component == 'argument':
        # One prompt per predicate
        # prompts = []
        # ids = []
        for pred in sentence_item.predicates:
            prediction_id = f'{sentence_item.sentence_id}|{pred.start_char}|{pred.end_char}'
            if prediction_id in prompts.keys():
                # logger.debug(f'Prediction ID {prediction_id} already exists. Skipping.')
                continue
            text_with_pred_targeted = target_spans(sentence_item.text, pred)
            prompt = jinja_template.render(
                sentence=sentence_item.model_dump(),
                predicate=pred.model_dump(),
                examples=examples,
                text_with_targets=text_with_pred_targeted
            )
            if prepend_previous_prompt:
                # Find previous prompt for this predicate
                found = False
                for prompt_dict in item_dict['prompts']:
                    if prompt_dict['prediction_id'] == prediction_id:
                        previous_prompt = prompt_dict['prompt']
                        raw_output = prompt_dict['raw_output']
                        found = True
                        break
                if not found:
                    logger.warning(f'Previous prompt not found for {prediction_id}. Skipping.')
                    continue
                prompt = f'{previous_prompt}\n{raw_output}\n{prompt}'
            prompts[prediction_id] = prompt
    elif component.startswith('sprl'):
        # One prompt per argument
        # prompts = []
        # ids = []
        if sprl_subset:
            label_definitions = {k: v for k, v in SPR_DEFINITIONS.items() if k in sprl_subset}
        else:
            label_definitions = SPR_DEFINITIONS
        for pred, rels in sentence_item.relations:
            for rel in rels:
                prediction_id = f'{sentence_item.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}'
                if prediction_id in prompts.keys():
                    logger.warning(f'Prediction ID {prediction_id} already exists. Skipping.')
                    continue
                text_with_arg_targeted = target_spans(sentence_item.text, pred, rel)
                prompt = jinja_template.render(
                    sentence=sentence_item.model_dump(),
                    predicate=pred.model_dump(),
                    relation=rel.model_dump(),
                    examples=examples,
                    label_definitions=label_definitions,
                    text_with_targets=text_with_arg_targeted,
                    sprl=True,
                    srl=False
                )
                if prepend_previous_prompt:
                    # Find previous prompt for this argument
                    found = False
                    for prompt_dict in item_dict['prompts']:
                        if prompt_dict['prediction_id'] == f'{sentence_item.sentence_id}|{pred.start_char}|{pred.end_char}':
                            previous_prompt = prompt_dict['prompt']
                            raw_output = prompt_dict['raw_output']
                            found = True
                            break
                    if not found:
                        logger.warning(f'Previous prompt not found for {prediction_id}. Skipping.')
                        continue
                    prompt = f'{previous_prompt}\n{raw_output}\n{prompt}'
                prompts[prediction_id] = prompt
    elif component.startswith('srl'):
        # One prompt per argument
        for pred, rels in sentence_item.relations:
            for rel in rels:
                prediction_id = f'{sentence_item.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}'
                if prediction_id in prompts.keys():
                    # logger.warning(f'Prediction ID {prediction_id} already exists. Skipping.')
                    continue
                
                text_with_arg_targeted = target_spans(sentence_item.text, pred, rel)
                role_definitions = get_roles_and_descs(pred.roleset_id)
                if role_definitions is None:
                    logger.debug(f'Roleset {pred.roleset_id} not found for prediction {prediction_id}. Skipping.')
                    continue
                prompt = jinja_template.render(predicate=pred.model_dump(), relation=rel.model_dump(),
                    examples=examples, text_with_targets=text_with_arg_targeted, role_definitions=role_definitions)
                if prepend_previous_prompt:
                    # Find previous prompt for this argument
                    found = False
                    if 'sprl' not in component: # srl alone
                        for prompt_dict in item_dict['prompts']:
                            if prompt_dict['prediction_id'] == f'{sentence_item.sentence_id}|{pred.start_char}|{pred.end_char}':
                                previous_prompt = prompt_dict['prompt']
                                raw_output = prompt_dict['raw_output']
                                found = True
                                break
                    else: # component is 'srl+sprl' so use sprl output/prediction_id check
                        for prompt_dict in item_dict['prompts']:
                            if prompt_dict['prediction_id'] == prediction_id:
                                previous_prompt = prompt_dict['prompt']
                                raw_output = prompt_dict['raw_output']
                                found = True
                                break
                    if not found:
                        logger.warning(f'Previous prompt not found for {prediction_id}. Skipping.')
                        continue
                    prompt = f'{previous_prompt}\n{raw_output}\n{prompt}'
                prompts[prediction_id] = prompt
    else:
        raise ValueError(f'Component {component} not recognized.')
    
    return prompts, item_dict

def get_one_arg_prompts_from_item(item_dict_with_gold_and_predicted: dict,
        include_oracle_sprl: bool = False, include_predicted_sprl: bool = False,
        template: str = "templates/template.jinja", **template_options) -> Tuple[dict, dict]:
    
    environment = jinja2.Environment()
    jinja_template = environment.from_string(open(template).read())

    # Handle template options
    examples = template_options.get('examples', None)
    system_message = template_options.get('system_message', None)
    prepend_system_message_to_prompt = template_options.get('prepend_system_message_to_prompt', False)
    sprl_subset = template_options.get('sprl_subset', None)
    srl = template_options.get('srl', False) # Whether or not SRL is the end of the pipeline; default is False (i.e. SPRL is the end instead)

    if include_oracle_sprl or include_predicted_sprl:
        if sprl_subset:
            label_definitions = {k: v for k, v in SPR_DEFINITIONS.items() if k in sprl_subset}
            pstring = '-'.join(sprl_subset) if len(sprl_subset) < 5 else 'intersection'
        else:
            label_definitions = SPR_DEFINITIONS
            pstring = 'all'

    if include_predicted_sprl:
        if 'sprl' in item_dict_with_gold_and_predicted['predicted'].keys() and pstring in item_dict_with_gold_and_predicted['predicted']['sprl'].keys():
            predicted_sentence_item = item_dict_with_gold_and_predicted['predicted']['sprl'][pstring]['item']
        else:
            logger.warning(f'No predicted item found for SPRL ({pstring}). Skipping.')
            return [], item_dict_with_gold_and_predicted
        
    gold_sentence_item = item_dict_with_gold_and_predicted['gold']['item']
    prompts = {}

    if not gold_sentence_item:
        logger.warning('No gold SentenceItem found in the item dict. No prompt will be created.')
        return [], item_dict_with_gold_and_predicted

    if not system_message:
        if srl:
            system_message = f'You\'re tasked with using Semantic Role Labeling (SRL) to identify Predicates, Arguments, and their Semantic Roles from text.'
        else:
            system_message = f'"You\'re tasked with using Semantic Proto-role Labeling (SPRL) to identify Predicates, Arguments, and their SPRL Properties from text.'

    
    for pred, rels in gold_sentence_item.relations:
        for rel in rels:
            prediction_id = f'{gold_sentence_item.sentence_id}|{pred.start_char}|{pred.end_char}|{rel.arg.start_char}|{rel.arg.end_char}'
            if prediction_id in prompts.keys():
                # logger.warning(f'Prediction ID {prediction_id} already exists. Skipping.')
                continue

            if include_oracle_sprl and not rel.has_sprl_label():
                logger.error(f'Cannot include oracle SPRL labels on gold SentenceItem {gold_sentence_item.sentence_id} because SPRL annotations were not found for relation {rel}. Skipping.')
                return [], item_dict_with_gold_and_predicted

            label_definitions_with_negations = None
            if include_oracle_sprl or include_predicted_sprl:
                # get all sprl properties that are positive
                label_definitions_with_negations = {} # subset of label_definitions such that only positive spr properties are included
                if include_oracle_sprl:
                    for k, v in label_definitions.items():
                        if k not in rel.sprl_label.keys():
                            logger.error(f'Oracle SPRL label {k} not found in relation {rel} in gold SentenceItem {gold_sentence_item.sentence_id}. Skipping.')
                            return [], item_dict_with_gold_and_predicted
                        if rel.sprl_label[k]:
                            label_definitions_with_negations[k] = v
                        else:
                            label_definitions_with_negations[f'negative_{k}'] = NEGATED_DEFINITIONS[k]
                if include_predicted_sprl:
                    # find matching predicted relation
                    matching_predicted_relation = None
                    for predicted_predicate, predicted_relations in predicted_sentence_item.relations:
                        if pred == predicted_predicate:
                            for predicted_relation in predicted_relations:
                                if rel.arg == predicted_relation.arg:
                                    matching_predicted_relation = predicted_relation
                                    break
                            if matching_predicted_relation:
                                break
                    if matching_predicted_relation is None:
                        logger.warning(f'No matching predicted relation found for gold relation {rel} in gold SentenceItem {gold_sentence_item.sentence_id}. Skipping.')
                        continue
                    for k, v in label_definitions.items():
                        if k not in matching_predicted_relation.sprl_label.keys():
                            logger.error(f'Predicted SPRL label {k} not found in relation {matching_predicted_relation} in predicted SentenceItem {predicted_sentence_item.sentence_id}. Skipping.')
                            return [], item_dict_with_gold_and_predicted
                        if matching_predicted_relation.sprl_label[k]:
                            label_definitions_with_negations[k] = v
                        else:
                            label_definitions_with_negations[f'negative_{k}'] = NEGATED_DEFINITIONS[k]
            text_with_arg_targeted = target_spans(gold_sentence_item.text, pred, rel)
            role_definitions = get_roles_and_descs(pred.roleset_id)
            if role_definitions is None:
                logger.debug(f'Roleset {pred.roleset_id} not found for prediction {prediction_id}. Skipping.')
                continue
            prompt = jinja_template.render(
                predicate=pred.model_dump(),
                relation=rel.model_dump(),
                examples=examples,
                text_with_targets=text_with_arg_targeted,
                role_definitions=role_definitions,
                label_definitions=label_definitions_with_negations
            )
            if prepend_system_message_to_prompt:
                prompt = f'{system_message}\n{prompt}'
            prompts[prediction_id] = prompt
    
    return prompts, item_dict_with_gold_and_predicted

def get_all_args_prompts_from_item(
        item_dict_with_gold_and_predicted: dict,
        template: str = "templates/all-args.jinja",
        **template_options) -> Tuple[dict, dict]:
    
    environment = jinja2.Environment()
    jinja_template = environment.from_string(open(template).read())

    # Handle template options
    examples = template_options.get('examples', None)
    system_message = template_options.get('system_message', None)
    prepend_system_message_to_prompt = template_options.get('prepend_system_message_to_prompt', False)
    sprl_subset = template_options.get('sprl_subset', None)
    include_sprl = template_options.get('include_sprl', False)

    label_definitions = None
    if include_sprl:
        if sprl_subset:
            label_definitions = {k: v for k, v in SPR_DEFINITIONS.items() if k in sprl_subset}
            pstring = '-'.join(sprl_subset) if len(sprl_subset) < 5 else 'intersection'
        else:
            label_definitions = SPR_DEFINITIONS
            pstring = 'all'
        
    gold_sentence_item = item_dict_with_gold_and_predicted['gold']['item']
    prompts = {}

    if not gold_sentence_item:
        logger.warning('No gold SentenceItem found in the item dict. No prompt will be created.')
        return [], item_dict_with_gold_and_predicted

    if not system_message:
        system_message = f'You\'re tasked with using Semantic Role Labeling (SRL) to identify Predicates, Arguments, and their Semantic Roles from text.'

    for pred, rels in gold_sentence_item.relations:
        prediction_id = f'{gold_sentence_item.sentence_id}|{pred.start_char}|{pred.end_char}'

        if prediction_id in prompts.keys():
            # logger.warning(f'Prediction ID {prediction_id} already exists. Skipping.')
            continue
        
        text_with_targets = target_all_relations(gold_sentence_item.text, pred, rels)
        role_definitions = get_roles_and_descs(pred.roleset_id)
        if role_definitions is None:
            logger.debug(f'Roleset {pred.roleset_id} not found for prediction {prediction_id}. Skipping.')
            continue
        prompt = jinja_template.render(
            predicate=pred.model_dump(),
            relations=[rel.model_dump() for rel in rels],
            original_text=gold_sentence_item.text,
            text_with_targets=text_with_targets,
            role_definitions=role_definitions,
            label_definitions=label_definitions,
            include_sprl=include_sprl,
            examples=examples
        )
        if prepend_system_message_to_prompt:
            prompt = f'{system_message}\n{prompt}'
        prompts[prediction_id] = prompt
    
    return prompts, item_dict_with_gold_and_predicted

def load_llama_sprl_prompts(
        path_to_data: str = f'{PATH_TO_THIS_REPO}/corpus/sprl/1.test.all.pkl',
        path_to_template_directory: str = f'{PATH_TO_THIS_REPO}/templates/true-false-na'
) -> dict:
    '''Load the SPRL items from gold file and generate prompts for them'''
    # Load items from corpus and initialize the items dict for the first time
    f = open(path_to_data, 'rb')
    data = pickle.load(f)
    sentence_items = [SentenceItem(**item) for item in data]
    items = {item.sentence_id: {'gold': {'item': item}} for item in sentence_items}

    # Jinja environment
    environment = jinja2.Environment()

    # Walk through template directory and generate prompts
    prompts = {}
    for spr_id in tqdm(items.keys(), total=len(items), desc=f'Generating prompts'):
        gold_item = items[spr_id]['gold']['item']
        # Get each relation
        for predicate, relations in gold_item.relations:
            for relation in relations:
                text_with_targets = target_spans(gold_item.text, predicate, relation, tags='[]')
                for spr_property in relation.sprl_label.keys():
                    prompt_id = f'{spr_id}|{predicate.start_char}|{predicate.end_char}|{relation.arg.start_char}|{relation.arg.end_char}|{spr_property}'
                    template = f'{path_to_template_directory}/{spr_property}.jinja'
                    if os.path.exists(template):
                        jinja_template = environment.from_string(open(template).read())
                        prompts[prompt_id] = {'prompt': jinja_template.render(
                            predicate=predicate.model_dump(),
                            relation=relation.model_dump(),
                            text_with_targets=text_with_targets)
                        }
                        prompts[prompt_id]['gold'] = relation.sprl_label[spr_property]
                    else:
                        logger.warning(f'Template {template} not found. Skipping.')
    return prompts