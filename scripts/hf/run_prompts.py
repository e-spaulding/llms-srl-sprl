# Main libraries
import transformers
from transformers import AutoTokenizer
import torch 
import json 
import logging
import pickle
import pytz
from yaml import load, Loader
from time import process_time
from tqdm import tqdm
from datetime import datetime

# Before importing the transformers library, ensure the models are downloaded and pulled from the correct place
PATH_TO_CACHE = '/.cache/' # this is where the models are downloaded - set this to something else if desired, or comment this out
import os
os.environ['TRANSFORMERS_CACHE'] = PATH_TO_CACHE # deprecated
os.environ['HF_HOME'] = PATH_TO_CACHE # new

def make_datetime_str():
    your_time_zone = ''
    timez = pytz.timezone(your_time_zone)
    now = datetime.now()
    now = now.astimezone(tz=timez)
    return now.strftime("%b%d-%I%M%S%p%Z")

config_paths = [
    'configs/OLMo-2-7b-spr2.yaml',
    'configs/llama-3.2-3b-Instruct-spr2.yaml'
]

for config_path in config_paths:
    # Get parameters from config
    with open(config_path, 'r') as f:
        config = load(f, Loader=Loader)

    TODAYS_DATE = make_datetime_str()
    BATCH_SIZE = config['batch_size']
    MODEL_NAME = config['model_name']
    PATH_TO_PROMPTS = config['path_to_prompts']

    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Load prompts
    prompt_contexts_by_id = pickle.load(open(PATH_TO_PROMPTS, 'rb'))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    t_0 = process_time()

    # Initialize config and model
    model_name_for_saving = MODEL_NAME.split('/')[-1]

    llm_config = transformers.AutoConfig.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True)

    model_max_position_embeddings = llm_config.to_dict()['max_position_embeddings']

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,  
        device_map="auto",
        config=llm_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side='left',
        truncation=True,
        max_length=model_max_position_embeddings,
        add_prefix_space=False) 
    tokenizer.pad_token = tokenizer.eos_token # add padding token

    results_dict = {id_: {'prompt': prompt_contexts_by_id[id_], 'clean_output': '', 'full_output': ''} for id_ in prompt_contexts_by_id.keys()}
    results_dict['config'] = llm_config.to_dict()
    results_dict['config']['batch_size'] = BATCH_SIZE
    for b in tqdm(
        range(0, len(prompt_contexts_by_id), BATCH_SIZE), 
        desc='Running prompts', 
        total=len(prompt_contexts_by_id)//BATCH_SIZE):

        batch_ids = list(prompt_contexts_by_id.keys())[b:b + BATCH_SIZE]
        batch_prompts = [prompt_contexts_by_id[id_]['prompt'] + '\nResponse:' for id_ in batch_ids]

        tokenized_input = tokenizer(batch_prompts, padding=True, return_tensors='pt', truncation=True, max_length=model_max_position_embeddings)
        if len(tokenized_input[0]) >= model_max_position_embeddings:
            for id_ in batch_ids:
                results_dict[id_]['clean_output'] = 'input_exceeded_max_length'
                results_dict[id_]['full_output'] = 'input_exceeded_max_length'
            continue

        max_output_length = len(tokenized_input[0]) + 4

        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized_input['input_ids'].to(device), 
                attention_mask=tokenized_input['attention_mask'].to(device),
                pad_token_id=tokenizer.eos_token_id,
                max_length=max_output_length)
        
        full_outputs = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        clean_outputs = [full_outputs[i][len(tokenizer.decode((tokenized_input['input_ids'][i]), clean_up_tokenization_spaces=True, skip_special_tokens=True)):].strip() for i in range(len(full_outputs))]
        for id_ in batch_ids:
            results_dict[id_]['clean_output'] = clean_outputs[batch_ids.index(id_)]
            results_dict[id_]['full_output'] = full_outputs[batch_ids.index(id_)]
        if b % 500 == 0:
            # save output
            results_dict['batch_progress'] = b
            with open(f'experiments/{TODAYS_DATE}-{model_name_for_saving}-sprl-in-progress.json', 'w') as fp:
                json.dump(results_dict, fp, indent=2)

    t_1 = process_time()
    # save results
    results_dict['total_elapsed_time'] = t_1 - t_0

    with open(f'experiments/{TODAYS_DATE}-{model_name_for_saving}.json', 'w') as fp:
        json.dump(results_dict, fp, indent=2)