import json
import logging
import os
import time
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

def aggregate_requests(
        prompts: dict,
        model_ver: str,
        system_message: str,
        output_path: str
):
    '''Aggregates OpenAI requests into a single .jsonl file for batch processing'''
    requests = [
        {
            'custom_id': prediction_id,
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': model_ver,
                'messages': [
                    {
                        'role': 'system',
                        'content': system_message
                    },
                    {
                        'role': 'user',
                        'content': prompts[prediction_id]
                    }
                ],
                'max_tokens': 1000
            }
        }
        for prediction_id in prompts.keys()
    ]
    requests = ensure_unique_ids(requests)
    with open(output_path, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
        f.close()

def send_batch(request_path: str, component: str, out_dir: str, config: dict) -> str:
    '''Sends batch of OpenAI requests to the API'''
    
    client = OpenAI(api_key=config['api_key'])

    if os.path.exists(os.path.join(out_dir, f'{component}-batch-output.jsonl')):
        logger.error(f'Batch output file {component}-batch-output.jsonl already exists. Skipping batch job.')
        return None

    with open(os.path.join(out_dir, request_path), 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose='batch'
        )
    
    batch_input_file_id = batch_input_file.id

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": f"srl/sprl batch from {request_path}"
        }
    )

    logger.info(f'Batch job {batch_job.id} sent. Status:')
    logger.info(batch_job)

    with open(f'{out_dir}/{component}-batch-info.txt', 'w') as f:
        f.write(f'Batch job {batch_job.id} sent. Status:\n')
        f.write(str(batch_job))
        f.close()

    return batch_job.id

def wait_for_batch(batch_id: str,
        client: OpenAI,
        interval: int = 300,
        cap: int = 6000): 
    '''Waits for a batch to complete and returns the finished batch'''
    counter = 0
    logger.info(f'Checking batch job {batch_id} status every {interval} seconds until a total of {cap} seconds have elapsed.')
    while(True):
        batch_job = client.batches.retrieve(batch_id)
        if batch_job.status == 'completed':
            return batch_job
        elif batch_job.status in ['failed', 'cancelled', 'cancelling', 'expired']:
            logger.error(f'Batch job {batch_id} returned status {batch_job.status}.')
            return None
        elif counter * interval > cap:
            logger.error(f'Batch job {batch_id} took too long to complete. Try running again later.')
            return None
        else:
            logger.info(f'Batch job {batch_id} still running. It has been roughly {counter * interval} seconds.')
            counter += 1
            time.sleep(interval)

def get_batch_output(batch_id: str,
        client: OpenAI,
        component: str,
        output_directory: str) -> str:
    '''Retrieves the output of a completed batch and saves it'''
    batch_job = client.batches.retrieve(batch_id)
    if batch_job.status == 'completed':
        file = client.files.content(batch_job.output_file_id)
        file_bytes = file.read()
        filename = f'{output_directory}/{component}-batch-output.jsonl'
        with open(filename, 'wb') as f:
            f.write(file_bytes)
            f.close()
            return filename
    else:
        return None

def ensure_unique_ids(
        requests: List[dict]
) -> List[dict]:
    '''Returns a list of OpenAI requests with duplicate custom_ids removed'''
    unique_ids = []
    unique_requests = []
    for req in requests:
        if req['custom_id'] not in unique_ids:
            unique_ids.append(req['custom_id'])
            unique_requests.append(req)
    return unique_requests
