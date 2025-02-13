import os
import pytz

from datetime import datetime
from typing import List, Union
from math import ceil

from ..data2prompts.items import SentenceItem

def make_datetime_str():
    your_time_zone = ''
    timez = pytz.timezone(your_time_zone)
    now = datetime.now()
    now = now.astimezone(tz=timez)
    return now.strftime("%b%d-%I%M%S%p%Z")

def prep_out_dir_for_experiments(out_dir: str,
        tstring_override: str = None):
    tz_string = make_datetime_str()
    subdirs = out_dir.strip().split('/')
    for i in range(1, len(subdirs)):
        if '/'.join(subdirs[:i]) == '':
            continue
        if not os.path.exists('/'.join(subdirs[:i])):
            os.mkdir('/'.join(subdirs[:i]))
    if out_dir[-1] == '/':
        out_dir = out_dir[:-1]

    # Make main dir for experiment [timestamped dir for this experiment]
    if tstring_override:
        tz_string = tstring_override
    if not os.path.exists(f'{out_dir}/{tz_string}/'):
        os.mkdir(f'{out_dir}/{tz_string}/')
    out_dir = f'{out_dir}/{tz_string}'

    return out_dir, tz_string

def ensure_output_path_exists(
        out_dir: str
):
    '''
    create directories for output files
    '''
    subdirs = out_dir.strip().split('/')
    for i in range(0, len(subdirs)+1):
        if '/'.join(subdirs[:i]) == '':
            continue
        if not os.path.exists('/'.join(subdirs[:i])):
            os.mkdir('/'.join(subdirs[:i]))
    if out_dir[-1] == '/':
        out_dir = out_dir[:-1]
    return out_dir

def two_way_split(
    sentences: List[Union[dict, SentenceItem]],
    test_p: float,
    *,
    limit: int = 0
):
    '''
    Split a dataset into a train and test set

    Args:
    - sentences: list of SentenceItem objects
    - test_p: proportion of the dataset to be used for testing
    - limit: maximum number of sentences to use
    '''
    if limit:
        sentences = sentences[:limit]
    test_n = ceil(len(sentences) * test_p)
    train_docs = sentences[test_n:]
    test_docs = sentences[:test_n]

    return train_docs, test_docs
