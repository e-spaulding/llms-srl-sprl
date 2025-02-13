import spacy
import logging

from typing import Tuple, Union
from nltk.corpus import propbank

logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm')

ROLE_HISTORIES = {
    'camp.01': 'camp_out.01',
    'scoop.02': 'scoop_up.02',
    'leave.01': 'leave.11',
    'pull.02': 'pull_out.02',
    'flare.01': 'flare_up.01',
    'stop.02': 'stop_by.02',
    'firm.02': 'firm_up.02',
    'dry.01': 'dry_up.01',
    'hold.11': 'hold_up.11',
    'bounce.02': 'bounce_back.02',
    'turn.15': 'turn_up.15',
    'wind.02': 'wind_up.02'
} # role histories from PB2.1 --> PB 3.1 
# https://github.com/propbank/propbank-documentation/blob/master/role-histories/pb2_1_5_to_pb3_1.tsv
# all args map to the same args in the new roleset

def get_roleset_ids(
        sentence_text: str,
        predicate_start_char: int,
        predicate_end_char: int   
):
    # Lemmatize the text
    doc = nlp(sentence_text)
    predicate_span = doc.char_span(predicate_start_char, predicate_end_char)
    if predicate_span is None:
        # logger.debug(f'Predicate span {predicate_start_char}-{predicate_end_char} not found in sentence.')
        return []
    predicate_lemma = predicate_span.lemma_

    # Try to find the roleset IDs using the lemma as the baseform
    if ' ' not in predicate_lemma:
        try:
            rolesets = propbank.rolesets(predicate_lemma)
        except ValueError:
            # logger.debug(f'Roleset {predicate_lemma} not found. No rolesets returned.')
            return []
    else:
        predicate_lemma = predicate_lemma.split(' ')[0]
        try:
            rolesets = propbank.rolesets(predicate_lemma)
        except ValueError:
            # logger.debug(f'Roleset {predicate_lemma} not found. No rolesets returned.')
            return []
    
    roleset_info = []
    for roleset in rolesets:
        roles = []
        roledescs = []
        for role in roleset.findall('roles/role'):
            roles.append(f'ARG{role.attrib["n"]}')
            roledescs.append(role.attrib['descr'])
        roleset_info.append({
            'id': roleset.attrib['id'],
            'desc': roleset.attrib['name'],
            'roles': [
                {
                    'role': role,
                    'desc': desc
                } for role, desc in zip(roles, roledescs)
            ]
        })
        
    return roleset_info

def get_roles_and_descs(
    roleset_id: str
):
    # Check if it's .XX
    if roleset_id.endswith('.XX'):
        return None
    # Check if it's an old role
    if roleset_id in ROLE_HISTORIES.keys():
        roleset_id = ROLE_HISTORIES[roleset_id]
    # Find the roleset in PB
    try:
        roleset = propbank.roleset(roleset_id)
    except ValueError:
        # Search through every single roleset
        roleset = None
        for rset in propbank.rolesets():
            if rset.attrib['id'] == roleset_id:
                roleset = rset
                break
        if roleset is None:
            # logger.debug(f'Roleset {roleset_id} not found.')
            return None
    roles_and_descs = {}
    for role in roleset.findall('roles/role'):
        if role.attrib["n"] == 'm':
            roles_and_descs['ARGM'] = role.attrib['descr']
        else:
            roles_and_descs[f'ARG{role.attrib["n"]}'] = role.attrib['descr']
    return roles_and_descs

def roleset_options_to_string(
        sentence_text: str,
        predicate_start_char: int,
        predicate_end_char: int
) -> Tuple[str, Union[str, None]]:
    '''
    Given a sentence text, predicate start and end character indices, this function produces a pretty string giving
    all eligible roleset options

    Parameters
    ----------
    sentence_text : str
        The text of the sentence
    predicate_start_char : int
        The start character index of the predicate
    predicate_end_char : int
        The end character index of the predicate
    '''
    roleset_info = get_roleset_ids(sentence_text, predicate_start_char, predicate_end_char)
    if len(roleset_info) == 0:
        return None, None
    if len(roleset_info) == 1:
        singular_roleset_option = roleset_info[0]['id']
    else:
        singular_roleset_option = None
    returnable = ''
    for roleset in roleset_info:
        roleset_id = roleset['id']
        roleset_desc = roleset['desc']
        returnable += f'{roleset_id}: {roleset_desc}\n'
        if roleset['roles'] is None:
            continue
        for role in roleset['roles']:
            returnable += f'\t{role["role"]}: {role["desc"]}\n'
    return returnable, singular_roleset_option

def get_roleset_ids_from_predicate_lemma(predicate_lemma: str):
    if ' ' not in predicate_lemma:
        try:
            rolesets = propbank.rolesets(predicate_lemma)
        except ValueError:
            # logger.debug(f'Roleset {predicate_lemma} not found. No rolesets returned.')
            return []
    else:
        query = predicate_lemma.split(' ')[0]
        lemma_with_underscore = predicate_lemma.replace(' ', '_')
        try:
            rolesets = propbank.rolesets(predicate_lemma)
        except ValueError:
            # logger.debug(f'Roleset {predicate_lemma} not found. No rolesets returned.')
            return []
        roleset_info = []
        for roleset in rolesets:
            roles = []
            roledescs = []
            id_ = roleset.attrib['id']
            # Skip the roleset if it does not match the lemma with underscore
            if id_.startswith(lemma_with_underscore):
                for role in roleset.findall('roles/role'):
                    roles.append(f'ARG{role.attrib["n"]}')
                    roledescs.append(role.attrib['descr'])
                roleset_info.append({
                    'id': roleset.attrib['id'],
                    'desc': roleset.attrib['name'],
                    'roles': [
                        {
                            'role': role,
                            'desc': desc
                        } for role, desc in zip(roles, roledescs)
                    ]
                })
        return roleset_info
    
    roleset_info = []
    for roleset in rolesets:
        roles = []
        roledescs = []
        for role in roleset.findall('roles/role'):
            roles.append(f'ARG{role.attrib["n"]}')
            roledescs.append(role.attrib['descr'])
        roleset_info.append({
            'id': roleset.attrib['id'],
            'desc': roleset.attrib['name'],
            'roles': [
                {
                    'role': role,
                    'desc': desc
                } for role, desc in zip(roles, roledescs)
            ]
        })
        
    return roleset_info