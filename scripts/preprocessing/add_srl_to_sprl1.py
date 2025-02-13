import re
import pickle

from nltk.corpus import propbank

from ..data2prompts.items import SentenceItem
from ..util.util import PATH_TO_THIS_REPO
from ..output_interpretation.utils import normalize_srl_label

# Loading WSJ from Conll12
print('Loading WSJ annotations via ontonotes...')
all_conll12_wsj_items = {}

paths = [
    f'{PATH_TO_THIS_REPO}/corpus/conll12/train.pkl',
    f'{PATH_TO_THIS_REPO}/corpus/conll12/dev.pkl',
    f'{PATH_TO_THIS_REPO}/corpus/conll12/test.pkl',
    f'{PATH_TO_THIS_REPO}/corpus/conll12/conll12-test.pkl'
]

for path in paths:
    f = open(path, 'rb')
    data = pickle.load(f)
    for item in data:
        if 'ontonotes/nw/wsj/' in item['sentence_id']:
            all_conll12_wsj_items[item['sentence_id']] = SentenceItem(**item)
    f.close()

# Load PropBank
print('Loading PropBank...')
pb_instances = propbank.instances()
pb_dict = {}
for inst in pb_instances:
    if inst.fileid not in pb_dict.keys():
        pb_dict[inst.fileid] = {}
        pb_dict[inst.fileid][inst.sentnum] = {inst.wordnum: [inst]}
    else:
        if inst.sentnum not in pb_dict[inst.fileid].keys():
            pb_dict[inst.fileid][inst.sentnum] = {inst.wordnum: [inst]}
        else:
            if inst.wordnum not in pb_dict[inst.fileid][inst.sentnum].keys():
                pb_dict[inst.fileid][inst.sentnum][inst.wordnum] = [inst]
            else:
                pb_dict[inst.fileid][inst.sentnum][inst.wordnum].append(inst)
print('PropBank loaded.')

def get_matching_instances_from_propbank(sentence_id, pred_token):
    filenum, sentnum = sentence_id.split('_')
    sentnum = int(sentnum)
    fileid = 'wsj_' + filenum + '.mrg'

    if fileid not in pb_dict.keys():
        return None
    if sentnum not in pb_dict[fileid].keys():
        return None
    if pred_token not in pb_dict[fileid][sentnum].keys():
        return None
    return pb_dict[fileid][sentnum][pred_token]

def get_pb_role_from_wordum(sentence_id, pred_token, wordnum):
    insts = get_matching_instances_from_propbank(sentence_id, pred_token)
    if insts is None or len(insts) == 0:
        return None
    for inst in insts:
        for (argloc, argid) in inst.arguments:
            if ',' in str(argloc) or '*' in str(argloc):
                locs = re.split(',|\*', str(argloc))
                for loc in locs:
                    that_wordnum = loc.split(':')[0]
                    if int(that_wordnum) == wordnum:
                        return argid
            try:
                that_wordnum = argloc.wordnum
            except AttributeError:
                argloc = argloc.pieces[-1]
                that_wordnum = argloc.wordnum
            if that_wordnum == wordnum:
                return argid
    return None

def get_token_num_from_span(sentence, start_char):
    chars_seen = 0
    tokenized_sentence = sentence.split(" ")
    for i, token in enumerate(tokenized_sentence):
        if start_char == 0:
            return i
        chars_seen += len(token)
        if chars_seen >= start_char:
            return i
        else:
            chars_seen += 1 # whitespace

def get_role_desc(roleset, role):
    try:
        roleset = propbank.roleset(roleset)
    except:
        print(f'Roleset {roleset} not found.')
        roleset = roleset.replace('.XX', '.01')
        print(f'Trying to find {roleset} instead.')
        roleset = propbank.roleset(roleset)
    n = role[3]
    for arg in roleset.findall('roles/role'):
        if arg.attrib['n'] == n:
            return arg.attrib['descr']
    return None

def find_matching_conll12_item(item):
    four_digit, sentnum = item.sentence_id.split('_')
    # create id needed to find conll12 item
    first_two_digits = four_digit[:2]
    conll12_id_to_find = f'ontonotes/nw/wsj/{first_two_digits}/wsj_{four_digit}-{sentnum}'
    if conll12_id_to_find in all_conll12_wsj_items.keys():
        return all_conll12_wsj_items[conll12_id_to_find]
    else:
        return None

def fuzzy_predicate_match_with_conll12(spr_predicate, conll12_predicate):
    spr_predicate.set_compare_roleset_ids(False)
    conll12_predicate.set_compare_roleset_ids(False)

    if spr_predicate == conll12_predicate:
        # easy enough
        return True
    
    # if they don't match, check discrepancies between start and end chars
    if spr_predicate.start_char == conll12_predicate.start_char and spr_predicate.end_char == conll12_predicate.end_char:
        return True
    
    if spr_predicate.start_char == conll12_predicate.start_char:
        if len(spr_predicate.text) > len(conll12_predicate.text):
            if spr_predicate.text.startswith(conll12_predicate.text):
                return True
        else:
            if conll12_predicate.text.startswith(spr_predicate.text):
                return True
    if spr_predicate.end_char == conll12_predicate.end_char:
        if len(spr_predicate.text) > len(conll12_predicate.text):
            if spr_predicate.text.endswith(conll12_predicate.text):
                return True
        else:
            if conll12_predicate.text.endswith(spr_predicate.text):
                return True
            
    return False

def fuzzy_arg_match_with_conll12(spr_arg, conll12_arg):

    if spr_arg == conll12_arg:
        return True
    
    if spr_arg.start_char == conll12_arg.start_char and spr_arg.end_char == conll12_arg.end_char:
        return True
    
    if spr_arg.start_char == conll12_arg.start_char:
        if len(spr_arg.text) > len(conll12_arg.text):
            if spr_arg.text.startswith(conll12_arg.text):
                return True
        else:
            if conll12_arg.text.startswith(spr_arg.text):
                return True
    if spr_arg.end_char == conll12_arg.end_char:
        if len(spr_arg.text) > len(conll12_arg.text):
            if spr_arg.text.endswith(conll12_arg.text):
                return True
        else:
            if conll12_arg.text.endswith(spr_arg.text):
                return True
            
    return False

def search_pb1_for_predicate_and_relations(item, p, rs):
    # Get token num of predicate
    pred_token = get_token_num_from_span(item.text, p.start_char)
    instances = get_matching_instances_from_propbank(item.sentence_id, pred_token)
    if instances is None:
        print(f'No instances found for predicate {pred_token} in sentence {item.sentence_id} for PB1')
        return
    else:
        roleset = instances[0].roleset
        p.roleset_id = roleset
        for pp in item.predicates:
            if pp.start_char == p.start_char and pp.end_char == p.end_char:
                pp.roleset_id = roleset
        for r in rs:
            # Get token num of argument
            arg_token = get_token_num_from_span(item.text, r.arg.start_char)
            # Find PB role
            pb_role = get_pb_role_from_wordum(item.sentence_id, pred_token, arg_token)
            if pb_role is not None:
                n = pb_role[3]
                reformatted_pb_role = f'ARG{n.upper()}'
                r.srl_label = reformatted_pb_role
            else:
                print(r)
                raise ValueError(f'PB role not found for predicate {pred_token} and arg {arg_token} in sentence {item.sentence_id}')

def search_pb1_for_relation(item, p, r):
    # Find matching instances
    pred_token = get_token_num_from_span(item.text, p.start_char)
    instances = get_matching_instances_from_propbank(item.sentence_id, pred_token)
    if instances is None:
        print(f'No instances found for predicate {pred_token} in sentence {item.sentence_id} for PB1')
        return
    else:
        # Get token num of argument
        arg_token = get_token_num_from_span(item.text, r.arg.start_char)
        # Find PB role
        pb_role = get_pb_role_from_wordum(item.sentence_id, pred_token, arg_token)
        if pb_role is not None:
            n = pb_role[3]
            reformatted_pb_role = f'ARG{n.upper()}'
            r.srl_label = reformatted_pb_role
        else:
            print(r)
            raise ValueError(f'PB role not found for predicate {pred_token} and arg {arg_token} in sentence {item.sentence_id}')


combos = [
    ['all']
]
splits = ['train', 'dev', 'test']
dataset = '1'

for combo in combos:
    for split in splits:
        labelstr = '-'.join(combo)
        path_to_data = f'{PATH_TO_THIS_REPO}/corpus/sprl-with-na/{dataset}.{split}.{labelstr}.pkl'

        # Load items from corpus
        f = open(path_to_data, 'rb')
        data = pickle.load(f)
        sentence_items = [SentenceItem(**item) for item in data]
        f.close()

        new_data = []

        for i, item in enumerate(sentence_items):
            # Get matching conll12 item
            conll12_item = find_matching_conll12_item(item)
            for p, rs in item.relations:
                # Get matching conll12 item
                if conll12_item:
                    # find predicate
                    predicate_found = False
                    for conll12_p, conll12_rs in conll12_item.relations:
                        if fuzzy_predicate_match_with_conll12(p, conll12_p):
                            predicate_found = True
                            p.roleset_id = conll12_p.roleset_id
                            for pp in item.predicates:
                                if pp.start_char == p.start_char and pp.end_char == p.end_char:
                                    pp.roleset_id = conll12_p.roleset_id
                            for r in rs:
                                # find matching conll12 relation
                                rel_found = False
                                for conll12r in conll12_rs:
                                    if fuzzy_arg_match_with_conll12(r.arg, conll12r.arg):
                                        rel_found = True
                                        r.srl_label = normalize_srl_label(conll12r.srl_label)
                                        break
                                if not rel_found:
                                    print(f'Relation not found in conll12 item for sentence {item.sentence_id}, looking in PB1...')
                                    search_pb1_for_relation(item, p, r)
                            break
                    if not predicate_found:
                        print(f'Predicate not found in conll12 item for sentence {item.sentence_id}, looking in PB1...')
                        search_pb1_for_predicate_and_relations(item, p, rs)
                else:
                    print(f'Conll12 item not found for sentence {item.sentence_id}, looking in PB1...')
                    search_pb1_for_predicate_and_relations(item, p, rs)
            if i % 1000 == 0:
                print(f'{dataset}.{split}.{labelstr} progress: {i+1} / {len(sentence_items)}')
            new_data.append(item.model_dump())

            output_filename = f'{PATH_TO_THIS_REPO}/corpus/srl+sprl-with-na/{dataset}.{split}.{labelstr}.pkl'
            with open(output_filename, 'wb') as output_file:
                pickle.dump(new_data, output_file)