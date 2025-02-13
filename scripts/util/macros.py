PATH_TO_THIS_REPO = ''
PATH_TO_PB_GUIDELINES = 'templates/propbank-guidelines.txt'

SPR_DEFINITIONS = {
    'instigation': 'ARG caused PRED to happen',
    'volition': 'ARG chose to be involved in PRED',
    'awareness': 'ARG was aware of being involved in PRED',
    'sentient': 'ARG was sentient',
    'change_of_location': 'ARG changed location during PRED',
    'exists_as_physical': 'ARG existed as a physical object',
    'existed_before': 'ARG existed before PRED began',
    'existed_during': 'ARG existed during PRED',
    'existed_after': 'ARG existed after PRED stopped',
    'change_of_possession': 'ARG changed possession during PRED',
    'changes_possession': 'ARG changed possession during PRED',
    'change_of_state': 'ARG was altered or somehow changed during or by the end of PRED',
    'stationary': 'ARG was stationary during PRED',
    'location_of_event': 'ARG described the location of PRED',
    'makes_physical_contact': 'ARG made physical contact with someone or something else involved in PRED',
    'was_used': 'ARG was used in carrying out PRED',
    'manipulated_by_another': 'ARG was used in carrying out PRED',
    'predicate_changed_argument': 'PRED caused a change in ARG',
    'was_for_benefit': 'PRED happened for the benefit of ARG',
    'partitive': 'Only a part of portion of ARG was involved in PRED',
    'change_of_state_continuous': 'The change in ARG happened throughout PRED'
} 
NEGATED_DEFINITIONS = {
    'instigation': 'ARG did not cause PRED to happen',
    'volition': 'ARG did not choose to be involved in PRED',
    'awareness': 'ARG was not aware of being involved in PRED',
    'sentient': 'ARG was not sentient',
    'change_of_location': 'ARG did not change location during PRED',
    'exists_as_physical': 'ARG did not exist as a physical object',
    'existed_before': 'ARG did not exist before PRED began',
    'existed_during': 'ARG did not exist during PRED',
    'existed_after': 'ARG did not exist after PRED stopped',
    'change_of_possession': 'ARG did not change possession during PRED',
    'changes_possession': 'ARG did not change possession during PRED',
    'change_of_state': 'ARG was not altered or somehow changed during or by the end of PRED', # consider rewording
    'stationary': 'ARG was not stationary during PRED',
    'location_of_event': 'ARG did not describe the location of PRED',
    'makes_physical_contact': 'ARG did not make physical contact with someone or something else involved in PRED',
    'was_used': 'ARG was not used in carrying out PRED',
    'manipulated_by_another': 'ARG was not used in carrying out PRED',
    'predicate_changed_argument': 'PRED did not cause a change in ARG',
    'was_for_benefit': 'PRED did not happen for the benefit of ARG',
    'partitive': 'It is not true that only a part of portion of ARG was involved in PRED',
    'change_of_state_continuous': 'The change in ARG did not happen throughout PRED'
}
LABELS_INTERSECTION = [
    'instigation', 
    'volition', 
    'awareness', 
    'sentient', 
    'change_of_location',
    'existed_before',
    'existed_during',
    'existed_after',
    'change_of_possession',
    'change_of_state'
]
LABELS_UNION = [
    'instigation',
    'volition',
    'awareness',
    'sentient',
    'change_of_location',
    'exists_as_physical',
    'existed_before',
    'existed_during',
    'existed_after',
    'change_of_possession',
    'change_of_state',
    'stationary',
    'location_of_event',
    'makes_physical_contact',
    'manipulated_by_another',
    'predicate_changed_argument',
    'was_for_benefit',
    'partitive',
    'change_of_state_continuous'
] # diff between this & the definitions above is that this one collapses two properties w/ different names but same definition together (was_used/manipulated, poss.)
SPR1_LABELS = [
    'instigation',
    'volition',
    'awareness',
    'sentient',
    'change_of_location',
    'exists_as_physical',
    'existed_before',
    'existed_during',
    'existed_after',
    'changes_possession',
    'change_of_state',
    'stationary',
    'location_of_event',
    'makes_physical_contact',
    'manipulated_by_another',
    'predicate_changed_argument'
]
SPR2_LABELS = [
    'instigation',
    'volition',
    'awareness',
    'sentient',
    'change_of_location',
    'existed_before',
    'existed_during',
    'existed_after',
    'change_of_possession',
    'change_of_state',
    'was_used',
    'was_for_benefit',
    'partitive',
    'change_of_state_continuous'
]

PREDICATE_REGEX = '(?P<tagged_pred>(<PRED>(?P<pred>(\s|\(|\)|-|\")*(\w(\s|\(|\)|-|\"|&|%|:|,|[0-9]|\.|\')*)+)<\/PRED>))'
ARGUMENT_REGEX = f'(?P<tagged_arg><ARG>(?P<arg>(\s|\(|\)|-|\"|\*|\.|\%|\')*(\w(\s|\(|\)|-|\"|&|%|:|,|\'\'|`|\.|[0-9]|\*|\')*)+)<\/ARG>)'
SRL_REGEX = '(?P<argstring>ARG)(-)?(?P<arg_number>0|1|2|3|4|5|M|C)(-)?(?P<function_tag>[A-Z]+)?'

API_KEY = ''