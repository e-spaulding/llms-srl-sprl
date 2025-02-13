from pydantic import BaseModel
from typing import List, Tuple

class SpanItem(BaseModel):
    text: str
    start_char: int
    end_char: int
    compare_text: bool = True
    '''
    Class to represent a span of text within a sentence, e.g.:

    SpanItem(text='the cat', start_char=0, end_char=7) for the sentence
    'the cat sat on the mat'
    '''

    def set_compare_text(self, mode: bool):
        self.compare_text = mode

    def __hash__(self):
        if self.compare_text:
            return hash((self.text, self.start_char, self.end_char))
        else:
            return hash((self.start_char, self.end_char))

    def __eq__(self, other):
        if self.compare_text:
            return (
                self.text == other.text
                and self.start_char == other.start_char
                and self.end_char == other.end_char
            )
        else:
            return (
                self.start_char == other.start_char
                and self.end_char == other.end_char
            )


class PredicateItem(SpanItem):
    roleset_id: str = ""
    compare_roleset_ids: bool = False
    '''
    Class to represent a predicate in a sentence, e.g.:

    PredicateItem(text='sat', start_char=8, end_char=11, roleset_id='sit.01') for the sentence
    'the cat sat on the mat'
    '''

    def set_compare_roleset_ids(self, mode: bool):
        self.compare_roleset_ids = mode

    def __str__(self):
        if self.roleset_id:
            return f"Predicate '{self.text}'/{self.roleset_id} [{self.start_char}-{self.end_char})"
        else:
            return f"Predicate '{self.text}' [{self.start_char}-{self.end_char})"

    def __hash__(self):
        if self.compare_roleset_ids and self.compare_text:
            return hash((self.text, self.start_char, self.end_char, self.roleset_id))
        elif self.compare_roleset_ids:
            return hash((self.start_char, self.end_char, self.roleset_id))
        elif self.compare_text:
            return hash((self.text, self.start_char, self.end_char))
        else:
            return hash((self.start_char, self.end_char))

    def __eq__(self, other):
        if self.compare_roleset_ids and self.compare_text:
            return (
                self.text == other.text
                and self.start_char == other.start_char
                and self.end_char == other.end_char
                and self.roleset_id == other.roleset_id
            )
        elif self.compare_roleset_ids:
            return (
                self.start_char == other.start_char
                and self.end_char == other.end_char
                and self.roleset_id == other.roleset_id
            )
        elif self.compare_text:
            return (
                self.text == other.text
                and self.start_char == other.start_char
                and self.end_char == other.end_char
            )
        else:
            return (
                self.start_char == other.start_char
                and self.end_char == other.end_char
            )
        
class RelationItem(BaseModel):
    arg: SpanItem
    srl_label: str = ""
    sprl_label: dict = {}
    compare_sprl_labels: bool = True
    '''
    Class to represent an SRL, SPRL, or SRL+SPRL annotated arg in a sentence
    '''

    def has_srl_label(self):
        return self.srl_label != ""
    
    def has_sprl_label(self):
        return len(self.sprl_label) > 0
    
    def set_compare_sprl_labels(self, mode: bool):
        self.compare_sprl_labels = mode

    def set_compare_text(self, mode: bool):
        self.arg.set_compare_text(mode)

    def __str__(self):
        span_string = f"Argument '{self.arg.text}' [{self.arg.start_char}-{self.arg.end_char})"
        if self.srl_label and len(self.sprl_label) > 0:
            srl_string = f"\t\tSemantic role: {self.srl_label}"
            sprl_string = "\n".join([f"\t\t{label}: {value}" for label, value in self.sprl_label.items()])
            return f"{span_string}\n{srl_string}\n{sprl_string}"
        elif self.srl_label:
            return f"{span_string}\n\t\tSemantic role: {self.srl_label}"
        elif len(self.sprl_label) > 0:
            sprl_string = "\n".join([f"\t\t{label}: {value}" for label, value in self.sprl_label.items()])
            return f"{span_string}\n{sprl_string}"
        else:
            return span_string

    def __hash__(self):
        if self.srl_label and len(self.sprl_label) > 0 and self.compare_sprl_labels:
            hashable_sprl_label = tuple(sorted(self.sprl_label.items()))
            return hash((self.arg, self.srl_label, hashable_sprl_label))
        elif self.srl_label:
            return hash((self.arg, self.srl_label))
        elif len(self.sprl_label) > 0 and self.compare_sprl_labels:
            hashable_sprl_label = tuple(sorted(self.sprl_label.items()))
            return hash((self.arg, hashable_sprl_label))
        else:
            return hash(self.arg)
    
    def __eq__(self, other):
        if self.srl_label and len(self.sprl_label) > 0 and self.compare_sprl_labels:
            hashable_sprl_label_self = tuple(sorted(self.sprl_label.items()))
            hashable_sprl_label_other = tuple(sorted(other.sprl_label.items()))
            return (
                self.arg == other.arg
                and self.srl_label == other.srl_label
                and hashable_sprl_label_self == hashable_sprl_label_other
            )
        elif self.srl_label:
            return self.arg == other.arg and self.srl_label == other.srl_label
        elif len(self.sprl_label) > 0 and self.compare_sprl_labels:
            hashable_sprl_label_self = tuple(sorted(self.sprl_label.items()))
            hashable_sprl_label_other = tuple(sorted(other.sprl_label.items()))
            return self.arg == other.arg and hashable_sprl_label_self == hashable_sprl_label_other
        else:
            return self.arg == other.arg
    
class SentenceItem(BaseModel):
    sentence_id: str
    text: str
    predicates: List[PredicateItem]
    relations: List[Tuple[PredicateItem, List[RelationItem]]]
    '''
    Class to represent a sentence with SRL, SPRL, or SRL+SPRL annotations
    '''

    def set_predicate_compare_roleset_ids(self, mode: bool):
        for p in self.predicates:
            p.set_compare_roleset_ids(mode)
        for p, rs in self.relations:
            p.set_compare_roleset_ids(mode)
    
    def set_relation_compare_sprl_labels(self, mode: bool):
        for p, rs in self.relations:
            for r in rs:
                r.set_compare_sprl_labels(mode)

    def set_compare_text(self, mode: bool):
        for p in self.predicates:
            p.set_compare_text(mode)
        for p, rs in self.relations:
            p.set_compare_text(mode)
            for r in rs:
                r.set_compare_text(mode)
    
    def reset_compare_to_default(self):
        self.set_predicate_compare_roleset_ids(False)
        self.set_relation_compare_sprl_labels(True)
        self.set_compare_text(True)

    def delete_relation(self, predicate: PredicateItem, relation: RelationItem):
        for p, rs in self.relations:
            if p == predicate:
                rs.remove(relation)
                return
    
    def delete_predicate(self, predicate: PredicateItem):
        self.predicates.remove(predicate)
        for p, rs in self.relations:
            if p == predicate:
                self.relations.remove((p, rs))
                return

    def __str__(self):
        predicates = ", ".join([str(p) for p in self.predicates])
        relation_string = ""
        for p, r in self.relations:
            relation_string += f"{p} relations:\n"
            relation_string += "\n".join([f"\t{rel}" for rel in r])
            relation_string += "\n"
        return f"Sentence {self.sentence_id}: {self.text}\nPredicates: {predicates}\n{relation_string}"
    

class PredictedSentenceItem(SentenceItem):
    '''
    Class to represent a sentence with predicted SRL, SPRL, or SRL+SPRL annotations
    '''
    prediction_id: str = '?'

    def __str__(self):
        return f"Prediction {self.prediction_id} for {super().__str__()}"