import re

def get_conll_sentence_lines(gold_conll_path):
    """
    Read the CoNLL formatted files

    Parameters
    ----------
        gold_conll_path: str
            The path to the conll format file
    Returns
    -------
        List[List[str]]
    """
    with open(gold_conll_path) as sf:
        conll_content = sf.read()

    n_2_split = re.split("\n[\s]*\n", conll_content)

    # Split sentences with \n\n
    conll_sentence_strs = [s for s in n_2_split if s.strip()]

    # Split tokens with \n for each sentence
    conll_sentence_lines = [s.split("\n") for s in conll_sentence_strs if s.strip()]

    # Convert multi-spaces to \t and split column of each token
    conll_sentence_lines_cols = [
        [
            re.sub(" +", "\t", line.strip()).split("\t")
            for line in sentence
            if line.strip()
        ]
        for sentence in conll_sentence_lines
    ]

    return conll_sentence_lines_cols


def get_pred_relations_2005(conll_sentence):
    """
    Convert the 2005 style conll format of srl into spacy doc
    """
    sentence_tokens = []
    # the column index where the srl relations start
    srl_token_start = 6
    curr_doc_sent_id = None
    relations = []
    rel_stacks = []

    for tok_id, line in enumerate(conll_sentence):
        word, pos, cp, ner, sense_num, sense = line[:srl_token_start]
        sentence_tokens.append(word)

        # each column contain the proposition relations
        srl_rels = line[srl_token_start:]

        if len(rel_stacks) == 0:
            for _ in range(len(srl_rels)):
                relations.append([])
                rel_stacks.append([])

        for i, rel in enumerate(srl_rels):
            if "(" in rel:
                # push
                rel_starts = [r.strip() for r in rel.split("(") if r.strip()]
                for rs in rel_starts:
                    arg_name = rs.strip(")*")
                    rel_stacks[i].append(
                        {
                            "label": arg_name.replace("A", "ARG"),
                            "token_start": int(tok_id),
                        }
                    )
                    if arg_name == "V":
                        rel_stacks[i][-1]["sense"] = sense + "." + sense_num

            if ")" in rel:
                # pop
                rel_ends = rel.count(")")
                for _ in range(rel_ends):
                    rel_popped = rel_stacks[i].pop(-1)
                    rel_popped["token_end"] = int(tok_id)
                    relations[i].append(rel_popped)
    sentence = " ".join(sentence_tokens)

    pred2relations = []
    for rels in relations:
        if len(rels):
            p = None
            rs = []
            for r in rels:
                if "sense" in r:
                    p = r
                else:
                    rs.append(r)
            assert p is not None
            pred2relations.append((p, rs))

    return sentence, pred2relations


def get_pred_relations_2012(conll_sentence):
    """
    Convert the 2012 style conll format of srl into spacy doc
    """
    sentence_tokens = []
    # the column index where the srl relations start
    srl_token_start = 8
    curr_doc_sent_id = None
    relations = []
    rel_stacks = []

    for line in conll_sentence:
        doc_id, sent_id, tok_id, word, pos, cp, predicate, sense = line[
            :srl_token_start
        ]
        sentence_tokens.append(word)
        if curr_doc_sent_id is None:
            curr_doc_sent_id = doc_id + '-' + sent_id
        elif curr_doc_sent_id != doc_id + '-' + sent_id:
            raise AssertionError("conll_sentence contains ids of multiple sentences")
        # each column contain the proposition relations
        srl_rels = line[srl_token_start:]

        if len(rel_stacks) == 0:
            for _ in range(len(srl_rels)):
                relations.append([])
                rel_stacks.append([])

        for i, rel in enumerate(srl_rels):
            if "(" in rel:
                # push
                rel_starts = [r.strip() for r in rel.split("(") if r.strip()]
                for rs in rel_starts:
                    arg_name = rs.strip(")*")
                    rel_stacks[i].append(
                        {
                            "label": arg_name,
                            "token_start": int(tok_id),
                        }
                    )
                    if arg_name == "V":
                        rel_stacks[i][-1]["sense"] = sense

            if ")" in rel:
                # pop
                rel_ends = rel.count(")")
                for _ in range(rel_ends):
                    rel_popped = rel_stacks[i].pop(-1)
                    rel_popped["token_end"] = int(tok_id)
                    relations[i].append(rel_popped)
    sentence = " ".join(sentence_tokens)

    pred2relations = []
    for rels in relations:
        if len(rels):
            p = None
            rs = []
            for r in rels:
                if "sense" in r:
                    p = r
                else:
                    rs.append(r)
            assert p is not None
            pred2relations.append((p, rs))

    return sentence, pred2relations, curr_doc_sent_id

def get_pred_relations_ewt(conll_sentence):
    """
    Convert the ewt conll format of srl into spacy doc
    """
    sentence_tokens = []
    # the column index where the srl relations start
    srl_token_start = 8
    curr_doc_sent_id = None
    relations = []
    rel_stacks = []

    for line in conll_sentence:
        doc_id, sent_id, tok_id, word, pos, cp, predicate, sense = line[
            :srl_token_start
        ]
        sentence_tokens.append(word)
        # strip off .xml
        stripped_doc_id = doc_id[:-4]
        splits_from_id = stripped_doc_id.split('/')
        sent_num = int(sent_id) + 1
        potential_id = splits_from_id[2] + '-' + splits_from_id[-1] + '-' + str(sent_num).zfill(4)
        if curr_doc_sent_id is None:
            curr_doc_sent_id = potential_id
        elif curr_doc_sent_id != potential_id:
            raise AssertionError("conll_sentence contains ids of multiple sentences")
        # each column contain the proposition relations
        srl_rels = line[srl_token_start:]

        if len(rel_stacks) == 0:
            for _ in range(len(srl_rels)):
                relations.append([])
                rel_stacks.append([])

        for i, rel in enumerate(srl_rels):
            if "(" in rel:
                # push
                rel_starts = [r.strip() for r in rel.split("(") if r.strip()]
                for rs in rel_starts:
                    arg_name = rs.strip(")*")
                    rel_stacks[i].append(
                        {
                            "label": arg_name,
                            "token_start": int(tok_id),
                        }
                    )
                    if arg_name == "V":
                        rel_stacks[i][-1]["sense"] = sense

            if ")" in rel:
                # pop
                rel_ends = rel.count(")")
                for _ in range(rel_ends):
                    rel_popped = rel_stacks[i].pop(-1)
                    rel_popped["token_end"] = int(tok_id)
                    relations[i].append(rel_popped)
    sentence = " ".join(sentence_tokens)

    pred2relations = []
    for rels in relations:
        if len(rels):
            p = None
            rs = []
            for r in rels:
                if "sense" in r:
                    p = r
                else:
                    rs.append(r)
            assert p is not None
            pred2relations.append((p, rs))

    return sentence, pred2relations, curr_doc_sent_id