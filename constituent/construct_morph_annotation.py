from __future__ import print_function
from collections import defaultdict
from util.enumerator import Enumerator
from grammar.lcfrs import LCFRS
from operator import mul
from functools import reduce


def extract_feat(the_input, features=["number", "person", "tense", "mood", "case", "degree"],
                 empties=["", "--"]):
    feats = []
    for feat in [feat for feat in the_input if feat[0] in features and feat[1] not in empties]:
        feats.append(feat)
    if len(feats) > 0:
        sorted(feats, key=lambda x: x[0])
    return tuple(feats)


def build_nont_splits_dict(grammar
                           , morph_log
                           , nonterminals
                           , features=["number", "person", "tense", "mood", "case", "degree"]
                           , empties=["", "--"]):
    """
    :type grammar: LCFRS
    :type morph_log: dict
    :type nonterminals: Enumerator
    """
    nont_splits = defaultdict(lambda: 0)
    split_id = defaultdict(dict)
    split_id_count = defaultdict(dict)

    print("\nBuilding feature log.\n")
    for entry in morph_log:
        # don't consider rules
        if type(entry[0]) is not int:
            # just consider nonterminals that span
            if len(entry[1:]) == 1:
                print(entry)
                feats = extract_feat(entry[1][0][0], features, empties)
                if len(feats) > 0:
                    if tuple(feats) not in split_id[entry[0]]:
                        nont_splits[entry[0]] += 1
                        feat_id = nont_splits[entry[0]]
                        split_id[entry[0]][tuple(feats)] = feat_id
                        split_id_count[entry[0]][tuple(feats)] = 0
                    split_id_count[entry[0]][tuple(feats)] += morph_log[entry]

    for nont in nont_splits:
        print(nont, nont_splits[nont])
        for feat in split_id[nont]:
            print(feat, split_id[nont][feat], split_id_count[nont][feat])

    nont_split_list = [1] * len(grammar.nonts())
    for nont in nont_splits:
        nont_split_list[nonterminals.object_index(nont)] = nont_splits[nont]
    rule_weights = []
    for idx in range(0, len(grammar.rules())):
        rule = grammar.rule_index(idx)
        lhs_nont = rule.lhs().nont()
        splits_array = [max([nont_splits[lhs_nont], 1])]
        for nont in rule.rhs():
            splits_array.append(max([nont_splits[nont], 1]))

        rhs_splits = reduce(mul, splits_array[1:], 1)
        splits = splits_array[0] * rhs_splits

        if splits == 1:
            rule_weights.append([rule.weight()])
        else:
            weights = [0.0] * splits
            for entry in morph_log:
                if entry[0] == rule.get_idx():
                    lhs_nont = rule.lhs().nont()
                    lhs_split = 0
                    if lhs_nont in nont_splits and nont_splits[lhs_nont] > 0:
                        lhs_feat = extract_feat(entry[1][0][0])
                        lhs_split = split_id[lhs_nont][lhs_feat] - 1
                        baseweight = 1.0 * morph_log[entry] / split_id_count[lhs_nont][lhs_feat]
                    else:
                        baseweight = rule.weight()
                    rhs_split_selection = []
                    for idx, rhs_nont in enumerate(rule.rhs()):
                        if splits_array[idx + 1] > 1:
                            rhs_feat = extract_feat(entry[2][0 + idx][0][0])
                            rhs_split_selection.append(split_id[rhs_nont][rhs_feat] - 1)
                        else:
                            rhs_split_selection.append(0)

                    weight_idx = sum(map(lambda t: t[0] * t[1], zip(splits_array[1:], [lhs_split] + rhs_split_selection[:-1]))) \
                                 + (lhs_split if len(rhs_split_selection) < 1 else rhs_split_selection[-1])
                    print(splits, weights, splits_array, [lhs_split] + rhs_split_selection, weight_idx)
                    weights[weight_idx] += baseweight * 1 / rhs_splits
            rule_weights.append(weights)

    print(rule_weights)