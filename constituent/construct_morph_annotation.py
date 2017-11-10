from __future__ import print_function
from collections import defaultdict
from util.enumerator import Enumerator
from grammar.lcfrs import LCFRS
from operator import mul
from functools import reduce


def extract_feat(the_input, features=["number", "person", "tense", "mood", "case", "degree", "category", "pos", "function", "gender"],
                 empties=["", "--"]):
    feats = []
    for feat in [feat for feat in the_input if feat[0] in features and feat[1] not in empties]:
        feats.append(feat)
    if len(feats) > 0:
        feats = sorted(feats, key=lambda x: x[0])
    return tuple(feats)


def pos_cat_feats(the_input, hmarkov=2, left=False):
    if left:
        markov_input = the_input[:2]
    else:
        markov_input = the_input[-hmarkov:]

    return tuple(map(lambda x: extract_feat(x, features=["pos", "category"]), markov_input))


def pos_cat_and_lex_in_unary(the_input, hmarkov=2, left=False, no_function=False):
    if len(the_input) > 1:
        if hmarkov == 0:
            markov_input = []
        else:
            if left:
                markov_input = the_input[:hmarkov]
            else:
                markov_input = the_input[-hmarkov:]
        return tuple(map(lambda x: extract_feat(x, features=["pos", "category"]), markov_input))
    else:
        if no_function:
            return tuple(map(lambda x: extract_feat(x, features=["number", "person", "tense", "mood", "case", "degree", "category", "pos", "gender"]), the_input))
        else:
            return tuple(map(extract_feat, the_input))





def build_nont_splits_dict(grammar
                           , morph_log
                           , nonterminals
                           , feat_function=lambda xs: tuple(map(extract_feat, xs))
                           , debug=False):
    """
    :type grammar: LCFRS
    :type morph_log: dict
    :type nonterminals: Enumerator
    """
    nont_splits = defaultdict(lambda: 0)
    split_id = defaultdict(dict)
    split_id_count = defaultdict(dict)

    if debug:
        print("\nBuilding feature log.\n")
    for entry in morph_log:
        # don't consider rules
        if type(entry[0]) is not int:
            # just consider nonterminals that span
            if True or len(entry[1:]) == 1:
                if debug:
                    print(entry)
                feats = feat_function(entry[1][0])
                if debug:
                    print(feats)
                if True or len(feats) > 0:
                    if feats not in split_id[entry[0]]:
                        nont_splits[entry[0]] += 1
                        feat_id = nont_splits[entry[0]]
                        split_id[entry[0]][feats] = feat_id
                        split_id_count[entry[0]][feats] = 0
                    split_id_count[entry[0]][feats] += morph_log[entry]

    if debug:
        print('\nnonterminal splits\n')
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
                    lhs_feat = feat_function(entry[1][0])
                    lhs_split = split_id[lhs_nont][lhs_feat] - 1
                    baseweight = 1.0 * morph_log[entry] / split_id_count[lhs_nont][lhs_feat]
                    weight_idx = lhs_split * reduce(mul, splits_array[1:], 1)

                    rhs_split_selection = []
                    # print(entry[2])
                    for idx, rhs_nont in enumerate(rule.rhs()):
                        # rhs_feat = extract_feat(entry[2][0 + idx][0][0])
                        rhs_feat = feat_function(entry[2][0 + idx][0])
                        # rhs_feat = feats = map(lambda x: extract_feat(x, features, empties), entry[2][0 + idx])
                        rhs_split = split_id[rhs_nont][rhs_feat] - 1
                        rhs_split_selection.append(rhs_split)
                        weight_idx += rhs_split * reduce(mul, splits_array[idx + 2:], 1)

                    weights[weight_idx] += baseweight  # * 1 / rhs_splits
                    if debug:
                        print(splits, weights, splits_array, [lhs_split] + rhs_split_selection, weight_idx, baseweight)
            rule_weights.append(weights)

    for weight, idx in zip(rule_weights, range(0, len(grammar.rules()))):
        if debug:
            print(idx, grammar.rule_index(idx), weight)

    nonterminal_splits = [max(1, nont_splits[nonterminals.index_object(idx)])
                          for idx in range(nonterminals.get_first_index(), nonterminals.get_counter())]
    if nont_splits[grammar.start()] == 0:
        root_weights = [1.0]
    else:
        root_weights = [0.0] * nont_splits[grammar.start()]
        count_sum = sum([split_id_count[grammar.start()][key] for key in split_id_count[grammar.start()]])
        for key in split_id_count[grammar.start()]:
            root_weights[split_id[grammar.start()][key] - 1] = split_id_count[grammar.start()][key] * 1.0 / count_sum
    return nonterminal_splits, root_weights, rule_weights, split_id
