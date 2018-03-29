from cython.operator cimport dereference as deref
from util.enumerator cimport Enumerator
from libcpp.vector cimport vector
from parser.trace_manager.sm_trainer_util cimport PyGrammarInfo, PyStorageManager
from parser.trace_manager.sm_trainer cimport PyLatentAnnotation, build_PyLatentAnnotation
import itertools
from collections import defaultdict
import grammar.lcfrs as gl


def build_sm_grammar(PyLatentAnnotation latent_annotation,
                     grammar,
                     PyGrammarInfo grammarInfo,
                     rule_pruning,
                     rule_smoothing=0.0):
    """
    Given a base LCFRS and a latent annotation object, construct a LCFRS with splitted states.
    :type latent_annotation: PyLatentAnnotation
    :type grammar: gl.LCFRS
    :type grammarInfo: PyGrammarInfo
    :type rule_pruning: float
    :type rule_smoothing: float
    :rtype: gl.LCFRS
    """
    new_grammar = gl.LCFRS(grammar.start() + "[0]")
    for i in range(0, len(grammar.rule_index())):
        rule = grammar.rule_index(i)

        rule_dimensions = [deref(latent_annotation.latentAnnotation).nonterminalSplits[nont]
                           for nont in deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]]
        rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

        lhs_dims = deref(latent_annotation.latentAnnotation).nonterminalSplits[
            deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
        ]

        for la in rule_dimensions_product:
            index = list(la)

            if rule_smoothing > 0.0:
                weight_av = sum([deref(latent_annotation.latentAnnotation).get_weight(i, [lhs] + list(la)[1:])
                    for lhs in range(lhs_dims)]) / lhs_dims

                weight = (1 - rule_smoothing) * deref(latent_annotation.latentAnnotation).get_weight(i, index) \
                         + rule_smoothing * weight_av
            else:
                weight = deref(latent_annotation.latentAnnotation).get_weight(i, index)
            if weight > rule_pruning:
                lhs_la = gl.LCFRS_lhs(rule.lhs().nont() + "[" + str(la[0]) + "]")
                for arg in rule.lhs().args():
                    lhs_la.add_arg(arg)
                nonts = [rhs_nont + "[" + str(la[1 + j]) + "]" for j, rhs_nont in enumerate(rule.rhs())]
                new_grammar.add_rule(lhs_la, nonts, weight, rule.dcp())

    return new_grammar

def construct_fine_grammar(PyLatentAnnotation latent_annotation,
                           grammar,
                           PyGrammarInfo grammarInfo,
                           arg_transform,
                           PyLatentAnnotation la_full,
                           smooth_transform=None,
                           smooth_weight=0.01):
    """
    :param grammar:
    :type grammar: gl.LCFRS
    :type grammarInfo: PyGrammarInfo
    :return:
    :rtype:
    """
    nont_translation = defaultdict(lambda: -1)
    nonterminals = Enumerator()

    def rename(nont, split_id, nont_id):
        if deref(latent_annotation.latentAnnotation).nonterminalSplits[nont_id] == 1:
            idx = nonterminals.object_index(nont)
            nont_translation[idx] = nont_id
            return nont
        else:
            new_nont = str(nont) + "[" + str(split_id) + str("]")
            nonterminals.object_index(new_nont)
            return new_nont

    new_grammar = gl.LCFRS(grammar.start())

    cdef vector[double] root_weights = deref(la_full.latentAnnotation).get_root_weights()
    cdef vector[size_t] smooth_rules = []
    latent_rule_weights = defaultdict(lambda: defaultdict(lambda: 0.0))

    for i in range(0, len(grammar.rule_index())):
        rule = grammar.rule_index(i)
        # nonts = [rule.lhs().nont()] + rule.rhs()
        nont_ids = deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]
        rule_dimensions = []

        for nont_id in nont_ids:
            rule_dimensions.append(deref(latent_annotation.latentAnnotation).nonterminalSplits[nont_id])

        rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

        # lhs_dims = deref(self.latentAnnotation).nonterminalSplits[
        #     deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
        # ]

        rule_dimensions_full = []
        for nont_id in nont_ids:
            rule_dimensions_full.append(deref(la_full.latentAnnotation).nonterminalSplits[nont_id])

        for la in rule_dimensions_product:
            index = list(la)
            weight = deref(latent_annotation.latentAnnotation).get_weight(i, index)
            if weight > 0.0:
                lhs_la = gl.LCFRS_lhs(rename(rule.lhs().nont(), la[0], nont_ids[0]))
                for arg in rule.lhs().args():
                    lhs_la.add_arg(arg_transform(arg, la))
                nonts = [rename(rhs_nont, la[1 + j], nont_ids[1 + j]) for j, rhs_nont in enumerate(rule.rhs())]
                new_rule = new_grammar.add_rule(lhs_la, nonts, weight, rule.dcp())

                product_range = []
                mask = []
                for i2, j2, la_idx in zip(rule_dimensions, rule_dimensions_full, index):
                    if i2 < j2:
                        product_range.append([x for x in range(j2)])
                        mask.append(False)
                    else:
                        product_range.append([la_idx])
                        mask.append(True)

                for laf in itertools.product(*product_range):
                    laf_masked = tuple([0 if mb else laf[mi] for mi, mb in enumerate(mask)])
                    latent_rule_weights[new_rule.get_idx()][laf_masked] \
                        = deref(la_full.latentAnnotation).get_weight(i, list(laf))

                if nonts == [] and smooth_transform is not None:
                    # smoothing part
                    lhs_smooth = gl.LCFRS_lhs(rename(rule.lhs().nont(), la[0], nont_ids[0]))
                    for arg in rule.lhs().args():
                        lhs_smooth.add_arg(smooth_transform(arg))
                    new_rule = new_grammar.add_rule(lhs_smooth, [], smooth_weight, rule.dcp())
                    smooth_rules.push_back(new_rule.get_idx())

                    for laf in itertools.product(*product_range):
                        laf_masked = tuple([0 if mb else laf[mi] for mi, mb in enumerate(mask)])
                        latent_rule_weights[new_rule.get_idx()][laf_masked] \
                            = smooth_weight

    cdef PyGrammarInfo new_grammar_info = PyGrammarInfo(new_grammar, nonterminals)
    cdef vector[size_t] nonterminal_splits = []
    for nont_id in range(0, nonterminals.get_counter()):
        old_idx = nont_translation[nont_id]
        if old_idx == -1:
            nonterminal_splits.push_back(1)

        else:
            nonterminal_splits.push_back(deref(la_full.latentAnnotation).nonterminalSplits[old_idx])

    cdef vector[vector[double]] rule_weights = []

    for idx, nonts in enumerate(deref(new_grammar_info.grammarInfo).rule_to_nonterminals):
        weights = []
        splits = [nonterminal_splits[nont] for nont in nonts]
        for la in itertools.product(*[range(s) for s in splits]):
            weights.append(latent_rule_weights[idx][la])
        rule_weights.push_back(weights)

    cdef PyStorageManager storage_manager = PyStorageManager()
    la_new_grammar = build_PyLatentAnnotation(nonterminal_splits, root_weights, rule_weights, new_grammar_info,
                                              storage_manager)
    return new_grammar, la_new_grammar, new_grammar_info, nonterminals, nont_translation, smooth_rules