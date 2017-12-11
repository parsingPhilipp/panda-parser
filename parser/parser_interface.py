from __future__ import print_function

__author__ = 'kilian'

from abc import ABCMeta, abstractmethod
from grammar.lcfrs import *
from parser.sDCPevaluation.evaluator import dcp_to_hybridtree, The_DCP_evaluator
from hybridtree.monadic_tokens import MonadicToken
from collections import defaultdict
from math import exp

class AbstractParser:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, grammar, input=None, save_preprocessing=None, load_preprocessing=None):
        """
        :type grammar: LCFRS
        :type input: list[str]
        :return:
        """
        pass

    @abstractmethod
    def best(self):
        """
        :rtype: float
        """
        pass

    @abstractmethod
    def recognized(self):
        """
        :rtype: bool
        """
        pass

    @abstractmethod
    def best_derivation_tree(self):
        """
        :rtype: AbstractDerivation
        :return:
        """
        pass

    @abstractmethod
    def all_derivation_trees(self):
        """
        :rtype: list(AbstractDerivation)
        :return:
        """
        pass

    def dcp_hybrid_tree_best_derivation(self, tree, tokens, ignore_punctuation, construct_token, punctuation_positions=None):
        """
        :param tree:
        :type tree: GeneralHybridTree
        :param tokens: list[MonadicToken]
        :param ignore_punctuation:
        :type ignore_punctuation: bool
        :return: The Hybrid Tree obtained through evaluation of the dcp-component of the best parse.
        :rtype: GeneralHybridTree
        """
        dcp_evaluation = self.dcp_best_derivation()
        if dcp_evaluation:
            return dcp_to_hybridtree(tree, dcp_evaluation, tokens, ignore_punctuation, construct_token, punct_positions=punctuation_positions)
        else:
            return None

    def dcp_best_derivation(self):
        der = self.best_derivation_tree()
        # print der
        if der is not None:
            # todo: comment out the next integrity check
            if not der.check_integrity_recursive(der.root_id(), der.getRule(der.root_id()).lhs().nont()):
                print(der)
                raise Exception()
            return The_DCP_evaluator(der).getEvaluation()
        else:
            return []

    @staticmethod
    def preprocess_grammar(grammar):
        """
        :type grammar: LCFRS
        :param term_labelling: the terminal labelling
        """
        pass

    @abstractmethod
    def set_input(self, input):
        pass

    @abstractmethod
    def parse(self):
        pass

    @abstractmethod
    def clear(self):
        pass

    def k_best_derivation_trees(self):
        pass

    def best_trees(self, derivation_to_tree):
        weights = defaultdict(lambda: 0.0)
        witnesses = defaultdict(list)
        for i, (weight, der) in enumerate(self.k_best_derivation_trees()):
            tree = derivation_to_tree(der)
            weights[tree] += weight
            witnesses[tree] += [i+1]
        the_derivations = [(tree, weights[tree]) for tree in weights]
        the_derivations.sort(key=lambda x: x[1], reverse=True)
        return [(tree, weight, witnesses[tree]) for tree, weight in the_derivations]


def best_hybrid_tree_for_best_derivation():
    pass


def hybird_tree_from_sdcp_evaluation_for_best_derivation(self):
    # TODO
    pass


def all_hybrid_trees_from_derivations():
    pass


def hybrid_trees_from_sdcp_evaluation_of_all_derivations():
    pass


def hybrid_tree_from_sdcp_with_heighest_weight():
    pass
