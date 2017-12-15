from __future__ import print_function
from discodop.containers import Grammar
from discodop.plcfrs import parse
from discodop.kbest import lazykbest
from parser.parser_interface import AbstractParser
from parser.derivation_interface import AbstractDerivation
import nltk
from math import log, exp
from parser.trace_manager.trace_manager import add, prod
from parser.supervised_trainer.trainer import PyDerivationManager
from parser.coarse_to_fine_parser.trace_weight_projection import py_edge_weight_projection
from parser.discodop_parser.grammar_adapter import rule_idx_from_label, transform_grammar


class DiscodopDerivation(AbstractDerivation):
    def __init__(self, nltk_tree, grammar):
        """
        :param nltk_tree:
        :type nltk_tree: nltk.Tree
        :param grammar:
        :type grammar: LCFRS
        """
        self.node_counter = 0
        self.rules = {}
        self.children = {}
        self.__init__rec(nltk_tree[0], grammar)
        self.parent = {}
        self.spans = None

    def __init__rec(self, nltk_tree, grammar):
        if isinstance(nltk_tree, str):
            return []

        rule_idx = rule_idx_from_label(nltk_tree.label())

        node = self.node_counter
        self.node_counter += 1
        self.rules[node] = grammar.rule_index(rule_idx)
        self.children[node] = []
        for c in nltk_tree:
            self.children[node] += self.__init__rec(c[0], grammar)
        return [node]

    def root_id(self):
        return 0

    def getRule(self, id):
        return self.rules[id]

    def child_ids(self, id):
        return self.children[id]

    def child_id(self, id, i):
        return self.children[id][i]

    def position_relative_to_parent(self, id):
        p = self.parent[id]
        return p, self.children[p].index(id)

    def ids(self):
        return range(0, self.node_counter)


class DiscodopKbestParser(AbstractParser):
    def __init__(self, grammar, input=None, save_preprocessing=None, load_preprocessing=None, k=50, heuristics=-1,
                 la=None, variational=False, sum_op=False):
        rule_list = list(transform_grammar(grammar))
        self.disco_grammar = Grammar(rule_list, start=grammar.start())
        self.chart = None
        self.input = input
        self.grammar = grammar
        self.k = k
        self.beam_beta = exp(-10)  # beam pruning factor, between 0 and 1; 1 to disable.
        self.beam_delta = 40  # maximum span length to which beam_beta is applied
        self.counter = 0
        self.la = la
        self.variational = variational
        self.op = add if sum_op else prod

    def best(self):
        pass

    def recognized(self):
        if self.chart and self.chart.root() != 0:
            return True
        else:
            return False

    def max_rule_product_derivation(self):
        if self.recognized():
            return self.__projection_based_derivation_tree(self.la, variational=False, op=prod)

    def max_rule_sum_derivation(self):
        if self.recognized():
            return self.__projection_based_derivation_tree(self.la, variational=False,
                                                           op=add)

    def variational_derivation(self):
        if self.recognized():
            return self.__projection_based_derivation_tree(self, variational=True, op=prod)

    def __projection_based_derivation_tree(self, la, variational=False, op=prod):
        manager = PyDerivationManager(self.grammar)
        manager.convert_chart_to_hypergraph(self.chart, self.disco_grammar, debug=True)
        edge_weights = py_edge_weight_projection(la, manager, variational=variational)
        return manager.viterbi_derivation(0, edge_weights, self.grammar, op=op)

    def best_derivation_tree(self):
        return self.__projection_based_derivation_tree(self.la, variational=self.variational, op=self.op)

    def all_derivation_trees(self):
        pass

    def set_input(self, input):
        self.input = input

    def parse(self):
        self.counter += 1
        self.chart, msg = parse(self.input, self.disco_grammar,
                                beam_beta=-log(self.beam_beta),
                                beam_delta=self.beam_delta)
        # if self.counter > 86:
        #     print(self.input)
        #     print(self.chart)
        #     print(msg)
        if self.chart:
            self.chart.filter()

    def clear(self):
        self.input = None
        self.chart = None

    def k_best_derivation_trees(self):
        for tree_string, weight in lazykbest(self.chart, self.k):
            tree = nltk.Tree.fromstring(tree_string)
            yield weight, DiscodopDerivation(tree, self.grammar)
