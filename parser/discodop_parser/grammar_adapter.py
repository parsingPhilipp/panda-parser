from __future__ import print_function, unicode_literals
from grammar.lcfrs import LCFRS, LCFRS_lhs, LCFRS_var
from discodop.grammar import sortgrammar
from discodop.tree import escape
from discodop.containers import Grammar
from discodop.plcfrs import parse
from discodop.kbest import lazykbest
from parser.parser_interface import AbstractParser
from parser.derivation_interface import AbstractDerivation
import nltk
import re
from math import log, exp

def transform_grammar(grammar):
    # TODO assert ordered rules, terminals only in rules with len(rhs) = 0
    for rule in grammar.rules():
        if rule.weight() == 0.0:
            continue
        fake_nont = rule.lhs().nont() + "-" + str(rule.get_idx())
        trans_rule_fake = (rule.lhs().nont(), fake_nont), tuple([(0,) for _ in rule.lhs().args()])
        yield trans_rule_fake, rule.weight()
        rhs = rule.rhs() if rule.rhs() else ['Epsilon']
        trans_rule = tuple([fake_nont] + rhs), transform_args(rule.lhs().args())
        yield trans_rule, 1.0


def transform_args(args):
    def transform_arg(arg):
        arg_new = []
        for elem in arg:
            if isinstance(elem, LCFRS_var):
                arg_new.append(elem.mem)
            else:
                assert len(arg) == 1
                return escape(elem)
        return tuple(arg_new)
    return tuple([transform_arg(arg) for arg in args])


class DiscodopKbestParser(AbstractParser):
    def __init__(self, grammar, input=None, save_preprocessing=None, load_preprocessing=None, k=50, heuristics=-1):
        rule_list = list(transform_grammar(grammar))
        self.disco_grammar = Grammar(rule_list, start=grammar.start())
        self.chart = None
        self.input = input
        self.grammar = grammar
        self.k = k
        self.beam_beta = exp(-10)  # beam pruning factor, between 0 and 1; 1 to disable.
        self.beam_delta = 40  # maximum span length to which beam_beta is applied

    def best(self):
        pass

    def recognized(self):
        if self.chart and self.chart.root() != 0:
            return True
        else:
            return False

    def best_derivation_tree(self):
        pass

    def all_derivation_trees(self):
        pass

    def set_input(self, input):
        self.input = input

    def parse(self):
        self.chart, msg = parse(self.input, self.disco_grammar,
                                beam_beta=-log(self.beam_beta),
                                beam_delta=self.beam_delta)

    def clear(self):
        self.input = None
        self.chart = None

    def k_best_derivation_trees(self):
        for tree_string, weight in lazykbest(self.chart, self.k):
            tree = nltk.Tree.fromstring(tree_string)
            yield weight, DiscodopDerivation(tree, self.grammar)


striplabelre = re.compile(r'^(.*)-(\d+)$')


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
        split = striplabelre.split(nltk_tree.label())
        assert len(split) == 4
        rule_idx = int(split[-2])

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
