from parser.parser_interface import AbstractParser
from grammar.LCFRS.lcfrs import LCFRS, LCFRS_rule
import heapq
from collections import defaultdict
from math import log


class PassiveItem:
    def __init__(self, nonterminal, rule):
        """
        :param nonterminal:
        :type rule: LCFRS_rule
        """
        self.rule = rule
        self.nonterminal = nonterminal
        self.weight = log(rule.weight())
        self.children = []

    def left_position(self):
        """
        :rtype: int
        """
        pass

    def complete(self):
        """
        :rtype: bool
        """
        return len(self.children) == self.rule.rank()

def rule_to_passive_items(rule, input):
    """
    :type rule: LCFRS_rule
    :type input: [str]
    :return:
    """
    empty = PassiveItem(rule.lhs().nont(), rule)



class ActiveItem:
    def __init__(self, rule):
        self.rule = rule


class ViterbiParser(AbstractParser):
    def __init__(self, grammar, input):
        """
        :type grammar: LCFRS
        :type input: list[str]
        """
        self.grammar = grammar
        self.input = input
        self.agenda = []
        self.active_chart = defaultdict(list)
        self.passive_chart = defaultdict(list)
        pass

    def __parse(self):
        for rule in self.grammar.epsilon_rules():
            for item in rule_to_passive_items(rule, self.input):
                self.__record_item(item)
        for terminal in set(input):
            for rule in self.grammar.lex_rules(terminal):
                for item in rule_to_passive_items(rule, input):
                    self.__record_item(item)
        while self.agenda:
            item = heapq.heappop(self.agenda)
            if isinstance(item, PassiveItem):
                low = item.left_position()
                nont = item.nonterminal()
                key = low, nont
                for active_item in self.active_chart[key]:
                    self.__combine(active_item, item)
                for rule in self.grammar.nont_corner_of(nont):
                    for active_item in rule_to_active(rule, input, low):
                        self.__combine(active_item, item)

    def __combine(self, active_item, passive_item):
        pass

    def __record_item(self, item):
        pass

    def recognized(self):
        pass

    def all_derivation_trees(self):
        pass

    def best(self):
        pass

    def best_derivation_tree(self):
        pass

