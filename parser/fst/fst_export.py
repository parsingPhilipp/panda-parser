from pynini import *
from grammar.lcfrs import LCFRS
from grammar.linearization import Enumerator
from math import log, e
from parser.derivation_interface import AbstractDerivation
from parser.parser_interface import AbstractParser
from collections import defaultdict
import sys
from parser.fst.lazy_composition import DelayedFstComposer

FINAL = 'THE-FINAL-STATE'
INITIAL = 'THE-INITIAL-STATE'


def compile_wfst_from_right_branching_grammar(grammar):
    """
    :type grammar: LCFRS
    :rtype: Fst
    Create a FST from a right-branching hybrid grammar.
    The Output of the is a rule tree in `polish notation <https://en.wikipedia.org/wiki/Polish_notation>`_
    """
    myfst = Fst()

    nonterminals = SymbolTable()
    for nont in grammar.nonts():
        sid = myfst.add_state()
        nonterminals.add_symbol(nont, sid)
        if nont == grammar.start():
            myfst.set_start(sid)
    sid = myfst.add_state()
    nonterminals.add_symbol(FINAL, sid)

    myfst.set_final(nonterminals.add_symbol(FINAL))

    rules = Enumerator(first_index=1)
    for rule in grammar.rules():
        rules.object_index(rule)

    terminals = SymbolTable()
    terminals.add_symbol('<epsilon>', 0)

    for rule in grammar.rules():
        if len(rule.rhs()) == 2:
            for rule2 in grammar.lhs_nont_to_rules(rule.rhs_nont(0)):
                if len(rule2.rhs()) == 0:
                    arc = Arc(terminals.add_symbol(rule2.lhs().args()[0][0]),
                              terminals.add_symbol(str(rules.object_index(rule))
                                                   + '-' + str(rules.object_index(rule2))),
                              make_weight(rule.weight() * rule2.weight()),
                              nonterminals.find(rule.rhs_nont(1)))
                    myfst.add_arc(nonterminals.find(rule.lhs().nont()), arc)
        elif len(rule.rhs()) == 0:
            arc = Arc(terminals.add_symbol(rule.lhs().args()[0][0]),
                      terminals.add_symbol(str(rules.object_index(rule))), make_weight(rule.weight()),
                      nonterminals.find(FINAL))
            myfst.add_arc(nonterminals.find(rule.lhs().nont()), arc)
        else:
            assert rule.lhs().nont() == grammar.start()
            arc = Arc(0, terminals.add_symbol(str(rules.object_index(rule))), make_weight(rule.weight()),
                      nonterminals.find(rule.rhs_nont(0)))
            myfst.add_arc(myfst.start(), arc)

    myfst.set_input_symbols(terminals)
    myfst.set_output_symbols(terminals)

    myfst.optimize(True)

    return myfst, rules


def compile_wfst_from_left_branching_grammar(grammar):
    """
        :type grammar: LCFRS
        :rtype: Fst, Enumerator
        Create a FST from a left-branching hybrid grammar.
        The Output of the is a rule tree in `reverse polish notation <https://en.wikipedia.org/wiki/Reverse_Polish_notation>`_
        """
    myfst = Fst()

    nonterminals = SymbolTable()
    sid = myfst.add_state()
    nonterminals.add_symbol(INITIAL)
    myfst.set_start(sid)

    for nont in grammar.nonts():
        sid = myfst.add_state()
        nonterminals.add_symbol(nont, sid)
        if nont == grammar.start():
            myfst.set_final(sid)

    rules = Enumerator(first_index=1)
    for rule in grammar.rules():
        rules.object_index(rule)

    terminals = SymbolTable()
    terminals.add_symbol('<epsilon>', 0)

    for rule in grammar.rules():
        if rule.rank() == 2:
            assert len(rule.lhs().arg(0)) == 2
            for rule2 in grammar.lhs_nont_to_rules(rule.rhs_nont(1)):
                if len(rule2.rhs()) == 0:
                    arc = Arc(terminals.add_symbol(rule2.lhs().args()[0][0]),
                              terminals.add_symbol(
                                  str(rules.object_index(rule2)) + '-' +
                                  str(rules.object_index(rule))),
                              make_weight(rule.weight() * rule2.weight()),
                              nonterminals.find(rule.lhs().nont()))
                    myfst.add_arc(nonterminals.find(rule.rhs_nont(0)), arc)
        elif rule.rank() == 0:
            assert len(rule.lhs().arg(0)) == 1
            arc = Arc(terminals.add_symbol(rule.lhs().args()[0][0]),
                      terminals.add_symbol(str(rules.object_index(rule))), make_weight(rule.weight()),
                      nonterminals.find(rule.lhs().nont()))
            myfst.add_arc(nonterminals.find(INITIAL), arc)
        else:
            assert rule.rank() == 1
            assert rule.lhs().nont() == grammar.start()
            assert len(rule.lhs().arg(0)) == 1
            arc = Arc(0, terminals.add_symbol(str(rules.object_index(rule))), make_weight(rule.weight()),
                      nonterminals.find(grammar.start())
                      )
            myfst.add_arc(nonterminals.find(rule.rhs_nont(0)), arc)

    myfst.set_input_symbols(terminals)
    myfst.set_output_symbols(terminals)

    myfst.optimize(True)

    return myfst, rules


def fsa_from_list_of_symbols2(input):
    return acceptor(''.join(['[' + s + ']' for s in input]))


def fsa_from_list_of_symbols(input, symbol_table):
    """
    :param input:
    :type input:
    :param symbol_table:
    :type symbol_table: SymbolTable
    :return: An acceptor for the given list of tokens.
    :rtype: Fst
    The symbol table gets extended, if new tokens occur in the input.
    """
    fsa = Fst()
    fsa.set_input_symbols(symbol_table)
    fsa.set_output_symbols(symbol_table)
    state = fsa.add_state()
    fsa.set_start(state)
    for x in input:
        next_state = fsa.add_state()
        try:
            arc = Arc(symbol_table.find(x), symbol_table.find(x), 0, next_state)
        except KeyError:
            arc = Arc(symbol_table.add_symbol(x), symbol_table.add_symbol(x), 0, next_state)
        fsa.add_arc(state, arc)
        state = next_state
    fsa.set_final(state)
    return fsa


def make_weight(weight):
    return -log(weight)


def retrieve_rules(linear_fst):
    linear_rules = []
    terminals = linear_fst.output_symbols()
    for s in range(linear_fst.num_states()):
        for arc in linear_fst.arcs(s):
            lab = terminals.find(arc.olabel)
            if isinstance(lab, str):
                linear_rules += [int(rule_string) for rule_string in lab.split("-")]
            else:
                linear_rules += [lab]
    return linear_rules

def retrieve_rules_(ids, terminals):
    linear_rules = []
    for i in ids:
        lab = terminals.find(i)
        if isinstance(lab, str):
            linear_rules += [int(rule_string) for rule_string in lab.split("-")]
        else:
            linear_rules += [lab]
    return linear_rules


def local_rule_stats(fst, stats, lim=sys.maxint):
    i = 0
    for path in fst.paths(output_token_type="symbol"):
        if i >= lim:
            return stats
        for lab in path[1].split(' '):
            if isinstance(lab, str):
                for rule in lab.split('-'):
                    stats[int(rule)] += 1
            else:
                stats[int(lab)] += 1
        i += 1

    return stats


def paths(fst):
    for path in fst.paths(output_token_type="symbol"):
        path_ = []
        for lab in path[1].split(' '):
            if isinstance(lab, str):
                for rule in lab.split('-'):
                    path_.append(int(rule))
            else:
                path_.append(int(lab))
        yield path_


class PolishDerivation(AbstractDerivation):
    def child_ids(self, id):
        if id % 2 == 1 or id == self._len - 1:
            return []
        else:
            return [id + 1, id + 2]

    def getRule(self, id):
        if id >= self._len:
            print
            print id
            print self._len
        return self._rule_list[id]

    def __str__(self):
        return self.der_to_str_rec(self.root_id(), 0)

    def der_to_str_rec(self, item, indentation):
        s = ' ' * indentation * 2 + str(self.getRule(item)) + '\t(' + str(item) + ')\n'
        for child in self.child_ids(item):
            s += self.der_to_str_rec(child, indentation + 1)
        return s

    def child_id(self, id, i):
        return id + i + 1

    def position_relative_to_parent(self, id):
        return id - 2 + (id % 2), (id + 1) % 2

    def root_id(self):
        return 0

    def terminal_positions(self, id):
        if id % 2 == 1 or id == self._len - 1:
            return [id / 2 + 1]
        else:
            return []

    def ids(self):
        return self._ids

    def __init__(self, rule_list):
        self._rule_list = rule_list
        self._len = len(rule_list)
        self._ids = range(self._len)


class ReversePolishDerivation(AbstractDerivation):
    def child_ids(self, id):
        if id % 2 == 1 or id == 0:
            return []
        else:
            return [id - 2, id - 1]

    def getRule(self, id):
        if id >= self._len:
            print
            print id
            print self._len
        return self._rule_list[id]

    def __str__(self):
        return self.der_to_str_rec(self.root_id(), 0)

    def der_to_str_rec(self, item, indentation):
        s = ' ' * indentation * 2 + str(self.getRule(item)) + '\t(' + str(item) + ')\n'
        for child in self.child_ids(item):
            s += self.der_to_str_rec(child, indentation + 1)
        return s

    def child_id(self, id, i):
        return id - 2 + i

    def position_relative_to_parent(self, id):
        return id + 2 - (id % 2), id % 2

    def root_id(self):
        return self._len - 1

    def terminal_positions(self, id):
        if id % 2 == 1 or id == 0:
            return [(id + 3) / 2]
        else:
            return []

    def ids(self):
        return self._ids

    def __init__(self, rule_list):
        self._rule_list = rule_list
        self._len = len(rule_list)
        self._ids = range(self._len)



class RightBranchingFSTParserLazy(AbstractParser):
    def recognized(self):
        if self._polish_rules:
            return True
        else:
            return False

    def best_derivation_tree(self):
        if self._polish_rules:
            polish_rules = map(self._rules.index_object, self._polish_rules)
            # remove dummy chain rule in case of dependency structures
            if len(polish_rules) % 2 == 0:
                polish_rules = polish_rules[1::]
            der = PolishDerivation(polish_rules)
            return der
        else:
            return None

    def __init__(self, grammar, input=None):
        self.input = input
        if input is not None:
            self.fst, self._rules = grammar.tmp_fst
            self.__composer = DelayedFstComposer(self.fst)
            self.parse()
        else:
            self.fst, self._rules = compile_wfst_from_right_branching_grammar(grammar)
            self.__composer = DelayedFstComposer(self.fst)

    def parse(self):
        # call lazy composer
        self._best_ = self.__composer.compose_(self.input)
        self._polish_rules = retrieve_rules_(self._best_, terminals=self.fst.output_symbols())

    def clear(self):
        self._best_ = None
        self.input = None
        self._polish_rules = None

    def best(self):
        # return pow(e, -float(shortestdistance(self._best)[-1]))
        pass

    def all_derivation_trees(self):
        pass

    @staticmethod
    def preprocess_grammar(grammar):
        grammar.tmp_fst = compile_wfst_from_right_branching_grammar(grammar)

class RightBranchingFSTParser(AbstractParser):
    def recognized(self):
        if self._polish_rules:
            return True
        else:
            return False

    def best_derivation_tree(self):
        if self._polish_rules:
            polish_rules = map(self._rules.index_object, self._polish_rules)
            # remove dummy chain rule in case of dependency structures
            if len(polish_rules) % 2 == 0:
                polish_rules = polish_rules[1::]
            der = PolishDerivation(polish_rules)
            return der
        else:
            return None

    def __init__(self, grammar, input=None):
        self.input = input
        if input is not None:
            self.fst, self._rules = grammar.tmp_fst
            self.parse()
        else:
            self.fst, self._rules = compile_wfst_from_right_branching_grammar(grammar)

    def parse(self):
        fsa = fsa_from_list_of_symbols(self.input, self.fst.mutable_input_symbols())
        intersection = fsa * self.fst
        self._best = shortestpath(intersection)

        self._best.topsort()
        self._polish_rules = retrieve_rules(self._best)

    def clear(self):
        self._best = None
        self.input = None
        self._polish_rules = None

    def best(self):
        return pow(e, -float(shortestdistance(self._best)[-1]))

    def all_derivation_trees(self):
        pass

    @staticmethod
    def preprocess_grammar(grammar):
        grammar.tmp_fst = compile_wfst_from_right_branching_grammar(grammar)


class LeftBranchingFSTParserLazy(AbstractParser):
    def recognized(self):
        if self._reverse_polish_rules:
            return True
        else:
            return False

    def best_derivation_tree(self):
        polish_rules = self._reverse_polish_rules
        if polish_rules:
            polish_rules = map(self._rules.index_object, polish_rules)
            # remove dummy chain rule in case of dependency structures
            if len(polish_rules) % 2 == 0:
                polish_rules = polish_rules[0:-1]
            der = ReversePolishDerivation(polish_rules)
            return der
        else:
            return None

    def __init__(self, grammar, input=None, load_preprocess=None, save_preprocess=None):
        self.input = input
        if input is not None:
            self.fst, self._rules = grammar.tmp_fst
            self.__composer = DelayedFstComposer(self.fst)
            self.parse()
        else:
            self.fst, self._rules = compile_wfst_from_left_branching_grammar(grammar)
            self.__composer = DelayedFstComposer(self.fst)

    def set_input(self, input):
        self.input = input

    def parse(self):
        # delayed composition
        self._best_ = self.__composer.compose_(self.input)

        self._reverse_polish_rules = retrieve_rules_(self._best_, self.fst.output_symbols())


    def best(self):
        # return pow(e, -float(shortestdistance(self._best)[-1]))
        pass

    def clear(self):
        self.input = None
        self._best_ = None
        self._reverse_polish_rules = None

    def all_derivation_trees(self):
        pass

    @staticmethod
    def preprocess_grammar(grammar):
        grammar.tmp_fst = compile_wfst_from_left_branching_grammar(grammar)

class LeftBranchingFSTParser(AbstractParser):
    def recognized(self):
        if self._reverse_polish_rules:
            return True
        else:
            return False

    def best_derivation_tree(self):
        polish_rules = self._reverse_polish_rules
        if polish_rules:
            polish_rules = map(self._rules.index_object, polish_rules)
            # remove dummy chain rule in case of dependency structures
            if len(polish_rules) % 2 == 0:
                polish_rules = polish_rules[0:-1]
            der = ReversePolishDerivation(polish_rules)
            return der
        else:
            return None

    def __init__(self, grammar, input=None, load_preprocess=None, save_preprocess=None):
        self.input = input
        if input is not None:
            self.fst, self._rules = grammar.tmp_fst
            self.parse()
        else:
            self.fst, self._rules = compile_wfst_from_left_branching_grammar(grammar)

    def set_input(self, input):
        self.input = input

    def parse(self):
        fsa = fsa_from_list_of_symbols(self.input, self.fst.mutable_input_symbols())
        intersection = compose(fsa, self.fst)
        self._best = shortestpath(intersection)

        self._best.topsort()

        self._reverse_polish_rules = retrieve_rules(self._best)


    def best(self):
        return pow(e, -float(shortestdistance(self._best)[-1]))

    def clear(self):
        self.input = None
        self._best = None
        self._reverse_polish_rules = None

    def all_derivation_trees(self):
        pass

    @staticmethod
    def preprocess_grammar(grammar):
        grammar.tmp_fst = compile_wfst_from_left_branching_grammar(grammar)