# Parsing of string with LCFRS/DCP hybrid grammar.
from __future__ import print_function
import math
import sys
import re

from grammar.lcfrs import *

from parser.naive.derivation import Derivation
from grammar.dcp import *
from parser.parser_interface import AbstractParser
from collections import namedtuple, defaultdict

RULE = 0
PAIR = 1
KEY = 2

if sys.version_info[0] == 3:
    unicode = str

# ###########################################################
# Parsing auxiliaries.

# Span, represents input positions from i+1 to j.
Span = namedtuple('Span', ['low', 'high'])
# class Span:
# # Constructor.
# # i: int
# # j: int
# def __init__(self, i, j):
# self.__i = i
# self.__j = j
# 
# # return: int
# def low(self):
# return self.__i
# 
#     # return: int
#     def high(self):
#         return self.__j
# 
#     # String representation.
#     # return: string
#     def __str__(self):
#         return '[' + str(self.low()) + '-' + str(self.high()) + ']'
#
#     def to_tuple(self):
#         return self.__i, self.__j


# Like LHS but terminals and some variables replaced by spans.
class LHS_instance:
    # Constructor.
    # nont: string
    # args: list of list of string/Span/LCFRS_var
    def __init__(self, nont, args=None):
        self.__nont = nont
        self.__args = []
        args = args or []
        for arg in args:
            self.__args += [list(arg)]

    # Add empty arg.
    def add_arg(self):
        self.__args += [[]]

    # Add member to last arg.
    # mem: string or Span or LCFRS_var
    def add_mem(self, mem):
        self.__args[-1] += [mem]

    # Number of arguments.
    # return: int
    def fanout(self):
        return len(self.__args)

    # return: string
    def nont(self):
        return self.__nont

    # return: list of string/Span/LCFRS_var
    def arg(self, i):
        return self.__args[i]

    # Make copy.
    # return: LHS_instance
    def clone(self):
        return LHS_instance(self.__nont, self.__args)

    # Spans must be in increasing order and neighbouring ones
    # must connect.
    # return: bool
    def consistent(self):
        pos = 0
        for arg in self.__args:
            gap = True
            for mem in arg:
                if isinstance(mem, Span):
                    if mem.low < pos:
                        return False
                    # elif not gap and str(mem.low()) != str(pos): TODO: str ??
                    elif not gap and mem.low != pos:
                        return False
                    pos = mem.high
                    gap = False
                else:
                    gap = True
                    # Assuming that grammar is epsilon-free, i.e. every gap >= 1
                    pos += 1
        return True

    # What is minimum and maximum of first position in next RHS member,
    # with given input length.
    # TODO: next? = i-th nont in rhs
    # TODO: assumes that their are no strings in __args
    # i: int
    # inp_len: int
    # return: pair of int
    def next_member_bounds(self, i, inp_len):
        pos = 0
        low = None
        for arg in self.__args:
            gap = True
            for mem in arg:
                if isinstance(mem, Span):
                    if low is not None:
                        return low, mem.low
                    pos = mem.high
                    gap = False
                else:
                    # mem instance of LCFRS_var
                    if mem.mem == i and mem.arg == 0:
                        if gap:  # TODO: variable precedes <i,0>
                            low = pos
                        else:
                            return pos, pos
                            # TODO: why does one return [last span.high, last span.high] (= empty)
                            # TODO: if no variable precedes <i,0>, and
                            # TODO: [last span.high, next span_low] or [last span.high, inp_length]
                            # TODO: otherwise
                    gap = True
        if low is not None:
            return low, inp_len
        else:
            return 0, inp_len

    # Replace all occurrences of LCFRS_var [i,j] by span
    # (Improved version, without reconstructing the lists.)
    # __author__ = 'kilian'
    # i: int
    # j: int
    # span: Span
    # def replace(self, i, j, span):
    #     for argI in range(len(self.__args)):
    #         for memI in range(len(self.__args[argI])):
    #             mem = self.__args[argI][memI]
    #             if isinstance(mem, LCFRS_var):
    #                 if mem.mem == i and mem.arg == j:
    #                     self.__args[argI][memI] = span
    def replace(self, i, j, span):
        for argI, arg in enumerate(self.__args):
            for memI, mem in enumerate(arg):
                if isinstance(mem, LCFRS_var):
                    if mem.mem == i and mem.arg == j:
                        self.__args[argI][memI] = span
                        # each variable occurs at most once
                        return

    def replace_consistent(self, nont_mem, nont_item):
        """
        Joint version of replace and consistent to improve the performance:
        Stops early, if found range does not match.
        :type nont_mem: int
        :type nont_item:
        :rtype: bool
        """
        pos = 0
        for argI, arg in enumerate(self.__args):
            gap = True
            for memI, mem in enumerate(arg):
                if isinstance(mem, LCFRS_var):
                    if mem.mem == nont_mem:
                        span = nont_item.arg(mem.arg)[0]
                        if span.low < pos:
                            return False
                        elif not gap and span.low != pos:
                            return False
                        self.__args[argI][memI] = span
                        pos = span.high
                        gap = False
                    else:
                        gap = True
                        # Assuming that the grammar is epsilon-free,
                        # i.e. every variable is replaced by nonempty string
                        pos += 1
                # elif isinstance(mem, Span):
                else:
                    if mem.low < pos:
                        return False
                    elif not gap and mem.low != pos:
                        return False
                    pos = mem.high
                    gap = False
        return True

    # TODO: Support for empty nonterminals (i.e. with fan-out 0)?
    # Assuming there are no variables left, the left-most position.
    # return: int
    def left_position(self):
        return self.__args[0][0].low

    # Take concatenated span for each argument.
    # This is assuming that there are no variables left.
    # TODO Also, every argument shall be consistent non-empty!
    def collapse(self):
        self.__args = [self.__collapse_arg(arg) for arg in self.__args]

    # arg: list of Span
    # return: list of (one) Span
    @staticmethod
    def __collapse_arg(arg):
        return [Span(arg[0].low, arg[-1].high)]

    # String representation.
    # return: string
    def __str__(self):
        return self.nont() + '(' \
               + '; '.join([
                               ' '.join(map(str, arg))
                               for arg in self.__args]) \
               + ')'

    # TODO: this might cause errors! difference between range and variable is blurred
    def new_key(self):
        return self.nont(), tuple([obj for arg in self.__args for obj in arg])

    # TODO: this might cause errors! difference between range and variable is blurred
    def key_ranges(self):
        return tuple([obj for arg in self.__args for obj in arg])


# In a rule instance where terminals and some variables are
# replaced by spans,
# and there is dot indicating how many RHS nonterminals
# have been resolved.
class Rule_instance:
    # rule: LCFRS_rule
    # lhs: LHS_instance
    # dot: int
    def __init__(self, rule, lhs, dot=0):
        self.__rule = rule
        self.__lhs = lhs
        self.__dot = dot

    # Get underlying rule.
    # return: LCFRS_rule
    def rule(self):
        return self.__rule

    # Get LHS.
    # return: LHS_instance
    def lhs(self):
        return self.__lhs

    # Get dot position.
    # return: int
    def dot(self):
        return self.__dot

    # Dot is at end.
    # return: bool
    def complete(self):
        return self.dot() == self.rule().rank()

    # Is consistent if LHS is.
    # return: bool
    def consistent(self):
        return self.lhs().consistent()

    # What is minimum and maximum of first position in next RHS member,
    # with given input length.
    # return: pair of int
    def next_member_bounds(self, inp_len):
        return self.lhs().next_member_bounds(self.dot(), inp_len)

    # Next nonterminal to be processed.
    # return: string
    def next_nont(self):
        return self.rule().rhs_nont(self.dot())

    # Simplify rule assuming no remaining variables.
    def collapse(self):
        self.__lhs.collapse()

    # String representation.
    # return: string
    def __str__(self):
        s = '[' + str(self.rule().weight()) + '] ' + str(self.lhs()) + ' -> '
        for i in range(self.rule().rank()):
            if self.dot() == i:
                s += '*'
            s += self.rule().rhs_nont(i)
            if i < self.rule().rank() - 1:
                s += ' '
        if self.dot() == self.rule().rank():
            s += '*'
        return s

    # Short string representation (without probability).
    # return: string
    def key(self):
        s = str(self.lhs()) + '->'
        for i in range(self.rule().rank()):
            if self.dot() == i:
                s += '*'
            s += self.rule().rhs_nont(i)
            if i < self.rule().rank() - 1:
                s += ' '
        if self.dot() == self.rule().rank():
            s += '*'
        return s

    def new_key(self):
        return self.lhs().key_ranges(), self.dot(), id(self.rule())


# For rule and input string, replace terminals by spans in all possible
# ways.
# rule: LCFRS_rule
# inp: list of string
def make_rule_instances(rule, inp):
    empty = LHS_instance(rule.lhs().nont())
    lhs_instances = make_rule_instances_from_args(empty, \
                                                  rule.lhs().args(), inp, 0)
    lhs_instances = [lhs for lhs in lhs_instances if lhs.consistent()]
    return [Rule_instance(rule, lhs) for lhs in lhs_instances]


# Make instances from remaining arguments.
# instance: LHS_instance
# args: list of list of string/LCFRS_var
# inp: list of string
# pos: int
# return: list of LHS_instance
def make_rule_instances_from_args(instance, args, inp, pos):
    if len(args) == 0:
        return [instance]
    else:
        first_arg = args[0]
        rest_args = args[1:]
        instance.add_arg()
        if len(first_arg) == 0:
            out = []
            for i in range(pos, len(inp) + 1):
                span = Span(i, i)
                new_instance = instance.clone()
                new_instance.add_mem(span)
                out += make_rule_instances_from_args(new_instance, \
                                                     rest_args, inp, i)
            return out
        else:
            return make_rule_instances_from_members(instance, \
                                                    first_arg, rest_args, inp, pos)


# Make instances from remaining members.
# instance: LHS_instance
# members: list of string/LCFRS_var
# args: list of list of string/LCFRS_var
# inp: list of string
# pos: int
# return: list of LHS_instance
def make_rule_instances_from_members(instance, members, args, inp, pos):
    if len(members) == 0:
        return make_rule_instances_from_args(instance, args, inp, pos)
    else:
        mem = members[0]
        rest_mems = members[1:]
        if isinstance(mem, str) or isinstance(mem, unicode):
            out = []
            for i in range(pos, len(inp)):
                if inp[i] == mem:
                    span = Span(i, i + 1)
                    new_instance = instance.clone()
                    new_instance.add_mem(span)
                    if new_instance.consistent():
                        out += make_rule_instances_from_members(new_instance, \
                                                                rest_mems, args, inp, i + 1)
            return out
        else:
            instance.add_mem(mem)
            return make_rule_instances_from_members(instance, \
                                                    rest_mems, args, inp, pos)


#######################################################
# Parser.

class LCFRS_parser(AbstractParser):
    def all_derivation_trees(self):
        assert 'Not implemented'

    # Constructor.
    # grammar: LCFRS
    # inp: list of string
    def __init__(self, grammar, input=None, save_preprocess=None, load_preprocess=None):
        super(LCFRS_parser, self).__init__(grammar, input)
        self.__g = grammar
        self.__nont_items = defaultdict(list)
        self.__rule_items = defaultdict(list)
        self.__agenda = []
        self.__agenda_set = set()
        self.__trace = defaultdict(list)
        self.__best = {}
        self.__inp = input
        if self.__inp is not None:
            self.__parse()
        else:
            self.preprocess_grammar(grammar)

    def parse(self):
        self.__parse()

    def set_input(self, input):
        self.__inp = input

    def clear(self):
        self.__inp = None
        self.__nont_items = defaultdict(list)
        self.__rule_items = defaultdict(list)
        self.__agenda = []
        self.__agenda_set = set()
        self.__trace = defaultdict(list)
        self.__best = {}

    def __parse(self):
        inp = self.__inp
        inp_len = len(inp)
        for rule in self.__g.epsilon_rules():
            for inst in make_rule_instances(rule, inp):
                self.__record_item(inst, rule)
        for term in set(inp):
            for rule in self.__g.lex_rules(term):
                for inst in make_rule_instances(rule, inp):
                    self.__record_item(inst, rule)
        while len(self.__agenda) != 0:
            item = self.__agenda.pop()
            if isinstance(item, LHS_instance):
                low = item.left_position()
                nont = item.nont()
                # key = str(low) + ' ' + nont
                key = low, nont
                self.__nont_items[key].append(item)
                for rule_item in self.__rule_items[key]:
                    # self.__combine(rule_item, item, rule_item.key(), str(item))
                    self.__combine(rule_item, item, (KEY, rule_item.new_key()), (KEY, item.new_key()))
                for rule in self.__g.nont_corner_of(nont):
                    for inst in make_rule_instances(rule, inp):
                        # self.__combine(inst, item, rule, str(item))
                        self.__combine(inst, item, rule, (KEY, item.new_key()))

            else:  # instance of Rule_instance
                (low, high) = item.next_member_bounds(inp_len)
                nont = item.next_nont()
                # these are possible start positions for the next nont_item for A_i
                # there might be multiple, if a variable precedes x_i,0
                for pos in range(low, high + 1):
                    # key = str(pos) + ' ' + nont
                    key = pos, nont
                    self.__rule_items[key].append(item)
                    for nont_item in self.__nont_items[key]:
                        # self.__combine(item, nont_item, item.key(), str(nont_item))
                        self.__combine(item, nont_item, (KEY, item.new_key()), (KEY, nont_item.new_key()))

                        # key = low, nont
                        # self.__rule_items[key].append(item)
                        # for nont_item in self.__nont_items[key]:
                        #     # self.__combine(item, nont_item, item.key(), str(nont_item))
                        #     self.__combine(item, nont_item, (KEY, item.new_key()), (KEY, nont_item.new_key()))

    # Combine rule item with nont item.
    # rule_item: Rule_instance
    # nont_item: LHS_instance
    # rule_trace: string or LCFRS_rule
    # nont_trace: string
    def __combine(self, rule_item, nont_item, rule_trace, nont_trace):
        lhs = rule_item.lhs().clone()
        dot = rule_item.dot()
        # for i in range(nont_item.fanout()):
        #     arg = nont_item.arg(i)
        #     low = arg[0].low
        #     high = arg[0].high
        #     lhs.replace(dot, i, Span(low, high))
        # if lhs.consistent():
        if lhs.replace_consistent(dot, nont_item):
            advanced_item = Rule_instance(rule_item.rule(), lhs, dot=dot + 1)
            self.__record_item(advanced_item, (PAIR, rule_trace, nont_trace))

    def __record_item(self, item, trace):
        """
        :type item: Rule_instance
        :param trace: pair or LCFRS_rule
        :return:
        """
        if item.complete():
            lhs = item.lhs()
            lhs.collapse()
            # key = str(lhs)
            key = KEY, lhs.new_key()
            if key not in self.__agenda_set:
                self.__agenda_set.add(key)
                self.__agenda.append(lhs)
            self.__trace[key].append(trace)
        else:
            key = KEY, item.new_key()
            # key = item.key()
            if key not in self.__agenda_set:
                self.__agenda_set.add(key)
                self.__agenda.append(item)
            self.__trace[key].append(trace)

    # Start item (which if presence indicates recognition).
    # return: LHS_instance
    def __start_item(self):
        start_lhs = LHS_instance(self.__g.start())
        start_lhs.add_arg()
        start_lhs.add_mem(Span(0, len(self.__inp)))
        return start_lhs

    # Return weight of best derivation.
    # Or -1 when none found.
    # return: float
    def best(self):
        start_lhs = self.__start_item()
        # elem = str(start_lhs)
        elem = KEY, start_lhs.new_key()
        trace = self.__trace[elem]
        if len(trace) == 0:
            return -1
        else:
            return self.__find_best_from(elem)

    # Find weight of best subderivation for all items, top-down.
    # elem: pair or string
    # return: float
    def __find_best_from(self, elem):
        # if isinstance(elem, str) or isinstance(elem, unicode):
        if isinstance(elem, tuple) and elem[0] == KEY:
            if elem in self.__best:
                return self.__best[elem]
            else:
                self.__best[elem] = sys.float_info.max  # avoid cycles
                traces = self.__trace[elem]
                best = sys.float_info.max
                for trace in traces:
                    best = min(best, self.__find_best_from(trace))
                self.__best[elem] = best
                return best
        elif isinstance(elem, tuple) and elem[0] == PAIR:
            # return self.__find_best_from(elem[0]) + self.__find_best_from(elem[1])
            return self.__find_best_from(elem[1]) + self.__find_best_from(elem[2])
        else:
            return -math.log(elem.weight())

    # Recognized?
    # return: bool
    def recognized(self):
        start_lhs = self.__start_item()
        # elem = str(start_lhs)
        elem = KEY, start_lhs.new_key()
        trace = self.__trace[elem]
        return len(trace) > 0

    # Return best derivation or None.
    # return: Derivation
    def best_derivation_tree(self):
        start_lhs = self.__start_item()
        # elem = str(start_lhs)
        elem = KEY, start_lhs.new_key()
        trace = self.__trace[elem]
        if len(trace) == 0:
            return None
        else:
            w = self.best()
            tree = Derivation()
            self.__best_derivation_tree_rec(elem, tree, tree.root_id(), w, [Span(0, len(self.__inp))])
            return tree

    # Get derivation tree of best parse. (includes Spans of sub derivations)
    # elem: string or tuple (identifier in trace)
    # tree: Derivation (tree that gets extended)
    # id: string position (Gorn) in Derivation tree that is extended
    # w: float (weight of best path in subderivation)
    # spans: list of Span (of subderivation)
    # return: Derivation
    def __best_derivation_tree_rec(self, elem, tree, id, w, spans):
        # if isinstance(elem, str) or isinstance(elem, unicode):
        if isinstance(elem, tuple) and elem[0] == KEY:
            # passive item:
            traces = self.__trace[elem]
            for trace in traces:
                if w == self.__find_best_from(trace):
                    return self.__best_derivation_tree_rec(trace, tree, id, w, spans)
            print('looking for', w, 'found:')
            for trace in traces:
                print(self.__find_best_from(trace))
            raise Exception('backtrace failed')
        elif isinstance(elem, tuple) and elem[0] == PAIR:
            # active item (elem[0]) was combined with passive item (elem[0])
            w1 = self.__find_best_from(elem[1])
            w2 = self.__find_best_from(elem[2])
            # extract span of the passive item

            # sub_span = extract_spans(elem[1])
            sub_span = new_extract_spans(elem[2])
            # extend tree at child position corresponding to the dot of active item
            # self.__best_derivation_tree_rec(elem[1], tree, id + tree.gorn_delimiter() + str(dot_position(elem[0])), w2,
            #                             sub_span)
            self.__best_derivation_tree_rec(elem[2], tree, id + tree.gorn_delimiter() + str(new_dot_position(elem[1])),
                                            w2,
                                            sub_span)
            # extend active item
            # self.__best_derivation_tree_rec(elem[0], tree, id, w1, spans)
            self.__best_derivation_tree_rec(elem[1], tree, id, w1, spans)

        else:
            # if all children have been added to derivation, add parent rule
            # as Rule_instance with detected spans
            lhs = LHS_instance(elem.lhs().nont())
            for span in spans:
                lhs.add_arg()
                lhs.add_mem(span)
            ri = Rule_instance(elem, lhs, elem.rank())
            tree.add_rule(id, ri, w)


# FIXME: This method only works, if nonterminals don't contain whitespace!!!
# FIXME: there must a better way to construct the Derivation tree
# the position of the nonterminal no rhs that follows the dot
# (counting started from 0)
# input: key-string of Rule_instance
# return: int
def dot_position(key):
    if isinstance(key, str) or isinstance(key, unicode):
        match = re.search(r'^.*->(.*)\*(.*)$', key)
        return len([i for i in match.group(1).split(' ') if i != ''])
    else:
        return 0


def new_dot_position(key):
    if isinstance(key, LCFRS_rule):
        return 0
    # elif: isinstance(key, tuple) and key[0] == KEY:
    else:
        return key[1][1]


# extract spans from key-string of passive item
# key: string (e.g. "A([0-4]; [12-15])" )
# return: list of Span
def extract_spans(key):
    spans = []
    match = re.search(r'^\s*.*\((\s*\[.*\]\s*)\)\s*$', key)
    for s in match.group(1).split('; '):
        match1 = re.search(r'^\[([0-9]+)-([0-9]+)\];*$', s)
        spans += [Span(int(match1.group(1)), int(match1.group(2)))]
    return spans


def new_extract_spans(key):
    # assert key[0] == KEY
    spans = []
    for low, high in list(key[1][1]):
        spans.append(Span(low, high))
    return spans


__all__ = ["LCFRS_parser", "LHS_instance", "Span"]