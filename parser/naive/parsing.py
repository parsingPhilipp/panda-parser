# Parsing of string with LCFRS/DCP hybrid grammar.

from collections import defaultdict
import codecs
import re
import math
import sys

from lcfrs import *
from dcp import *
from constituency_tree import HybridTree
from decomposition import expand_spans
from derivation import Derivation
from parser.sDCPevaluation.evaluator import The_DCP_evaluator, dcp_to_hybridtree
from general_hybrid_tree import GeneralHybridTree


############################################################
# Parsing auxiliaries.

# Span, represents input positions from i+1 to j.
class Span:
    # Constructor.
    # i: int
    # j: int
    def __init__(self, i, j):
        self.__i = i
        self.__j = j

    # return: int
    def low(self):
        return self.__i

    # return: int
    def high(self):
        return self.__j

    # String representation.
    # return: string
    def __str__(self):
        return '[' + str(self.low()) + '-' + str(self.high()) + ']'



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
                    if mem.low() < pos:
                        return False
                    # elif not gap and str(mem.low()) != str(pos): TODO: str ??
                    elif not gap and mem.low() != pos:
                        return False
                    pos = mem.high()
                    gap = False
                else:
                    gap = True
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
                        return (low, mem.low())
                    pos = mem.high()
                    gap = False
                else:
                    # mem instance of LCFRS_var
                    if mem.mem() == i and mem.arg() == 0:
                        if gap: # TODO: variable precedes <i,0>
                            low = pos
                        else:
                            return (pos, pos)
                        #TODO: why does one return [last span.high, list span.high] (= empty)
                        #TODO: if no variable precedes <i,0>, and
                        #TODO: [last span.high, next span_low] or [last span.high, inp_length]
                        #TODO: otherwise
                    gap = True
        if low is not None:
            return (low, inp_len)
        else:
            return (0, inp_len)


    # Replace all occurrences of LCFRS_var [i,j] by span
    # (Improved version, without reconstructing the lists.)
    # __author__ = 'kilian'
    # i: int
    # j: int
    # span: Span
    def replace(self, i, j, span):
        for argI in range(len(self.__args)):
            for memI in range(len(self.__args[argI])):
                mem = self.__args[argI][memI]
                if isinstance(mem, LCFRS_var):
                    if mem.mem() == i and mem.arg() == j:
                        self.__args[argI][memI] = span

    # TODO: Support for empty nonterminals (i.e. with fan-out 0)?
    # Assuming there are no variables left, the left-most position.
    # return: int
    def left_position(self):
        return self.__args[0][0].low()

    # Take concatenated span for each argument.
    # This is assuming that there are no variables left.
    # TODO Also, every argument shall be consistent non-empty!
    def collapse(self):
        self.__args = [self.__collapse_arg(arg) for arg in self.__args]
    # arg: list of Span
    # return: list of (one) Span
    def __collapse_arg(self, arg):
        return [Span(arg[0].low(), arg[-1].high())]

    # String representation.
    # return: string
    def __str__(self):
        return self.nont() + '(' \
               + '; '.join([
                    ' '.join(map(str, arg))
                    for arg in self.__args]) \
               + ')'

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
            if i < self.rule().rank()-1:
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
            if i < self.rule().rank()-1:
                s += ' '
        if self.dot() == self.rule().rank():
            s += '*'
        return s

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
                    span = Span(i, i+1)
                    new_instance = instance.clone()
                    new_instance.add_mem(span)
                    if new_instance.consistent():
                        out += make_rule_instances_from_members(new_instance, \
                                                                rest_mems, args, inp, i+1)
            return out
        else:
            instance.add_mem(mem)
            return make_rule_instances_from_members(instance, \
                                                    rest_mems, args, inp, pos)



#######################################################
# Parser.

class LCFRS_parser:
    # Constructor.
    # grammar: LCFRS
    # inp: list of string
    def __init__(self, grammar, inp):
        self.__g = grammar
        self.__inp = inp
        # Mapping from nonterminal and input position to nont items.
        self.__nont_items = defaultdict(list)
        # Mapping from nonterminal and input position to rule items.
        self.__rule_items = defaultdict(list)
        self.__agenda = []
        # To ensure item is not added a second time to agenda.
        self.__agenda_set = set()
        # Maps item to list of trace elements.
        self.__trace = defaultdict(list)
        # Maps each item to weight of best parse.
        self.__best = {}
        self.__parse()

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
                key = str(low) + ' ' + nont
                self.__nont_items[key].append(item)
                for rule_item in self.__rule_items[key]:
                    self.__combine(rule_item, item, rule_item.key(), str(item))
                for rule in self.__g.nont_corner_of(nont):
                    for inst in make_rule_instances(rule, inp):
                        self.__combine(inst, item, rule, str(item))
            else: # instance of Rule_instance
                (low, high) = item.next_member_bounds(inp_len)
                nont = item.next_nont()
                for pos in range(low, high+1):
                    key = str(pos) + ' ' + nont
                    self.__rule_items[key].append(item)
                    for nont_item in self.__nont_items[key]:
                        self.__combine(item, nont_item, item.key(), str(nont_item))

    # Combine rule item with nont item.
    # rule_item: Rule_instance
    # nont_item: LHS_instance
    # rule_trace: string or LCFRS_rule
    # nont_trace: string
    def __combine(self, rule_item, nont_item, rule_trace, nont_trace):
        lhs = rule_item.lhs().clone()
        dot = rule_item.dot()
        for i in range(nont_item.fanout()):
            arg = nont_item.arg(i)
            low = arg[0].low()
            high = arg[0].high()
            lhs.replace(dot, i, Span(low, high))
        if lhs.consistent():
            advanced_item = Rule_instance(rule_item.rule(), lhs, dot=dot+1)
            self.__record_item(advanced_item, (rule_trace, nont_trace))

    # item: Rule_instance
    # trace: pair or LCFRS_rule
    def __record_item(self, item, trace):
        if item.complete():
            lhs = item.lhs()
            lhs.collapse()
            key = str(lhs)
            if not key in self.__agenda_set:
                self.__agenda_set.add(key)
                self.__agenda.append(lhs)
            self.__trace[key].append(trace)
        else:
            key = item.key()
            if not key in self.__agenda_set:
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
        elem = str(start_lhs)
        trace = self.__trace[elem]
        if len(trace) == 0:
            return -1
        else:
            return self.__find_best_from(elem)
    # Find weight of best subderivation for all items, top-down.
    # elem: pair or string
    # return: float
    def __find_best_from(self, elem):
        if isinstance(elem, str) or isinstance(elem, unicode):
            if elem in self.__best:
                return self.__best[elem]
            else:
                self.__best[elem] = sys.float_info.max # avoid cycles
                traces = self.__trace[elem]
                best = sys.float_info.max
                for trace in traces:
                    best = min(best, self.__find_best_from(trace))
                self.__best[elem] = best
                return best
        elif isinstance(elem, tuple):
            return self.__find_best_from(elem[0]) + self.__find_best_from(elem[1])
        else:
            return -math.log(elem.weight())

    # Recognized?
    # return: bool
    def recognized(self):
        start_lhs = self.__start_item()
        elem = str(start_lhs)
        trace = self.__trace[elem]
        return len(trace) > 0


    def newDCP(self):
        der = self.newBestDerivation()
        if der:
            return The_DCP_evaluator(der).getEvaluation()
        else:
            return []


    def new_DCP_Hybrid_Tree(self, tree, poss, words, ignore_punctuation):
        dcp_evaluation = self.newDCP()
        if dcp_evaluation:
            return dcp_to_hybridtree(tree, dcp_evaluation, poss, words, ignore_punctuation)
        else:
            return None


    # Return best derivation or None.
    # return: Derivation
    def newBestDerivation(self):
        start_lhs = self.__start_item()
        elem = str(start_lhs)
        trace = self.__trace[elem]
        if len(trace) == 0:
            return None
        else:
            w = self.best()
            tree = Derivation()
            self.__newDerivationTreeRec(elem, tree, tree.root_id(), w, [Span(0, len(self.__inp))])
            return tree

    # Get derivation tree of best parse. (includes Spans of sub derivations)
    # elem: string or tuple (identifier in trace)
    # tree: Derivation (tree that gets extended)
    # id: string position (Gorn) in Derivation tree that is extended
    # w: float (weight of best path in subderivation)
    # spans: list of Span (of subderivation)
    # return: Derivation
    def __newDerivationTreeRec(self, elem, tree, id, w, spans):
        if isinstance(elem, str) or isinstance(elem, unicode):
            # passive item:
            traces = self.__trace[elem]
            for trace in traces:
                if w == self.__find_best_from(trace):
                    return self.__newDerivationTreeRec(trace, tree, id, w, spans)
            print 'looking for', w, 'found:'
            for trace in traces:
                print self.__find_best_from(trace)
            raise Exception('backtrace failed')
        elif isinstance(elem, tuple):
            # active item (elem[0]) was combined with passive item (elem[0])
            w1 = self.__find_best_from(elem[0])
            w2 = self.__find_best_from(elem[1])
            # extract span of the passive item
            sub_span = extract_spans(elem[1])
            # extend tree at child position corresponding to the dot of active item
            self.__newDerivationTreeRec(elem[1], tree, id + tree.gorn_delimiter() + str(dot_position(elem[0])), w2, sub_span)
            # extend active item
            self.__newDerivationTreeRec(elem[0], tree, id, w1, spans)
        else:
            # it all children have been added to derivation, add parent rule
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

# extract spans from key-string of passive item
# key: string (e.g. "A([0-4]; [12-15])" )
# return: list of Span
def extract_spans(key):
    spans = []
    match = re.search(r'^\s*.*\((\s*\[.*\]\s*)\)\s*$',key)
    for s in match.group(1).split('; '):
        match1 = re.search(r'^\[([0-9]+)-([0-9]+)\];*$', s)
        spans += [Span(int(match1.group(1)), int(match1.group(2)))]
    return spans