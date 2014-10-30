# Parsing of string with LCFRS/DCP hybrid grammar.

from collections import defaultdict
import codecs
import re
import math
import sys

from lcfrs import *
from dcp import *
# from hybridtree import HybridTree
from constituency_tree import HybridTree
from decomposition import expand_spans
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

def test_LHS_instance():
    lhs = LHS_instance("A")
    lhs.add_arg()
    for mem in [Span(0,2),"blub",Span(3,4)]:
        lhs.add_mem(mem)
    print lhs.consistent()
    # print lhs.next_member_bounds(0,3)
    lhs = LHS_instance("A")
    lhs.add_arg()
    for mem in [Span(0,2),Span(3,4)]:
        lhs.add_mem(mem)
    print lhs.consistent()
    print lhs.next_member_bounds(0,3)
    lhs = LHS_instance("A")
    lhs.add_arg()
    for mem in [Span(0,2),Span(2,4)]:
        lhs.add_mem(mem)
    lhs.add_arg()
    for mem in [Span(6,9),Span(9,12)]:
        lhs.add_mem(mem)
    print lhs.consistent()
    print lhs
    lhs.collapse()
    print lhs

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

    # Replace variable for i-th member, j-th argument by span.
    # i: int
    # j: int
    # span: Span
    # TODO: the naming of the variables is pretty confusing!
    # TODO: Replace j-th variable of i-th non-terminal by span
    # def replace(self, i, j, span):
    #     new_args = []
    #     for arg in self.__args:
    #         new_arg = []
    #         for mem in arg:
    #             if isinstance(mem, LCFRS_var) and \
    #                             mem.mem() == i and mem.arg() == j:
    #                 new_arg += [span]
    #             else:
    #                 new_arg += [mem]
    #         new_args += [new_arg]
    #     self.__args = new_args

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
    # def __str__(self):
    #     s = self.nont() + '('
    #     for i in range(self.fanout()):
    #         arg = self.arg(i)
    #         for j in range(len(arg)):
    #             s += str(arg[j])
    #             if j < len(arg)-1:
    #                 s += ' '
    #         if i < self.fanout()-1:
    #             s += '; '
    #     return s + ')'
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
class Derivation:
    def gorn_delimiter(self):
        return '.'

    def gorn_delimiter_regex(self):
        return '\.'

    def __init__(self):
        self.__rules = {}
        self.__root = ''
        self.__weights = {}

    # add a rule to the derivation at position id
    # id : string (Gorn position / identifier)
    # rule: Rule_instance
    def add_rule(self, id, rule, weight):
        self.__rules[id] = rule
        self.__weights[id] = weight

    def getRule(self, id):
        return self.__rules[id]

    # id : string
    # return: list of Rule_instance
    def children(self, id):
        return [self.getRule(id + self.gorn_delimiter() + str(i))
                for i in range(self.getRule(id).rule().rank())]

    def child_ids(self, id):
        return [id + self.gorn_delimiter() + str(i) for i in range(self.getRule(id).rule().rank())]

    def root_id(self):
        return self.__root

    def root(self):
        return (self.getRule(''))

    def __str__(self):
        return der_to_str(self)

    def terminal_positions(self, id):
        child_positions = []
        for child in self.child_ids(id):
            child_positions += self.__all_positions(child)
        return [p for p in self.__all_positions(id) if p not in child_positions]

    def __all_positions(self, id):
        rule = self.getRule(id)
        positions = []
        for i in range(rule.lhs().fanout()):
            span = rule.lhs().arg(i)[0]
            positions += range(span.low() + 1, span.high() + 1)
        return positions

    def ids(self):
        return self.__rules.keys()


# return string
def der_to_str(der):
    return der_to_str_rec(der, der.root_id())

# return: string
def der_to_str_rec(der, id):
    s = ' ' * len(id) + str(der.getRule(id).rule()) + '\t(' + str(der.getRule(id).lhs()) + ')\n'
    for child in der.child_ids(id):
        s += der_to_str_rec(der, child)
    return s

# Turn a derivation tree into a hybrid tree.
# Assuming poss and ordered_labels to have equal length.
# der: Derivation
# poss: list of string (POS-tags)
# ordered_labels: list of words
# disconnected: list of positions in ordered_labels that are disconnected
# return: GeneralHybridTree
def derivation_to_hybrid_tree(der, poss, ordered_labels, disconnected = []):
    tree = GeneralHybridTree()
    j = 1
    for i in range(len(ordered_labels)):
        if i in disconnected:
            tree.add_node("d" + str(i), ordered_labels[i], poss[i], True, False)
        else:
            tree.add_node("c" + str(j), ordered_labels[i], poss[i], True, True)
            j += 1
    for id in der.ids():
        tree.add_node(id, der.getRule(id).lhs().nont())
        for child in der.child_ids(id):
            tree.add_child(id,child)
        for position in der.terminal_positions(id):
            tree.add_child(id, "c" + str(position))
    tree.set_root('')
    tree.reorder()
    return tree


#######################################################
# Old Derivation.
# A derivation is (rule, children).

# Get spans. So (rule, children) is turned into
# (rule, list of spans, children).
# der: 'derivation'
# return: triple 
def add_spans_to_derivation(der):
    der_len = add_lengths_to_derivation(der)
    return add_spans_to_derivation_recur(der_len, [0])
# Assuming starts of spans, add spans to derivation.
# der: triple
# starts: list of int
# return: triple
def add_spans_to_derivation_recur(der, starts):
    (rule, lengths, children) = der
    spans = []
    start_map = {} # maps (i,j) to start of i-th member j-th arg
    for k,arg in enumerate(rule.lhs().args()):
	pos = starts[k]
	for mem in arg:
	    if isinstance(mem, str) or isinstance(mem, unicode):
		pos += 1
	    else:
		start_map[(mem.mem(),mem.arg())] = pos
		pos += children[mem.mem()][1][mem.arg()]
	spans += [(starts[k],pos-1)]
    children_spans = []
    for i,child in enumerate(children):
	child_starts = []
	for j in range(len(child[1])):
	    child_starts += [start_map[(i,j)]]
	child_spans = add_spans_to_derivation_recur(child, child_starts)
	children_spans += [child_spans]
    return (rule, spans, children_spans)
    
# Add length of spans. (rule, children) is turned into
# (rule, list of lengths of spans, children).
# der: 'derivation'
# return: triple
def add_lengths_to_derivation(der):
    (rule, children) = der
    lengths = []
    child_lengths = [add_lengths_to_derivation(child) \
	for child in children]
    for arg in rule.lhs().args():
	length = 0
	for mem in arg:
	    if isinstance(mem, str) or isinstance(mem, unicode):
		length += 1
	    else:
		length += child_lengths[mem.mem()][1][mem.arg()]
	lengths += [length]
    return (rule, lengths, child_lengths)

# Turn derivation into hybrid tree, assuming POSs and input words.
# der: 'derivation'
# poss: list of string
# words: list of string
# return: HybridTree
def derivation_to_hybridtree(der, poss, words):
    tree = HybridTree()
    for (i, (pos, word)) in enumerate(zip(poss, words)):
	tree.add_leaf(i, pos, word)
    with_spans = add_spans_to_derivation(der)
    (id, _) = derivation_to_hybridtree_recur(with_spans, tree, len(poss))
    tree.set_root(id)
    tree.reorder()
    return tree
# As above, recur, with identifiers starting at next_id.
# Return id of root node and next id.
# der: 'derivation'
# tree: HybridTree
# next_id: string
# return: pair of string 
def derivation_to_hybridtree_recur(der, tree, next_id):
    (rule, rule_spans, children) = der
    label = rule.lhs().nont()
    id = next_id
    next_id += 1
    tree.set_label(id, label)
    for child in children:
        (tree_child, next_id) = \
	    derivation_to_hybridtree_recur(child, tree, next_id)
	tree.add_child(id, tree_child)
    all_terms = set(expand_spans(rule_spans))
    child_terms = set([t for (_, child_spans, _) in children \
			for t in expand_spans(child_spans)])
    top_terms = all_terms - child_terms
    for term in top_terms:
	tree.add_child(id, term)
    return (id, next_id)

#####################################################
# DCP evaluation.

# Evaluate DCP rules to compute terms at root.
# der: 'derivation'
# return: list of DCP_term/DCP_pos
def derivation_to_dcp(der):
    with_spans = add_spans_to_derivation(der)
    return eval_dcp(with_spans, [], -1, 0)
# General DCP evaluation.
# der: 'derivation'
# ancestors: list of derivations
# mem: int
# arg: int
# return: list of DCP_term
def eval_dcp(der, ancestors, mem, arg):
    (rule, spans, children) = der
    for dcp_rule in rule.dcp():
	lhs = dcp_rule.lhs()
	rhs = dcp_rule.rhs()
	if lhs.mem() == mem and lhs.arg() == arg:
	    return [t for term in rhs \
			for t in eval_dcp_term(term, der, ancestors)]
    return []
# term: DCP_term/DCP_var/DCP_index
# der: 'derivation'
# ancestors: list of derivations
# return: list of DCP_term/DCP_pos
def eval_dcp_term(term, der, ancestors):
    (rule, spans, children) = der
    if isinstance(term, DCP_term):
        head = term.head()
        arg = term.arg()
        ground = [t for arg_term in arg \
                  for t in eval_dcp_term(arg_term, der, ancestors)]
        return [DCP_term(head, ground)]
    elif isinstance(term, DCP_var):
        mem = term.mem()
        arg = term.arg()
        if mem >= 0:
            return eval_dcp(children[mem], ancestors + [(der,mem)], -1, arg)
        else:
            if ancestors != []:
                prefix = ancestors[0:-1]
                (last_der,last_mem) = ancestors[-1]
                return eval_dcp(last_der, prefix, last_mem, arg)
            else:
                raise Exception('value outside derivation ' + term)
    elif isinstance(term, DCP_index):
        i = term.index()
        pos = position_of_terminal(i, rule, spans,
                                   [child_span for (_,child_span,_) in children])
        return [DCP_pos(pos)]
    else:
        raise Exception('strange term ' + term)

# With spans, compute the position of the i-th terminal in the rule.
# i: int
# rule: LCFRS_rule
# spans: list of pair of int
# child_spans: list of list of pair of int
# return: int
def position_of_terminal(i, rule, spans, child_spans):
    n_terms = 0
    for arg in range(rule.lhs().rank()):
	(low, high) = spans[arg]
	pos = low
	for mem in rule.lhs().arg(arg):
	    if isinstance(mem, str) or isinstance(mem, unicode):
		if n_terms == i:
		    return pos
		n_terms += 1
		pos += 1
	    else:
		(low, high) = child_spans[mem.mem()][mem.arg()]
		pos = high+1
    raise Exception('missing terminal in ' + str(rule))

# Turn DCP value into hybrid tree.
# dcp: list of DCP_term/DCP_pos
# poss: list of string
# words: list of string
def dcp_to_hybridtree(tree, dcp, poss, words, ignore_punctuation):
    if len(dcp) != 1:
        raise Exception('DCP has multiple roots')
    j = 0
    for (i, (pos, word)) in enumerate(zip(poss, words)):
    #    tree.add_leaf(str(i), pos, word)
        if ignore_punctuation and re.search('^\$.*$',pos):
            tree.add_node(str(i)+'p', word, pos, True, False)
        elif ignore_punctuation:
            tree.add_node(str(j), word, pos, True, True)
            j += 1
        else:
            tree.add_node(str(i), word, pos, True, True)
    (id, _) = dcp_to_hybridtree_recur(dcp[0], tree, len(poss))
    tree.set_root(id)
    tree.reorder()
    return tree
# As above, recur, with identifiers starting at next_id.
# Return id of root node and next id.
# dcp: list of DCP_term/DCP_pos
# tree: GeneralHybridTree
# next_id: string
# return: pair of string
def dcp_to_hybridtree_recur(dcp, tree, next_id):
    head = dcp.head()
    if isinstance(head, DCP_pos):
        # FIXME : inconsistent counting of positions in hybrid tree requires -1
        id = str(head.pos() - 1)
    elif isinstance(head, DCP_string):
        label = head
        id = str(next_id)
        next_id += 1
        tree.add_node(id, label)
        tree.set_label(id, label)
    else:
        raise Exception
    tree.set_dep_label(id, head.dep_label())
    for child in dcp.arg():
        (tree_child, next_id) = \
            dcp_to_hybridtree_recur(child, tree, next_id)
        tree.add_child(id, tree_child)
    return (id, next_id)

#######################################################
# New DCP evaluation

class The_DCP_evaluator(DCP_evaluator):
    # der: Derivation
    def __init__(self, der):
        self.__der = der
        # self.__evaluate(der.root_id())

    def getEvaluation(self):
        return self.__evaluate("", -1, 0)

    # General DCP evaluation.
    # id : position in derivation tree
    # mem: int
    # arg: int
    # return: list of DCP_term
    def __evaluate(self, id, mem, arg):
        rule = self.__der.getRule(id).rule()
        for dcp_rule in rule.dcp():
            lhs = dcp_rule.lhs()
            rhs = dcp_rule.rhs()
            if lhs.mem() == mem and lhs.arg() == arg:
                # return [t for term in rhs \
			     #    for t in self.__eval_dcp_term(term, id)]
                result = []
                for term in rhs:
                    evaluation = self.__eval_dcp_term(term, id)
                    result += evaluation
                return result

    # term: DCP_term/DCP_var
    # der: 'derivation'
    # return: list of DCP_term/DCP_pos
    def __eval_dcp_term(self, term, id):
        return term.evaluateMe(self, id)

    # Evaluation Methods for term-heads
    # s: DCP_string
    def evaluateString(self, s, id):
        return s

    # index: DCP_index
    def evaluateIndex(self, index, id):
        i = index.index()
        pos = sorted(self.__der.terminal_positions(id))[i]
        return DCP_pos(pos, index.dep_label())

    # term: DCP_term
    def evaluateTerm(self, term, id):
        head = term.head()
        arg  = term.arg()
        evaluated_head = head.evaluateMe(self, id)
        ground = [t for arg_term in arg for t in self.__eval_dcp_term(arg_term, id)]
        return [DCP_term(evaluated_head, ground)]

    def evaluateVariable(self, var, id):
        mem = var.mem()
        arg = var.arg()
        if mem >= 0:
            return self.__evaluate(id + self.__der.gorn_delimiter() + str(mem), -1, arg)
        else:
            match = re.search(r'^(.*)' + self.__der.gorn_delimiter_regex() + '([0-9]+)$' ,id)
            # print match.group(1), match.group(2)
            if match:
                return self.__evaluate(match.group(1), int(match.group(2)), arg)
            else:
                raise Exception('strange var ' + var)

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

    # Return best derivation or None.
    # return: derivation
    def __best_der(self):
        start_lhs = self.__start_item()
        elem = str(start_lhs)
        trace = self.__trace[elem]
        if len(trace) == 0:
            return None
        else:
            w = self.best()
            return self.__der_tree(elem, [], w)

    # Print trace of best parse.
    def print_parse(self):
        der = self.__best_der()
        if der:
            print_derivation(der)
        else:
            print 'no parses'

    # Get labelled spans of best parse.
    # return list of list of int/string
    def labelled_spans(self):
        spans = []
        der = self.__best_der()
        if der:
            der_with_spans = add_spans_to_derivation(der)
            labelled_spans_recur(der_with_spans, spans)
        return sorted(spans)

    # Get hybrid tree of best parse.
    # poss: list of string
    # words: list of string
    # return: HybridTree
    def best_tree(self, poss, words):
        der = self.__best_der()
        if der:
            return derivation_to_hybridtree(der, poss, words)
        else:
            return None

    # Get DCP value of best parse.
    # return: list of DCP_term/DCP_pos
    def dcp(self):
        der = self.__best_der()
        if der:
            return derivation_to_dcp(der)
        else:
            return []

    def newDCP(self):
        der = self.newBestDerivation()
        if der:
            return The_DCP_evaluator(der).getEvaluation()
        else:
            return []

    # Make hybrid tree out of DCP value.
    # poss: list of string
    # words: list of string
    # return: HybridTree
    def dcp_hybrid_tree(self, poss, words, ignore_punctuation):
        der = self.__best_der()
        if der:
            dcp = derivation_to_dcp(der)
            return dcp_to_hybridtree(dcp, poss, words, ignore_punctuation)
        else:
            return None

    def new_DCP_Hybrid_Tree(self, tree, poss, words, ignore_punctuation):
        dcp_evaluation = self.newDCP()
        if dcp_evaluation:
            return dcp_to_hybridtree(tree, dcp_evaluation, poss, words, ignore_punctuation)
        else:
            return None

    # Get trace of best parse in form of derivation tree.
    # elem: string or tuple
    # children: list of trace
    # w: float
    # return: pair
    def __der_tree(self, elem, children, w):
        if isinstance(elem, str) or isinstance(elem, unicode):
            traces = self.__trace[elem]
            for trace in traces:
                if w == self.__find_best_from(trace):
                    return self.__der_tree(trace, children, w)
            print 'looking for', w, 'found:'
            for trace in traces:
                print self.__find_best_from(trace)
            raise Exception('backtrace failed')
        elif isinstance(elem, tuple):
            w1 = self.__find_best_from(elem[0])
            w2 = self.__find_best_from(elem[1])
            child = self.__der_tree(elem[1], [], w2)
            return self.__der_tree(elem[0], [child] + children, w1)
        else:
            return (elem, children)


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



# Print derivation with indentation.
# der: derivation
def print_derivation(der):
    print_derivation_recur(der, 0)
# der: derivation
# level: int
def print_derivation_recur(der, level):
    (rule, children) = der
    print ' ' * level, str(rule)
    for child in children:
	print_derivation_recur(child, level+1)

# Turn derivation into set of labelled spans.
# der: derivation
# spans: list of lists of int/string
def labelled_spans_recur(der, spans):
    (rule, rule_spans, children) = der
    labelled_span = [rule.lhs().nont()]
    for rule_span in rule_spans:
	labelled_span += [rule_span[0], rule_span[1]]
    spans += [labelled_span]
    for child in children:
        labelled_spans_recur(child, spans)

#######################################################
# Testing.

def test_lcfrs():
    g = read_LCFRS('examples/testgram.gra')
    # inp = 'a f d g e f b c'.split()
    inp = 'a a b b c c d d'.split()
    # print str(g).encode('iso-8859-1')
    print g
    g.make_proper()
    print g
    # print g
    # print inp
    # rules = g.epsilon_rules()
    # rules = g.lex_rules('f')
    # for rule in rules:
	# print rule
	# instances = make_rule_instances(rule, ['f', 'f', 'f', 'e'])
	# for inst in instances:
	    # print inst
    p = LCFRS_parser(g, inp)
    print "Recognized?", p.recognized()
    if (p.recognized()):
        print p.best()
        p.print_parse()

        der = p.newBestDerivation()
        tree = derivation_to_hybrid_tree(der, inp, inp, [])
        print tree.labelled_spans()
        print tree.labelled_yield()
        print tree.unlabelled_structure()
    # p.print_parse()
    # print dcp_terms_to_str(p.dcp())

# test_lcfrs()
