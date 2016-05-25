# Linear context-free rewriting system (LCFRS).
# Rules are augmented with DCP rules.
# Together this forms LCFRS/DCP hybrid grammars.

from collections import defaultdict, namedtuple
import codecs
import re
from time import time

from grammar.sDCP.dcp import parse_dcp, dcp_rules_to_str, dcp_rules_to_key

# ##########################################################################
# Parts of the grammar.

LCFRS_var = namedtuple('LCFRS_var', ['mem', 'arg'])

# Variable of LCFRS rule. 
# Represents i-th member in RHS and j-th argument thereof.
# class LCFRS_var:
# # Constructor.
# # i: int
# # j: int
#     def __init__(self, i, j):
#         self.__i = i
#         self.__j = j
#
#     # Member number part of variable.
#     # return: int
#     @property
#     def mem(self):
#         return self.__i
#
#     # Argument number part of variable.
#     # return: int
#     @property
#     def arg(self):
#         return self.__j
#
#     # String representation.
#     # return: string
#     def __str__(self):
#         return '<' + str(self.mem) + ',' + str(self.arg) + '>'
#
#     def __eq__(self, other):
#         # if isinstance(other, LCFRS_var):
#         return self.arg == other.arg and self.mem == other.mem
#         # else:
#         #     return False
#
#     def __hash__(self):
#         return hash((self.__i, self.__j))


# LHS of LCFRS rule.
class LCFRS_lhs:
    # Constructor.
    # nont: string
    def __init__(self, nont):
        self.__nont = nont
        self.__args = []

    # Add one argument.
    # arg: list of string and LCFRS_var
    def add_arg(self, arg):
        self.__args += [arg]

    # Number of arguments.
    # return: int
    def fanout(self):
        return len(self.__args)

    # Get nonterminal.
    # return: string
    def nont(self):
        return self.__nont

    # Get all arguments.
    # return: list of list of string/LCFRS_var
    def args(self):
        return self.__args

    # Get i-th argument.
    # i: int
    # return: list of string/LCFRS_var
    def arg(self, i):
        if i >= len(self.__args):
            pass
        return self.__args[i]

    # String representation.
    # return: string
    def __str__(self):
        s = self.nont() + '('
        for i in range(self.fanout()):
            arg = self.arg(i)
            for j in range(len(arg)):
                s += str(arg[j])
                if j < len(arg) - 1:
                    s += ' '
            if i < self.fanout() - 1:
                s += '; '
        s += ')'
        return s

    # Shorter string representation than above (fewer spaces).
    # return: string
    def key(self):
        s = self.nont() + '('
        for i in range(self.fanout()):
            arg = self.arg(i)
            for j in range(len(arg)):
                s += str(arg[j])
                if j < len(arg) - 1:
                    s += ' '
            if i < self.fanout() - 1:
                s += ';'
        s += ')'
        return s


# LCFRS rule, optionally with DCP rules.
class LCFRS_rule:
    # Constructor.
    # lhs: LCFRS_lhs
    # weight: real
    # dcp: list of DCP_rule
    def __init__(self, lhs, weight=1, dcp=None):
        self.__weight = weight
        self.__lhs = lhs
        self.__rhs = []
        self.__dcp = dcp

    # Add single RHS nonterminal.
    # nont: string
    def add_rhs_nont(self, nont):
        self.__rhs += [nont]

    # Increase weight.
    # weight: real
    def add_weight(self, weight):
        self.__weight += weight

    # Set DCP.
    # dcp: list of DCP_rule
    def set_dcp(self, dcp):
        self.__dcp = dcp

    # Set weight.
    # weight: real
    def set_weight(self, weight):
        self.__weight = weight

    # Get weight.
    # return: real
    def weight(self):
        return self.__weight

    # Get DCP.
    # return: list of DCP_rule
    def dcp(self):
        return self.__dcp

    # Get LHS.
    # return: LCFRS_lhs
    def lhs(self):
        """
        :return:
        :rtype: LCFRS_lhs
        """
        return self.__lhs

    # Get rank (length of RHS).
    # return: int
    def rank(self):
        return len(self.__rhs)

    # Get all RHS nonterminals.
    # return: list of string
    def rhs(self):
        return self.__rhs

    # Get i-th RHS nonterminal.
    # return: string
    def rhs_nont(self, i):
        return self.rhs()[i]

    # Size in terms of RHS length plus 1.
    # return: int
    def size(self):
        return 1 + len(self.rhs())

    # Get occurrences of terminals.
    # return: list of string
    def terms(self):
        terms = []
        for i in range(self.lhs().fanout()):
            for elem in self.lhs().arg(i):
                if isinstance(elem, str) or isinstance(elem, unicode):
                    terms += [elem]
        return terms

    # Return problem with rule if any.
    # fanout: mapping from nonterminals (string) to fanout (int).
    # return: string or None
    def well_formed(self, fanout):
        for i in range(self.rank()):
            nont = self.rhs_nont(i)
            if nont not in fanout:
                return 'lacks definition of nonterminal ' + nont
            nont_fanout = fanout[nont]
            variables = self.__get_vars(i)
            if variables != range(nont_fanout):
                return 'wrong variables in ' + str(self)
        return None

    def ordered(self):
        """
        :rtype: Bool
        :return: Do the variables of each rhs nonterminal occur in ascending order in the components on the lhs.
        """
        for mem in range(self.rank()):
            arg = -1
            for comp in range(self.lhs().fanout()):
                for obj in self.lhs().arg(comp):
                    if isinstance(obj, LCFRS_var):
                        if obj.mem == mem and arg + 1 != obj.arg:
                            return False
                        elif obj.mem == mem and arg + 1 == obj.arg:
                            arg += 1
        return True

    # Get variables from i-th member.
    # i: int 
    # return: list of int (argument numbers).
    def __get_vars(self, i):
        variables = []
        for j in range(self.lhs().fanout()):
            for elem in self.lhs().arg(j):
                if isinstance(elem, LCFRS_var) and elem.mem == i:
                    variables += [elem.arg]
        return variables

    # String representation.
    # return: string
    def __str__(self):
        s = '[' + str(self.weight()) + '] ' + str(self.lhs()) + ' -> '
        for i in range(self.rank()):
            s += self.rhs_nont(i)
            if i < self.rank() - 1:
                s += ' '
        if self.dcp() is not None:
            # s += '\n:: ' + dcp_rules_to_str(self.dcp())
            s += '\t\t' + dcp_rules_to_str(self.dcp())
        return s

    # Short string representation (without probability).
    # return: string
    def key(self):
        s = self.lhs().key() + '->'
        for i in range(self.rank()):
            s += self.rhs_nont(i)
            if i < self.rank() - 1:
                s += ' '
        if self.dcp() is not None:
            s += '::' + dcp_rules_to_key(self.dcp())
        return s

        # def __hash__(self):
        #     # TODO: there might be collisions. Since the number of rules is finite,
        #     # TODO: we can give every rule a unique number during training.
        #     return hash(self.key())


###########################################################################
# The grammar.

# LCFRS. 
# The start symbol can be explicitly set, or is determined by first rule
# that is added. Its fanout must be 1.
# Grammar is assumed to be monotone: no re-ordering of variables from LHS to
# RHS.
class LCFRS:
    # Constructor.
    # start: string
    # unit: real 
    def __init__(self, start=None, unit=1):
        # Unit weight; the 1 value of the (plus-times) semiring;
        # used as default weight of rules.
        self.__unit = unit
        # Mapping from nonterminal to (fixed) fanout.
        self.__nont_to_fanout = {}
        # Start symbol.
        self.__start = None
        # Rules, in order in which they were added.
        self.__rules = []
        # Mapping from nonterminal to list of rules with that nont as LHS.
        self.__lhs_nont_to_rules = defaultdict(list)
        # Mapping from string representation of rule (without weight) to rule
        # if it already exists.
        self.__key_to_rule = {}
        # Epsilon rules.
        self.__epsilon_rules = []
        # Mapping from terminal to lexical rules where terminal occurs as
        # first element.
        self.__first_term_of = defaultdict(list)
        # Mapping from nonterminal to rules where nonterminal occurs as
        # first element in RHS.
        self.__nont_corner_of = defaultdict(list)
        # Mapping from nonterminal A to nonterminals B
        # that may generate the first terminal A's first component
        self.__left_corner_of = None
        self.__full_left_corner = None
        self.__nont_lex_predict = None
        self.__left_branch_predict = None
        if start:
            self.__start = start
            self.__nont_to_fanout[start] = 1

    # Add rule to grammar.
    # lhs: 
    # nonts: list of string
    # weight: real 
    # dcp: list of DCP_rule
    # return: LCFRS_rule
    def add_rule(self, lhs, nonts, weight=None, dcp=None):
        if weight is None:
            weight = self.__unit
        rule = LCFRS_rule(lhs, weight=weight, dcp=dcp)
        for nont in nonts:
            rule.add_rhs_nont(nont)
        if rule.key() in self.__key_to_rule:
            rule = self.__key_to_rule[rule.key()]
            rule.add_weight(weight)
            return rule
        if not lhs.nont() in self.__nont_to_fanout or \
                        self.__nont_to_fanout[lhs.nont()] == lhs.fanout():
            self.__nont_to_fanout[lhs.nont()] = lhs.fanout()
        else:
            raise Exception('unexpected fanout in ' + str(rule))
        if lhs.fanout() == 0:
            raise Exception('0 fanout in ' + str(rule))
        self.__rules += [rule]
        self.__key_to_rule[rule.key()] = rule
        self.__lhs_nont_to_rules[rule.lhs().nont()] += [rule]
        if rule.rank() == 0:
            terms = rule.terms()
            if len(terms) > 0:
                self.__first_term_of[terms[0]] += [rule]
            else:
                self.__epsilon_rules += [rule]
        else:
            self.__nont_corner_of[rule.rhs_nont(0)] += [rule]
        if self.__start is None:
            self.__start = lhs.nont()
            if lhs.fanout() != 1:
                raise Exception('start symbol should have fanout 1')
        return rule

    # Get unit element.
    # return: real
    def unit(self):
        return self.__unit

    # Get start symbol.
    # return: string
    def start(self):
        return self.__start

    def rules(self):
        """
        :rtype: list[LCFRS_rule]
        :return: Get all rules in grammar.
        """
        return self.__rules

    # Get all nonterminals in grammar (LHS of rules).
    # return: list of LCFRS_rule
    def nonts(self):
        return self.__nont_to_fanout.keys()

    # Get total size of grammar.
    # return: int
    def size(self):
        n = 0
        for rule in self.rules():
            n += rule.size()
        return n

    # Maps nonterminal to fanout.
    # nont: string
    # return: int
    def fanout(self, nont):
        return self.__nont_to_fanout[nont]

    # Maps nonterminal to rules that have nonterminal as first
    # member in RHS.
    # nont: string
    # return: list of LCFRS_rule
    def nont_corner_of(self, nont):
        return self.__nont_corner_of[nont]

    # Return problems with grammar is any.
    # return: string or None
    def well_formed(self):
        for rule in self.rules():
            check = rule.well_formed(self.__nont_to_fanout)
            if check is not None:
                return check
        return None

    def ordered(self):
        for rule in self.rules():
            if not rule.ordered():
                return False, rule
        return True, None

    # Get zero-rank rules in which terminal is first terminal.
    # term: string
    # return: list of LCFRS_rule
    def lex_rules(self, term):
        return self.__first_term_of[term]

    # Get epsilon rules.
    # return: list of LCFRS_rule
    def epsilon_rules(self):
        return self.__epsilon_rules

    # Adjust weights to make grammar proper.
    def make_proper(self):
        for nont in self.__lhs_nont_to_rules:
            rules = self.__lhs_nont_to_rules[nont]
            if len(rules) > 0:
                total = sum([rule.weight() for rule in rules])
                for rule in rules:
                    rule.set_weight(1.0 * rule.weight() / total)

    # Join grammar into this.
    # other: LCFRS
    def add_gram(self, other):
        for rule in other.__rules:
            lhs = rule.lhs()
            nonts = rule.rhs()
            weight = rule.weight()
            dcp = rule.dcp()
            self.add_rule(lhs, nonts, weight=weight, dcp=dcp)

    # String representation. First print rules for start symbol.
    # Otherwise leave order unchanged.
    # return: string
    def __str__(self):
        s = ''
        for rule in self.__lhs_nont_to_rules[self.start()]:
            if rule.lhs().nont() == self.start():
                s += str(rule) + '\n'
        for rule in self.__rules:
            if rule.lhs().nont() != self.start():
                s += str(rule) + '\n'
        return s

    def lhs_nont_to_rules(self, nont):
        """
        :param nont:
        :rtype: list[LCFRS_rule]
        """
        return self.__lhs_nont_to_rules[nont]

    def init_left_corner(self):
        time1 = time()
        self.__left_corner_of = defaultdict(set)
        self.__left_corner_lex = defaultdict(list)
        changed = False
        for term in self.__first_term_of:
            for rule in self.__first_term_of[term]:
                nont = rule.lhs().nont()
                self.__left_corner_of[nont].add(nont)
                self.__left_corner_lex[nont, term].append(rule)
                changed = True

        while changed:
            changed = False
            for nont in self.__left_corner_of.keys():
                for rule in self.__nont_corner_of[nont]:
                    for lex_nont in self.__left_corner_of[nont]:
                        if lex_nont not in self.__left_corner_of[rule.lhs().nont()]:
                            self.__left_corner_of[rule.lhs().nont()].add(lex_nont)
                            changed = True
        print "Left-corner: ", time() - time1

    def full_left_corner(self, nontp):
        if not self.__left_corner_of:
            self.init_left_corner()
        if not self.__full_left_corner:
            self.__full_left_corner = defaultdict(list)
            for nont in self.nonts():
                for rule in self.lhs_nont_to_rules(nont):
                    if rule.rhs():
                        self.__full_left_corner[nont].append(rule.rhs_nont(0))
                        changed = True
            while changed:
                changed = False
                for nont in self.nonts():
                    for nont2 in self.full_left_corner(nont):
                        for nont3 in self.full_left_corner(nont2):
                            if nont3 not in self.full_left_corner(nont):
                                self.__full_left_corner[nont].append(nont3)
                                changed = True
        return self.__full_left_corner[nontp]

    def left_corner(self, nont):
        if not self.__left_corner_of:
            self.init_left_corner()
        return self.__left_corner_of[nont]

    def nont_lex(self, nont, term):
        return self.__left_corner_lex[nont, term]

    def nont_lex_predict(self, nont_p, term_p):
        if not self.__nont_lex_predict:
            start = time()
            lc_tmp = defaultdict(set)

            changed = False
            for term in self.__first_term_of:
                for rule in self.__first_term_of[term]:
                    nont = rule.lhs().nont()
                    if term not in lc_tmp[nont]:
                        lc_tmp[nont].add(term)
                        changed = True
            while changed:
                changed = False
                for term in self.__first_term_of:
                    for nont in lc_tmp.keys():
                        for rule in self.__nont_corner_of[nont]:
                            for term in lc_tmp[nont]:
                                if term not in lc_tmp[rule.lhs().nont()]:
                                    lc_tmp[rule.lhs().nont()].add(term)
                                    changed = True

            self.__nont_lex_predict = defaultdict(list)

            for nont in self.nonts():
                for rule in self.lhs_nont_to_rules(nont):
                    if rule.rhs():
                        for term in lc_tmp[rule.rhs_nont(0)]:
                            self.__nont_lex_predict[(nont, term)].append(rule)
                    else:
                        self.__nont_lex_predict[(nont, rule.lhs().arg(0)[0])].append(rule)

            print "nont lex prediction table: ", time() - start
        return self.__nont_lex_predict[(nont_p, term_p)]

    def left_branch_predict(self, nont_p, term_p):
        if not self.__left_branch_predict:
            start = time()
            rc_tmp = defaultdict(set)

            changed = False
            for term in self.__first_term_of:
                for rule in self.__first_term_of[term]:
                    nont = rule.lhs().nont()
                    if term not in rc_tmp[nont]:
                        rc_tmp[nont].add(term)
                        changed = True

            nt_right_corner = defaultdict(list)
            for rule in self.rules():
                if rule.rhs():
                    nt_right_corner[rule.rhs_nont(-1)].append(rule)

            while changed:
                changed = False
                for term in self.__first_term_of:
                    for nont in rc_tmp.keys():
                        for rule in nt_right_corner[nont]:
                            for term in rc_tmp[nont]:
                                if term not in rc_tmp[rule.lhs().nont()]:
                                    rc_tmp[rule.lhs().nont()].add(term)
                                    changed = True

            self.__left_branch_predict = defaultdict(list)

            for nont in self.nonts():
                for rule in self.lhs_nont_to_rules(nont):
                    if rule.rhs():
                        for term in rc_tmp[rule.rhs_nont(-1)]:
                            self.__left_branch_predict[(nont, term)].append(rule)
                    else:
                        self.__left_branch_predict[(nont, rule.lhs().arg(0)[0])].append(rule)
            print "left branching prediction table", time() - start
        return self.__left_branch_predict[(nont_p, term_p)]

############################################################
# Reading in grammar.

# Read grammar from file.
# file_name: string
# return: LCFRS
def read_LCFRS(file_name):
    gram_file = codecs.open(file_name, encoding='iso-8859-1')
    g = LCFRS()
    last_rule = None
    for line in gram_file:
        match_lcfrs = re.search(r'^\s*\[\s*([0-9\.]+)\s*\](.*)->(.*)$', line)
        match_dcp = re.search(r'^\s*::(.*)$', line)
        if match_lcfrs:
            # A(<1,2> foo <3,4> bar ; <4,2> foo <2,1> bar) -> B C D E
            w = float(match_lcfrs.group(1))
            lhs = read_lhs(match_lcfrs.group(2))
            rhs = read_rhs(match_lcfrs.group(3))
            last_rule = g.add_rule(lhs, rhs, weight=w)
        elif match_dcp:
            dcp_str = match_dcp.group(1)
            dcp = parse_dcp(dcp_str)
            if last_rule is not None:
                last_rule.set_dcp(dcp)
                last_rule = None
        elif not re.search(r'^\s*$', line):
            raise Exception('Strange line: ' + line)
    return g


# Read LHS of LCFRS rule. E.g.
# A(<1,2> foo <3,4> bar ; <4,2> foo <2,1> bar)
# s: string
# return: LCFRS_lhs
def read_lhs(s):
    match = re.search(r'^\s*(\S+)\((.*)\)\s*$', s)
    if match:
        nont = match.group(1)
        rest = match.group(2)
        lhs = LCFRS_lhs(nont)
        for arg_s in rest.split(';'):
            lhs.add_arg(read_arg(arg_s.strip()))
        return lhs
    else:
        raise Exception('Strange LHS: s')


# Read LHS argument of LCFRS rule. E.g.
# <1,2> foo <3,4> bar
# s: string
# return: list of variables (LCFRS_var) and terminals (string).
def read_arg(s):
    arg = []
    for elem in s.split(' '):
        elem = elem.strip()
        match = re.search(r'^<([0-9]+),([0-9]+)>$', elem)
        if match:
            # e.g. <1,2>
            i = match.group(1)
            j = match.group(2)
            arg += [LCFRS_var(int(i), int(j))]
        else:
            if re.search(r'^\S+$', elem):
                arg += [elem]
    return arg


# Read RHS of LCFRS rule with nonterminals. E.g.
# A B C
# s: string
# return: list of string
def read_rhs(s):
    rhs = []
    for nont in s.split(' '):
        nont = nont.strip()
        if re.search(r'^\S+$', nont):
            rhs += [nont]
    return rhs


#######################################################
# Testing.

def test_lcfrs():
    g = read_LCFRS('examples/testgram.gra')
    print g

# test_lcfrs()
