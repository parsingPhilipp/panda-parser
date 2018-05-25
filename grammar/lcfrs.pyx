# Linear context-free rewriting system (LCFRS).
# Rules are augmented with DCP rules.
# Together this forms LCFRS/DCP hybrid grammars.

from collections import defaultdict, namedtuple
from grammar.dcp import dcp_rules_to_str, dcp_rules_to_key
from grammar.rtg import RTG_like, RTG

# ##########################################################################
# Parts of the grammar.

# LCFRS_var = namedtuple('LCFRS_var', ['mem', 'arg'])

# Variable of LCFRS rule.
# Represents i-th member in RHS and j-th argument thereof.
cdef class LCFRS_var:
    cdef int _mem
    cdef int _arg
    # Constructor.
    def __init__(self, int mem, int arg):
        self._mem = mem
        self._arg = arg

    # Member number part of variable.
    # return: int
    @property
    def mem(self):
        return self._mem

    @mem.setter
    def mem(self, int value):
        self._mem = value

    # Argument number part of variable.
    # return: int
    @property
    def arg(self):
        return self._arg

    @arg.setter
    def arg(self, int value):
        self._arg = value

    # String representation.
    # return: string
    def __str__(self):
        return '<' + str(self.mem) + ',' + str(self.arg) + '>'

    def __eq__(self, other):
        if isinstance(other, LCFRS_var):
            return self.mem == other.mem and self.arg == other.arg
        else:
            return False

    def __hash__(self):
        return hash((self._mem, self._arg))


# LHS of LCFRS rule.
cdef class LCFRS_lhs:
    cdef str __nont
    cdef list __args

    # Constructor.
    # nont: string
    def __init__(self, nont):
        self.__nont = nont
        self.__args = []

    # Add one argument.
    # arg: list of string and LCFRS_var
    cpdef list add_arg(self, list arg):
        self.__args += [arg]

    # Number of arguments.
    # return: int
    cpdef int fanout(self):
        return len(self.__args)

    # Get nonterminal.
    # return: string
    cpdef str nont(self):
        return self.__nont

    # Get all arguments.
    # return: list of list of string/LCFRS_var
    cpdef list args(self):
        return self.__args

    # Get i-th argument.
    # i: int
    # return: list of string/LCFRS_var
    def arg(self, int i):
        if i >= len(self.__args):
            pass
        return self.__args[i]

    # String representation.
    # return: string
    def __str__(self):
        cdef int i
        cdef int j
        cdef str s

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
    cpdef str key(self):
        cdef int i
        cdef int j
        cdef str s

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
cdef class LCFRS_rule:
    cdef double __weight
    cdef LCFRS_lhs __lhs
    cdef list __rhs
    cdef list __dcp
    cdef int __idx

    cpdef int get_idx(self):
        return self.__idx

    cpdef int set_idx(self, int idx):
        self.__idx = idx

    # Constructor.
    # lhs: LCFRS_lhs
    # weight: real
    # dcp: list of DCP_rule
    def __init__(self, LCFRS_lhs lhs, double weight=1, dcp=None, int idx=0):
        self.__weight = weight
        self.__lhs = lhs
        self.__rhs = []
        self.__dcp = dcp
        self.__idx = idx

    # Add single RHS nonterminal.
    # nont: string
    cpdef void add_rhs_nont(self, str nont):
        self.__rhs += [nont]

    # Increase weight.
    # weight: real
    cpdef void add_weight(self, double weight):
        self.__weight += weight

    # Set DCP.
    # dcp: list of DCP_rule
    cpdef void set_dcp(self, list dcp):
        self.__dcp = dcp

    # Set weight.
    # weight: real
    cpdef void set_weight(self, double weight):
        self.__weight = weight

    # Get weight.
    # return: real
    cpdef double weight(self):
        return self.__weight

    # Get DCP.
    # return: list of DCP_rule
    cpdef list dcp(self):
        return self.__dcp

    # Get LHS.
    # return: LCFRS_lhs
    cpdef LCFRS_lhs lhs(self):
        """
        :return:
        :rtype: LCFRS_lhs
        """
        return self.__lhs

    # Get rank (length of RHS).
    # return: int
    cpdef int rank(self):
        return len(self.__rhs)

    # Get all RHS nonterminals.
    # return: list of string
    cpdef list rhs(self):
        return self.__rhs

    # Get i-th RHS nonterminal.
    # return: string
    cpdef str rhs_nont(self, int i):
        return self.__rhs[i]

    # Size in terms of RHS length plus 1.
    # return: int
    cpdef int size(self):
        return 1 + len(self.__rhs)

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
        cdef int i
        for i in range(self.rank()):
            nont = self.rhs_nont(i)
            if nont not in fanout:
                return 'lacks definition of nonterminal ' + nont
            nont_fanout = fanout[nont]
            variables = self.__get_vars(i)
            if variables != list(range(nont_fanout)):
                return 'wrong variables in ' + str(self)
        return None

    cpdef bint ordered(self):
        """
        :rtype: bool
        :return: Do the variables of each rhs nonterminal occur in ascending order in the components on the lhs.
        """
        cdef int arg
        cdef int mem
        cdef int comp
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
    cpdef list __get_vars(self, int i):
        cdef list variables = []
        cdef int j
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
    cpdef str key(self):
        cdef str s
        cdef int i
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
class LCFRS(RTG_like):
    def to_rtg(self):
        rtg = RTG(self.__start)
        for rule in self.rules():
            rtg.construct_and_add_rule(rule.lhs().nont(),
                                       rule.get_idx(),
                                       rule.rhs())
        return rtg

    def initial(self):
        return self.__start

    tmp = None
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
        # Mapping from rule idx to rule
        self.__idx_to_rule = {}
        # Epsilon rules.
        self.__epsilon_rules = []
        # Mapping from terminal to lexical rules where terminal occurs as
        # first element.
        self.__first_term_of = defaultdict(list)
        # Mapping from nonterminal to rules where nonterminal occurs as
        # first element in RHS.
        self.__nont_corner_of = defaultdict(list)
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
        """
        :type lhs: LCFRS_lhs
        :type nonts: list
        :type weight: double
        """
        if weight is None:
            weight = self.__unit
        rule = LCFRS_rule(lhs, weight=weight, dcp=dcp, idx=len(self.__idx_to_rule))
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
        for i,nont in enumerate(nonts):
            fanout_i = 0
            for arg in lhs.args():
                for elem in arg:
                    if isinstance(elem, LCFRS_var) and elem.mem == i:
                        fanout_i = max(fanout_i, elem.arg + 1)
            if not nont in self.__nont_to_fanout or \
                self.__nont_to_fanout[nont] == fanout_i:
                self.__nont_to_fanout[nont] = fanout_i
            else:
                raise Exception('unexpected fanout in ' + str(rule))
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
        self.__idx_to_rule[rule.get_idx()] = rule
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

    def rule_index(self, i=None):
        if i is None:
            return self.__idx_to_rule
        else:
            return self.__idx_to_rule[i]

    # Get all nonterminals in grammar (LHS of rules).
    # return: list of LCFRS_rule
    def nonts(self):
        return self.__nont_to_fanout  # .keys()

    # Get total size of grammar.
    # return: int
    def size(self):
        return sum([rule.size() for rule in self.rules()])

    # Maps nonterminal to fanout.
    # nont: string
    # return: int
    def fanout(self, str nont):
        return self.__nont_to_fanout[nont]

    # Maps nonterminal to rules that have nonterminal as first
    # member in RHS.
    # nont: string
    # return: list of LCFRS_rule
    def nont_corner_of(self, str nont):
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
    def lex_rules(self, str term):
        return self.__first_term_of[term]

    # Get epsilon rules.
    # return: list of LCFRS_rule
    def epsilon_rules(self):
        return self.__epsilon_rules

    def purge_rules(self, threshold, feature_log=None):
        """
        :param threshold: remove rules with probability <= threshold from grammar
        :type threshold: float
        """
        if feature_log is not None:
            assert False
        to_remove = []
        i = 0
        while i < len(self.__rules):
            rule = self.__rules[i]
            if rule.weight() <= threshold:
                to_remove.append(rule)
                self.__rules = self.__rules[:i] + self.__rules[i+1:]
            else:
                i += 1

        # remove rule from auxiliary structures
        for rule in to_remove:
            del self.__idx_to_rule[rule.get_idx()]
            self.__lhs_nont_to_rules[rule.lhs().nont()].remove(rule)
            if rule.rank() > 0:
                self.__nont_corner_of[rule.rhs_nont(0)].remove(rule)
            elif rule.rank() == 0:
                terms = rule.terms()
                if terms:
                    self.__first_term_of[terms[0]].remove(rule)

        # rebuild rule index
        new_index = {}
        next_idx = 0
        for idx in self.__idx_to_rule:
            new_index[next_idx] = self.__idx_to_rule[idx]
            new_index[next_idx].set_idx(next_idx)
            next_idx += 1
        self.__idx_to_rule = new_index
        # TODO: handle features by building a translation dict: old idx -> new idx


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
    def add_gram(self, other, feature_logging=None):
        if feature_logging is not None:
            selfLog = feature_logging[0]
            otherLog = feature_logging[1]
        for other_rule in other.__rules:
            lhs = other_rule.lhs()
            nonts = other_rule.rhs()
            weight = other_rule.weight()
            dcp = other_rule.dcp()
            self_rule = self.add_rule(lhs, nonts, weight=weight, dcp=dcp)

            if feature_logging is not None:
                for key in otherLog:
                    if key[0] == other_rule.get_idx():
                        selfLog[(self_rule.get_idx(),) + key[1:]] += otherLog[key]
                        selfLog[(lhs.nont(), key[1])] += otherLog[key]
                        # for entry in zip(nonts, list(key[2])):
                        #    selfLog[(entry[0],) + entry[1]] += 1

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


__all__ = ["LCFRS", "LCFRS_var", "LCFRS_lhs", "LCFRS_rule"]