# Definite clause program rules. A list of such rules is
# part of a LCFRS/DCP hybrid grammar rule.

import re

# abc (abstract base classes)
from abc import ABCMeta, abstractmethod

###########################################################################
# Parts of the rules.


# Common interface for all objects that occur on rhs of DCP_rules
class DCP_rhs_object:
    __metaclass__ = ABCMeta

    # evaluator: DCP_evaluator
    # id: string (gorn term of LCFRS-Derivation tree)
    @abstractmethod
    def evaluateMe(self, evaluator, id=None):
        """
        :type evaluator: DCP_evaluator
        :param id: node id (gorn term) in LCFRS-derivation tree
        :type id: str
        :return: evaluated DCP_rhs object
        """
        pass


# Interface for DCP_evaluation
class DCP_evaluator:
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluateString(self, s, id):
        """
        :type s: DCP_string
        :param id: node id (gorn term) in LCFRS-derivation tree
        :type id: str
        :return: evaluated DCP_string
        :rtype: DCP_string
        """
        pass

    @abstractmethod
    def evaluateIndex(self, index, id):
        """
        :type index: DCP_index
        :param id: node id (gorn term) in LCFRS-derivation tree
        :type id: str
        :return: the input position to which the index points
        :rtype: DCP_position
        """
        pass

    @abstractmethod
    def evaluateTerm(self, term, id):
        """
        :type term: DCP_term
        :param id: node id (gorn term) in LCFRS-derivation tree
        :type id: str
        :return: evaluated DCP_term
        :rtype: DCP_term
        """
        pass

    @abstractmethod
    def evaluateIndex(self, var, id):
        pass

    @abstractmethod
    def evaluateVariable(self, var, id):
        pass


# Variable identifying argument (synthesized or inherited).
# In LHS this is (-1,j) and in RHS this is (i,j),
# for i-th member in RHS, and j-th argument.
class DCP_var(DCP_rhs_object):
    # Constructor.
    # i: int
    # j: int
    def __init__(self, i, j):
        self.__i = i
        self.__j = j

    # Member number part of variable, or -1 for LHS.
    # return: int
    def mem(self):
        return self.__i

    # Argument number part of variable.
    # return: int
    def arg(self):
        return self.__j

    # String representation.
    # return: string
    def __str__(self):
        if self.mem() < 0:
            return '<' + str(self.arg()) + '>'
        else:
            return '<' + str(self.mem()) + ',' + str(self.arg()) + '>'

    def evaluateMe(self, evaluator, id=None):
        return evaluator.evaluateVariable(self, id)



# Index, pointing to terminal in left (LCFRS) component of hybrid grammar.
# Terminals are indexed in left-to-right order.
class DCP_index(DCP_rhs_object):
    # Constructor.
    # i: int
    # edge_label: string
    def __init__(self, i, edge_label=None):
        self.__i = i
        self.__edge_label = edge_label

    # The index.
    # return: int
    def index(self):
        return self.__i

    def edge_label(self):
        return self.__edge_label

    # String representation.
    # return: string
    def __str__(self):
        if self.__edge_label:
            s = ':{' + self.__edge_label + '}'
        else:
            s = ''
        return '[' + str(self.index()) + s + ']'

    # Evaluator Invocation
    def evaluateMe(self, evaluator, id=None):
        return evaluator.evaluateIndex(self, id)


# A terminal of DCP_rule that is not linked to some terminal
# in the LCFRS component of the hybrid grammar
class DCP_string(DCP_rhs_object):
    def __init__(self, string, edge_label=None):
        self.__string = string
        self.__edge_label = edge_label

    def set_edge_label(self, edge_label):
        self.__edge_label = edge_label

    def edge_label(self):
        return self.__edge_label

    # Evaluator invocation
    def evaluateMe(self, evaluator, id=None):
        return evaluator.evaluateString(self, id)

        # String representation.
        # return: string

    def get_string(self):
        return self.__string

    def __str__(self):
        if self.__edge_label:
            s = ':{' + self.__edge_label + '}'
        else:
            s = ''
        return self.__string + s

# An index replaced by an input position, according to parsing of a string with
# the left (LCFRS) component of hybrid grammar.
class DCP_position:
    # Constructor.
    # pos: int
    # edge_label: string
    def __init__(self, position, edge_label=None):
        self.__position = position
        self.__edge_label = edge_label

    # The position.
    # return: int
    def position(self):
        return self.__position

    def edge_label(self):
        return self.__edge_label

    # String representation.
    # return: string
    def __str__(self):
        if self.__edge_label:
            s = ':{' + self.__edge_label + '}'
        else:
            s = ''
        return '[' + str(self.position()) + s + ']'


# A terminal occurrence (may linked to LCFRS terminal),
# consisting of a DCP_string or DCP_index and a list of child terminal
# occurrences.
class DCP_term(DCP_rhs_object):
    # Constructor.
    # head: DCP_rhs_object (DCP_string / DCP_index)
    # arg: list of DCP_term/DCP_index TODO: outdated
    # arg: list of DCP_rhs_object (DCP_term + DCP_var)
    def __init__(self, head, arg):
        self.__head = head
        self.__arg = arg

    # The label.
    # return: string
    def head(self):
        return self.__head

    # The children.
    # return: list of DCP_term/DCP_index TODO: outdated
    # return: list of DCP_rhs_object (DCP_term / DCP_var)
    def arg(self):
        return self.__arg

    # String representation.
    # return: string
    def __str__(self):
        return str(self.head()) + '(' + dcp_terms_to_str(self.arg()) + ')'

    # Evaluator invocation
    def evaluateMe(self, evaluator, id=None):
        return evaluator.evaluateTerm(self, id)


# Rule defining argument value by term.
class DCP_rule:
    def __init__(self, lhs, rhs):
        """
        :type lhs: DCP_var
        :type rhs: list(DCP_rhs_object)
        """
        self.__lhs = lhs
        self.__rhs = rhs

    # The LHS.
    # return: DCP_var
    def lhs(self):
        return self.__lhs

    # The RHS.
    # return: list of DCP_term/DCP_index TODO: outdated
    # return: list of DCP_rhs_object
    def rhs(self):
        return self.__rhs

    # String representation.
    # return: string
    def __str__(self):
        return str(self.lhs()) + '=' + dcp_terms_to_str(self.rhs())


################################################################
# Auxiliary.

# Turn list of terms into string. The terms are separated by whitespace.
# l: list of DCP_term/DCP_index TODO: outdated
# l: list of DCP_rhs_object
#
# return: string
def dcp_terms_to_str(l):
    return ' '.join([str(o) for o in l])


# Turn list of DCP_rules into string. The rules are separated by semicolons.
# l: list of DCP_rule
# return: string
def dcp_rules_to_str(l):
    return '; '.join([str(r) for r in l])


# As above, but more compact, omitting whitespace.
# l: list of DCP_rule
# return: string
def dcp_rules_to_key(l):
    return ';'.join([str(r) for r in l])


##################################################
# Parsing of the DCP part of hybrid-grammar rules.

# Parse rules.
# s: string (of rules separated by semicolons)
# return: list of DCP_rule
def parse_dcp(s):
    return [parse_dcp_rule(rule_str) \
            for rule_str in s.split(';')]


# Parse rule.
# s: string
# return: DCP_rule
def parse_dcp_rule(s):
    parts = s.split('=')
    if len(parts) != 2:
        raise Exception('strange DCP rule: ' + s)
    lhs_str = parts[0]
    rhs_str = parts[1]
    (lhs, rest) = parse_dcp_var(lhs_str)
    if re.search(r'\S', rest):
        raise Exception('strange DCP rule: ' + s)
    (rhs, rest) = parse_dcp_terms(rhs_str)
    if re.search(r'\S', rest):
        raise Exception('strange DCP rule: ' + s)
    return DCP_rule(lhs, rhs)


# Parse variable. Return variable and rest of input.
# s: string
# return: pair of DCP_var and string (remainder)
def parse_dcp_var(s):
    match_lhs = re.search(r'^\s*<([0-9]+)>(.*)$', s)
    match_rhs = re.search(r'^\s*<([0-9]+),([0-9]+)>(.*)$', s)
    if match_lhs:
        arg = int(match_lhs.group(1))
        rest = match_lhs.group(2)
        return DCP_var(-1, arg), rest
    elif match_rhs:
        mem = int(match_rhs.group(1))
        arg = int(match_rhs.group(2))
        rest = match_rhs.group(3)
        return DCP_var(mem, arg), rest
    else:
        raise Exception('strange DCP var: ' + s)


# Read terms, separated by whitespace, until bracket close or until nothing
# left.
# s: string
# return: pair of list of DCP_rhs_object and string (remainder).
def parse_dcp_terms(s):
    terms = []
    while not re.search(r'^\s*$', s) and not re.search(r'^\s*\).*$', s):
        if re.search(r'^\s*<', s):
            (var, s) = parse_dcp_var(s)
            terms += [var]
        # elif re.search(r'^\s*\[', s):
        # (index, s) = parse_dcp_index(s)
        # terms += [index]
        else:
            # try to match a term starting with DCP_index at root
            match = re.search(r'^\s*(\[[0-9]+(:{.*})?\])\s*\((.*)', s)
            if match:
                (head, s) = parse_dcp_index(s)
            else:
                # try to match a term starting with DCP_string at root
                match = re.search(r'^\s*([^\s\(<\[]+)\((.*)$', s)
                if not match:
                    raise Exception('unexpected input in ' + s)
                (head, s) = parse_dcp_string(s)
            (child_terms, s) = parse_dcp_terms(s)
            terms += [DCP_term(head, child_terms)]
            close_match = re.search(r'^\s*\)(.*)$', s)
            if not close_match:
                raise Exception('missing close bracket in ' + s)
            else:
                s = close_match.group(1)
    return terms, s


# Parse index. Return index and rest of input.
# line: s
# return: pair of DCP_var and remainder of string
def parse_dcp_index(s):
    match = re.search(r'^\s*\[([0-9]+)(:{\w*})?\]\s*\((.*)$', s)
    if match:
        i = int(match.group(1))
        rest = match.group(3)
        dep_label = None
        if match.group(2):
            match2 = re.search(r'^:{(\w*)}$', match.group(2))
            if match2:
                dep_label = match2.group(1)
        return DCP_index(i, dep_label), rest
    else:
        raise Exception('strange DCP index: ' + s)


# Parse string, Return string and rest of input.
# s: string
# return: pair of DCP_string and remainder of string
def parse_dcp_string(s):
    match = re.search(r'^\s*([^\s\(<\[]+)\((.*)$', s)
    if match:
        return DCP_string(match.group(1)), match.group(2)
    else:
        raise Exception('strange DCP terminal: ' + s)


#######################################################
# Testing.

def test_dcp():
    inp = '<1>=NP(VP(<0,1> a()[0:{142_abAB}]( )<0,1>)Det()) [1](); <2>= [2:{}]() <1,0> '
    rules = parse_dcp(inp)
    print dcp_rules_to_str(rules)

# test_dcp()
