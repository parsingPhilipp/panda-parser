__author__ = 'kilian'

from grammar.lcfrs import LCFRS_var
from collections import namedtuple

nonterminal_type = str
terminal_type = str

Range = namedtuple('Range', ['left', 'right'])


def join(range1, range2):
    """
    :param range1:
    :type range1: Range
    :param range2:
    :type range2: Range
    :return:
    :rtype: Range
    """
    if range1.right == range2.left:
        return Range(range1.left, range2.right)
    else:
        return None


def length(range):
    return range.right - range.left


def extend(range, diff):
    """
    :param diff:
    :type diff: int
    :return:
    :rtype: Range
    """
    assert isinstance(diff, int) and diff > 0
    return Range(range.left, range.right + diff)


class PassiveItem:
    def __init__(self, rule, variables):
        """
        :param rule:
        :type rule: LCFRS_rule
        :param variables:
        :return:
        """
        self._rule = rule
        self._variables = variables

        # Caching some frequently needed values
        self.__complete_to = None
        self.__max_mem = None

    def fanout(self):
        return self._rule.lhs().fanout()

    def nont(self):
        """
        :rtype: nonterminal_type
        :return:
        """
        return self._rule.lhs().nont()

    def range(self, variable):
        """

        :param variable:
        :type variable: LCFRS_var
        :return:
        :rtype: Range
        """
        return self._variables[variable]

    def variables(self):
        return self._variables

    def complete_to(self):
        """
        :rtype: int
        :return:
        """
        if self.__complete_to is None:
            self.__complete_to = self.max_arg(-1)
        return self.__complete_to

    def rule(self):
        """
        :return:
        :rtype: LCFRS_rule
        """
        return self._rule

    def rule_id(self):
        """
        :return:
        :rtype: int
        """
        return id(self._rule)

    def action_id(self):
        """
        :return:
        :rtype: str
        """
        return 'P'

    def __str__(self):
        s = '{' + ','.join(
            ['{' + ','.join([str(self.range(LCFRS_var(-1, arg))) for arg in range(self.complete_to() + 1)]) + '}']
            + ['{' + ','.join([str(self.range(LCFRS_var(mem, arg))) for arg in range(self.max_arg(mem) + 1)]) + '}' for
               mem in range(self.max_mem() + 1)]) + '}'
        return '[' + self.action_id() + ':' + str(self._rule) + ':' + s + ']'

    def __eq__(self, other):
        # if id(self) == id(other):
        #     return True
        # if not isinstance(other, PassiveItem):
        #     return False
        if self.rule() != other.rule():
            return False
        # if self.complete_to() != other.complete_to():
        #     return False

        for var in self._variables.keys():
            if var in other.variables().keys():
                if self.range(var) != other.range(var):
                    return False
            else:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # def __hash__(self):
    # return hash(tuple(self.children()))

    def max_mem(self):
        if self.__max_mem is None:
            self.__max_mem = max([var.mem for var in self._variables.keys()])
        return self.__max_mem

    def max_arg(self, mem):
        args = [var.arg for var in self._variables.keys() if var.mem == mem]
        if args:
            return max(args)
        else:
            return -1

    def copy(self):
        item = PassiveItem(self._rule, self._variables)
        return item
