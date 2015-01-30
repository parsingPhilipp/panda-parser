__author__ = 'kilian'

nonterminal_type = str
terminal_type = str

class Range:
    def __init__(self, left, right):
        """

        :param left:
        :type left: int
        :param right:
        :type right: int
        :return:
        """
        self.__left = left
        self.__right = right
        if left > right or left < 0:
            assert (False and "Invalid range!")

    def left(self):
        return self.__left

    def right(self):
        return self.__right

    def join(self, range2):
        """
        :param range2:
        :type range2: Range
        :return:
        :rtype: Range
        """
        if self.right() == range2.left():
            return Range(self.left(), range2.right())
        else:
            return None

    def extend(self, diff):
        """

        :param diff:
        :type diff: int
        :return:
        :rtype: Range
        """
        assert isinstance(diff, int) and diff > 0
        self.__right += diff

    def to_tuple(self):
        """
        :rtype: (int,int)
        :return:
        """
        return self.__left, self.__right

    def __str__(self):
        return 'r<' + str(self.left()) + ',' + str(self.right()) + '>'

    def __eq__(self, other):
        assert isinstance(other, Range)
        return self.left() == other.left() and self.right() == other.right()

    def length(self):
        return self.__right - self.__left



class PassiveItem:
    __rule = None
    __children = None
    __ranges = []

    def __init__(self, rule, ranges, children):
        """
        :param rule:
        :type rule: LCFRS_rule
        :param ranges:
        :type ranges: list[Range]
        :param children:
        :type children: list[nonterminal_type]
        :return:
        """
        self.__rule = rule
        self.__children = children
        self.__ranges = ranges

    def fanout(self):
        return self.__rule.lhs().fanout()

    def nont(self):
        """
        :rtype: nonterminal_type
        :return:
        """
        return self.__rule.lhs().nont()

    def range(self, c_index):
        """

        :param c_index:
        :return:
        :rtype: Range
        """
        return self.__ranges[c_index]

    def children(self):
        return self.__children

    def complete_to(self):
        """
        :rtype: int
        :return:
        """
        return len(self.__ranges)

    def rule(self):
        """
        :return:
        :rtype: LCFRS_rule
        """
        return self.__rule

    def __str__(self):
        return '[P:' + str(self.rule()) + ':' + '{' + ','.join(map(str, self.__ranges)) + '}]'
