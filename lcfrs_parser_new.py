__author__ = 'kilian'

from lcfrs import *
from collections import deque


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
        if range2.right() == self.left():
            return Range(self.left(), range2.right())
        else:
            return None

    def extend(self, diff):
        """

        :param diff:
        :type diff: int
        :return:
        """
        assert (isinstance(diff, int) and int > 0)
        self.right += diff


class ActiveItem:
    def __init__(self, rule):
        """
        :param rule:
        :type rule: LCFRS_rule
        :type grammar: LCFRS
        """
        self._rule = rule
        self._children = {}
        self._variables = {}
        self._dot_component
        self._dot_position

    def predict(self, grammar, nont_component, position):

        for c_index, component in enumerate(self._rule.lhs().args()):
            if c_index != nont_component:
                self._variables[LCFRS_var(0, c_index)] = component
            else:
                self._variables

        for index, nont in enumerate(self._rule.rhs()):
            for component in grammar.fanout(nont):
                self._variables[LCFRS_var(index, c_index)] = LCFRS_var(index, c_index)

    def dot_position(self):
        """
        :rtype: (int,int)
        :return:
        """
        return self._dot_component, self._dot_position

    def range(self, component):
        """

        :param component:
        :type component: LCFRS_var
        :return:
        :rtype: Range
        """
        assert False

    def set_range(self, variable, range):
        """
        :param variable:
        :type variable: LCFRS_var
        :param range:
        :type range: Range
        :return:
        """
        assert False

    def nont(self):
        """
        :rtype: nonterminal_type
        :return:
        """
        return self._rule.lhs().nont()

    def action_id(self):
        assert False

    def convert_to_passive_item(self):
        """
        :return:
        :rtype: PassiveItem
        """
        # since our grammar is ordered, we can assumed all components up to the dot component to be found
        ranges = [self.range(LCFRS_var(0, comp)) for comp in range(self._dot_component + 1)]

        item = PassiveItem(self._rule, ranges, self._children)
        return item


class ScanItem(ActiveItem):
    def process(self, grammar, parser):
        """
        :param grammar:
        :type grammar: LCFRS
        :type parser: Parser
        :return:
        """
        c_index, j = self.dot_position()
        word_tuple_string = self._rule.lhs().arg(c_index)
        obj = word_tuple_string[j]
        current_component = LCFRS_var(0, c_index)
        current_position = self.range(LCFRS_var(0, c_index))

        if isinstance(obj, LCFRS_var):
            arg = obj.arg()
            index = obj.mem()
            nont = self._rule.rhs_nont(arg)
            # TODO
            found_variables = self._variables.get(arg)
            for rule in grammar.nont_corner_of(nont):
                parser.predict(rule, arg, index, current_position.right(), found_variables)

            self.__class__ = CombineItem
            parser.record_active_item(self)
        else:
            assert isinstance(obj, terminal_type)
            if parser.terminal(current_position) == obj:
                # if terminal matches current input position

                self.set_range(current_component, current_position.extend(1))
                self._dot_position += 1

                if j == len(word_tuple_string):
                    # end of word tuple component reached:
                    item = self.convert_to_passive_item()
                    parser.record_passive_item(item)

                else:
                    # there are more variables or terminals in the current component
                    parser.record_active_item(self)


    def action_id(self):
        return 'S'


class CombineItem(ActiveItem):
    def action_id(self):
        return 'C'

    def process(self, grammar, parser):
        c_index, j = self.dot_position()
        word_tuple_string = self._rule.lhs().arg(c_index)
        variable = word_tuple_string[j]
        assert isinstance(variable, LCFRS_var)
        current_component = LCFRS_var(0, c_index)
        current_position = self.range(LCFRS_var(0, c_index))

        # start positions of strings
        range_constraints = [self.range(LCFRS_var(variable.arg(), comp))[0] for comp in range(variable.mem())]
        # add end of range right to dot
        range_constraints += [current_position]

        passive_items = parser.query_passive_items(self.nont(), range_constraints)

        for item in passive_items:
            assert isinstance(item, PassiveItem)

            consistent = True

            for comp in range(variable.mem()):
                if item.range(comp) != self.range(LCFRS_var(variable.arg(), comp)):
                    consistent = False
                    break

            if consistent:
                # TODO record new Active or Passive Item (Copy!)
                # TODO merge range, update range record
                # TODO set pointer to child
                # TODO update dot position
                assert False


class PassiveItem:
    __rule = None
    __children = None
    __ranges = []

    def __init__(self, rule, ranges, children):
        """
        :param rule:
        :type rule: LCFRS_rule
        :param ranges:
        :type ranges: list(pair(int))
        :param children:
        :type children: list(nont)
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


class Parser():
    __active_items = {}
    __passive_items = {}

    def __init__(self, grammar, word):
        """

        :param grammar:
        :type grammar: LCFRS
        :param word:
        :return:
        """
        self.__grammar = grammar
        self.__word = word
        self.__agenda = deque()
        self.__init_agenda()

        self.parse()

    def __init_agenda(self):
        pass
        # predict S-rules

    def record_passive_item(self, item):
        """

        :param item:
        :type item: PassiveItem
        :return:
        """
        key = tuple([item.nont()] + [item.range(c_index).left() for c_index in range(item.complete_to())])
        if key in self.__passive_items:
            self.__passive_items[key] += [item]
        else:
            self.__passive_items[key] = [item]


    def record_active_item(self, item):
        """

        :param item:
        :type item: ActiveItem
        :return:
        """
        key = tuple([item.nont()] + [item.range(c_index).left() for c_index in range(item.complete_to())])
        if key in self.__active_items:
            pass
        else:
            self.__active_items[key] = None
            self.__agenda.append(item)

    def query_passive_items(self, nont, range_constraints):
        key = tuple([nont] + range_constraints)
        return self.__passive_items[key]

    def parse(self):
        while self.__agenda:
            item = self.__agenda.popleft()
            item.process(self)
