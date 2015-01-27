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
        :rtype: Range
        """
        assert (isinstance(diff, int) and int > 0)
        self.right += diff

    def to_tuple(self):
        """
        :rtype: (int,int)
        :return:
        """
        return self.__left, self.__right


class ActiveItem:
    def __init__(self, rule, children, variables, dot_component, dot_position):
        """
        :param rule:
        :type rule: LCFRS_rule
        """
        self._rule = rule
        self._children = children
        self._variables = variables
        self._dot_component = dot_component
        self._dot_position = dot_position

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
        return self._variables[component]

    def set_range(self, variable, range):
        """
        :param variable:
        :type variable: LCFRS_var
        :param range:
        :type range: Range
        :return:
        """
        self._variables[variable] = range

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

    def rule_id(self):
        """
        :return:
        :rtype: int
        """
        return id(self._rule)


class ScanItem(ActiveItem):
    def process(self, parser):
        """
        :type parser: Parser
        :return:
        """
        c_index, j = self.dot_position()
        word_tuple_string = self._rule.lhs().arg(c_index)
        obj = word_tuple_string[j]
        current_component = LCFRS_var(0, c_index)
        current_range = self.range(current_component)

        if isinstance(obj, LCFRS_var):
            arg = obj.arg()
            index = obj.mem()
            nont = self._rule.rhs_nont(arg)
            # TODO
            found_variables = self._variables.get(arg)
            parser.predict(nont, index, current_range.right(), found_variables)

            self.__class__ = CombineItem
            parser.record_active_item(self)
        else:
            assert isinstance(obj, terminal_type)
            if parser.terminal(current_range.right()) == obj:
                # if terminal matches current input position
                current_range.extend(1)
                self._dot_position += 1

                if self._dot_position == len(word_tuple_string):
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

    def process(self, parser):
        c_index, j = self.dot_position()
        word_tuple_string = self._rule.lhs().arg(c_index)
        variable = word_tuple_string[j]
        assert isinstance(variable, LCFRS_var)
        current_component = LCFRS_var(0, c_index)
        current_position = self.range(LCFRS_var(0, c_index))

        # start positions of strings
        range_constraints = [self.range(LCFRS_var(variable.arg(), comp)).left() for comp in range(variable.mem())]
        # add end of range right to dot
        range_constraints += [current_position.right()]

        nont = self._rule.rhs_nont(current_component.arg())
        passive_items = parser.query_passive_items(nont, range_constraints)

        for item in passive_items:
            assert isinstance(item, PassiveItem)

            consistent = True

            for comp in range(variable.mem()):
                if item.range(comp) != self.range(LCFRS_var(variable.arg(), comp)):
                    consistent = False
                    break

            if consistent:
                # new child vector
                children = self._children[0:variable.arg() - 1] + [item] + self._children[variable.arg():]

                # new variables dict, set ranges for found variables
                variables = dict(self._variables)
                variables[current_component] = current_position.join(item.range(variable.mem()))
                variables[variable] = item.range(variable.mem())

                active_item = ScanItem(self._rule, children, variables, c_index, j + 1)
                if j + 1 == len(word_tuple_string):
                    passive_item = active_item.convert_to_passive_item()
                    parser.record_passive_item(passive_item)
                else:
                    parser.record_active_item(active_item)


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

    def rule(self):
        """
        :return:
        :rtype: LCFRS_rule
        """
        return self.__rule


class Parser():
    __active_items = defaultdict
    __passive_items = defaultdict

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
        self.predict(self.__grammar.start(), 0, 0, [])

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
        key = tuple([item.action_id(), item.nont(), item.rule_id(), item.dot_position()] + [
            item.range(LCFRS_var(0, c_index)).to_tuple() for c_index in range(item.dot_position()[1])])  # ( + 1? )
        if key in self.__active_items:
            pass
        else:
            self.__active_items[key] = None
            self.__agenda.append(item)

    def query_passive_items(self, nont, range_constraints):
        """

        :param nont:
        :type nont: nonterminal_type
        :param range_constraints:
        :type range_constraints: list(int)
        :return:
        :rtype: list(PassiveItem)
        """
        key = tuple([nont] + range_constraints)
        return self.__passive_items[key]

    def parse(self):
        while self.__agenda:
            item = self.__agenda.popleft()
            item.process(self)

    def predict(self, nont, component, input_position, found_variables):
        """
        :type nont: nonterminal_type
        :param component:
        :type component: int
        :param input_position:
        :type input_position: int
        :param found_variables:
        :type found_variables: list(Range)
        """
        assert len(found_variables) == component
        if component == 0:
            for rule in self.__grammar.nont_corner_of(nont):
                # TODO: filtering
                initial_range = Range(component, input_position)
                variables = defaultdict
                variables[LCFRS_var(0, 0)] = initial_range
                item = ScanItem(rule, [], variables, 0, 0)
                self.record_active_item(item)
        else:
            range_constraints = map(lambda x: x.left(), found_variables)
            for passive_item in self.query_passive_items(nont, range_constraints):
                assert isinstance(passive_item, PassiveItem)

                consistent = False
                for c_index in range(component):
                    if found_variables[c_index] != passive_item.range(c_index):
                        consistent = False
                        break

                if consistent:
                    variables = defaultdict
                    for c_index, r in enumerate(found_variables + [Range(component, input_position)]):
                        variables[LCFRS_var(0, c_index)] = r
                    for arg, child in enumerate(passive_item.children()):
                        assert isinstance(child, PassiveItem)
                        for c_index in range(child.complete_to()):
                            variables[LCFRS_var(arg + 1, c_index)] = child.range(c_index)

                    item = ScanItem(passive_item.rule(), list(passive_item.children()), variables, component, 0)

                    self.record_active_item(item)

    def terminal(self, position):
        """

        :param position:
        :type position: int
        :return:
        :rtype: terminal_type
        """
        return self.__word[position]