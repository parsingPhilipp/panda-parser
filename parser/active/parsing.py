__author__ = 'kilian'

from lcfrs import *
from collections import deque
from parser.parser_interface import AbstractParser
from derivation import Derivation
from parse_items import *


class ActiveItem:
    def __init__(self, rule, children, variables, dot_component, dot_position, remaining_input):
        """
        :param rule:
        :type rule: LCFRS_rule
        """
        self._rule = rule
        self._children = children
        self._variables = variables
        self._dot_component = dot_component
        self._dot_position = dot_position
        self._remaining_input = remaining_input

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
        ranges = [self.range(LCFRS_var(-1, comp)) for comp in range(self._dot_component + 1)]

        item = PassiveItem(self._rule, ranges, self._children)
        return item

    def rule_id(self):
        """
        :return:
        :rtype: int
        """
        return id(self._rule)

    def __str__(self):
        s = '{' + ','.join(
            ['{' + ','.join([str(self.range(LCFRS_var(-1, arg))) for arg in range(self._dot_component + 1)]) + '}']
            + ['{' + ','.join([str(child.range(i)) for i in range(child.complete_to())]) + '}' for child in
               self._children if isinstance(child, PassiveItem)]) + '}'
        return '[' + self.action_id() + ':' + str(self._rule) + ':' + s + ': ' + str(self._remaining_input) + ']'

    def remaining_input(self):
        return self._remaining_input

    def max_mem(self):
        return max([var.mem() for var in self._variables.keys()])

    def max_arg(self, mem):
        args = [var.arg() for var in self._variables.keys() if var.mem() == mem]
        if args:
            return max(args)
        else:
            return -1



class ScanItem(ActiveItem):
    def process(self, parser):
        """
        :type parser: Parser
        """
        c_index, j = self.dot_position()
        word_tuple_string = self._rule.lhs().arg(c_index)
        obj = word_tuple_string[j]
        current_component = LCFRS_var(-1, c_index)
        current_range = self.range(current_component)

        if isinstance(obj, LCFRS_var):
            mem = obj.mem()
            index = obj.arg()
            nont = self._rule.rhs_nont(mem)
            # TODO
            found_variables = [self.range(LCFRS_var(mem, j)) for j in range(obj.arg())]

            remaining_input = self._remaining_input - number_of_consumed_terminals(self._rule, c_index,
                                                                                   self._dot_position, mem)

            predicted_new = parser.predict(nont, index, current_range.right(), remaining_input, found_variables)

            self.__class__ = CombineItem
            parser.record_active_item(self, False)

        else:
            assert isinstance(obj, terminal_type)
            if parser.in_input(current_range.right()) and parser.terminal(current_range.right()) == obj:
                # if terminal matches current input position
                current_range.extend(1)
                self._dot_position += 1
                self._remaining_input -= 1

                if self._dot_position == len(word_tuple_string):
                    # end of word tuple component reached:
                    item = self.convert_to_passive_item()
                    parser.record_passive_item(item)

                elif self._remaining_input > 0:
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
        current_component = LCFRS_var(-1, c_index)
        current_position = self.range(LCFRS_var(-1, c_index))

        # start positions of strings
        range_constraints = [self.range(LCFRS_var(variable.mem(), comp)).left() for comp in range(variable.arg())]
        # add end of range right to dot
        range_constraints += [current_position.right()]

        nont = self._rule.rhs_nont(variable.mem())
        passive_items = parser.query_passive_items(nont, range_constraints)

        for item in passive_items:
            assert isinstance(item, PassiveItem)

            consistent = True

            for comp in range(variable.arg()):
                if item.range(comp) != self.range(LCFRS_var(variable.mem(), comp)):
                    consistent = False
                    break

            if consistent:
                remaining_input = self._remaining_input - item.range(variable.arg()).length()
                if not (remaining_input > 0 or (remaining_input == 0 and j + 1 == len(word_tuple_string))):
                    continue

                # new child vector
                children = self._children[0:variable.mem()] + [item] + self._children[variable.mem() + 1:]

                # new variables dict, set ranges for found variables
                variables = dict(self._variables)
                new_position = current_position.join(item.range(variable.arg()))
                variables[current_component] = new_position
                variables[variable] = item.range(variable.arg())

                active_item = ScanItem(self._rule, children, variables, c_index, j + 1, remaining_input)
                if j + 1 == len(word_tuple_string):
                    passive_item = active_item.convert_to_passive_item()
                    parser.record_passive_item(passive_item)
                else:
                    parser.record_active_item(active_item)


class Parser(AbstractParser):
    def best_derivation_tree(self):
        root = None
        if root:
            derivation = Derivation()
            derivation.add_passive_item(root, None)
            return derivation
        else:
            return None

    def best(self):
        pass

    def recognized(self):
        for item in self.query_passive_items(self.__grammar.start(), [0]):
            if item.range(0) == Range(0, len(self.__word)):
                return True
        return False

    def __init__(self, grammar, word, debug=False):
        """

        :param grammar:
        :type grammar: LCFRS
        :param word:
        :return:
        """
        self.__debug = debug
        self.__grammar = grammar
        self.__word = word
        self.__active_items = defaultdict()
        self.__passive_items = defaultdict()
        self.__process_counter = 0

        self.__scan_agenda = deque()
        self.__combine_agenda = deque()
        self.__init_agenda()

        self.parse()

    def __init_agenda(self):
        self.predict(self.__grammar.start(), 0, 0, len(self.__word), [])

    def record_passive_item(self, item):
        """

        :param item:
        :type item: PassiveItem
        :return:
        """
        key = tuple([item.nont()] + [item.range(c_index).left() for c_index in range(item.complete_to())])
        if key in self.__passive_items:
            if not item in self.__passive_items[key]:
                self.__passive_items[key] += [item]
                if self.__debug:
                    print " recorded   ", item
            elif self.__debug:
                print " skipped    ", item
        else:
            self.__passive_items[key] = [item]
            if self.__debug:
                print " recorded   ", item

    def record_active_item(self, item, force=False):
        """

        :param item:
        :type item: ActiveItem
        :return:
        """
        # FIXME: these filters lead to incompleteness
        key = tuple([item.action_id(), item.nont(), item.rule_id(), item.dot_position(), item.remaining_input()] + [
            item.range(LCFRS_var(-1, c_index)).to_tuple() for c_index in range(item.dot_position()[0] + 1)] + [
                        item.range(LCFRS_var(mem, arg)).to_tuple() for mem in range(item.max_mem() + 1) for arg in
                        range(item.max_arg(mem) + 1)]
            )
        if isinstance(item, CombineItem):
            if (not force) and key in self.__active_items.keys():
                if self.__debug:
                    print " skipped    ", item
                    print "            ", key
                return False
            else:
                self.__active_items[key] = None
                self.__combine_agenda.append(item)
                if self.__debug:
                    print " recorded   ", item
                    print "            ", key
                return True
        else:
            assert isinstance(item, ScanItem)
            if key in self.__active_items.keys():
                if self.__debug:
                    print " skipped    ", item
                    print "            ", key
                return False
            else:
                self.__active_items[key] = None
                self.__scan_agenda.append(item)
                return True

    def query_passive_items(self, nont, range_constraints):
        """

        :param nont:
        :type nont: nonterminal_type
        :param range_constraints:
        :type range_constraints: list[int]
        :return:
        :rtype: list[PassiveItem]
        """
        key = tuple([nont] + range_constraints)
        return self.__passive_items.get(key, [])

    def parse(self):
        while self.__combine_agenda or self.__scan_agenda:
            while self.__scan_agenda:
                item = self.__scan_agenda.popleft()
                if self.__debug:
                    self.__process_counter += 1
                    if self.__process_counter % 100 == 0:
                        pass
                    print "process  {:>3d}".format(self.__process_counter), item

                item.process(self)
            if self.__combine_agenda:
                item = self.__combine_agenda.pop()
                if self.__debug:
                    self.__process_counter += 1
                    if self.__process_counter == 58:
                        pass
                    print "process  {:>3d}".format(self.__process_counter), item
                item.process(self)

    def predict(self, nont, component, input_position, remaining_input, found_variables):
        """
        :type nont: nonterminal_type
        :param component:
        :type component: int
        :param input_position:
        :type input_position: int
        :param found_variables:
        :type found_variables: list[Range]
        :rtype: Bool
        """
        assert len(found_variables) == component
        predicted_new = False
        if component == 0:
            for rule in self.__grammar.lhs_nont_to_rules(nont):
                # TODO: filtering
                if minimum_string_size(rule, 0) > remaining_input:
                    continue
                if not do_all_terminals_occur_in_input(rule, component, self.__word, input_position):
                    continue
                # TODO: filtering end
                initial_range = Range(input_position, input_position)
                variables = defaultdict()
                variables[LCFRS_var(-1, 0)] = initial_range
                item = ScanItem(rule, ['?'] * rule.rank(), variables, 0, 0, remaining_input)
                predicted_new = self.record_active_item(item) or predicted_new
        else:
            range_constraints = map(lambda x: x.left(), found_variables)
            for passive_item in self.query_passive_items(nont, range_constraints):
                assert isinstance(passive_item, PassiveItem)

                consistent = True
                for c_index in range(component):
                    if found_variables[c_index] != passive_item.range(c_index):
                        consistent = False
                        break

                if consistent:
                    # TODO: filtering
                    if minimum_string_size(passive_item.rule(), component) > remaining_input:
                        continue
                    if not do_all_terminals_occur_in_input(passive_item.rule(), component, self.__word, input_position):
                        continue
                    # TODO: filtering end
                    variables = defaultdict()
                    for c_index, r in enumerate(found_variables + [Range(input_position, input_position)]):
                        variables[LCFRS_var(-1, c_index)] = r
                    for arg, child in enumerate(passive_item.children()):
                        if isinstance(child, PassiveItem):
                            for c_index in range(child.complete_to()):
                                variables[LCFRS_var(arg, c_index)] = child.range(c_index)
                        else:
                            assert child == '?'

                    # children = []
                    children = list(passive_item.children())

                    item = ScanItem(passive_item.rule(), children, variables, component, 0,
                                    remaining_input)

                    predicted_new = self.record_active_item(item) or predicted_new
        return predicted_new

    def terminal(self, position):
        """

        :param position:
        :type position: int
        :return:
        :rtype: terminal_type
        """
        return self.__word[position]

    def in_input(self, position):
        """

        :param position:
        :type position: int
        :return:
        :rtype: Bool
        """
        return 0 <= position < len(self.__word)


def derivation_tree(derivation, item, parent):
    """
    :type derivation: Derivation
    :param item:
    :type item: PassiveItem
    :type parent: PassiveItem
    :return:
    """
    id = derivation.add_passive_item(item, parent)
    for child in item.children():
        derivation_tree(derivation, child, id)


def minimum_string_size(rule, start_component, end_component=None):
    """
    :param rule:
    :type rule: LCFRS_rule
    :param start_component:
    :type start_component: int
    :param end_component:
    :type end_component: int
    :return:
    """

    if end_component is None or end_component > rule.lhs().fanout():
        end_component = rule.lhs().fanout()

    size = 0

    for component_index in range(start_component, end_component):
        component = rule.lhs().arg(component_index)
        size += len(component)

    return size


def number_of_consumed_terminals(rule, start_component, start_position, current_mem, end_component=None):
    """

    :param rule:
    :type rule: LCFRS_rule
    :param start_component:
    :param start_position:
    :param end_component:
    :return:
    """
    if end_component is None or end_component > rule.lhs().fanout():
        end_component = rule.lhs().fanout()

    terminals = 0

    for component_index in range(start_component, end_component):
        word_tuple_component = rule.lhs().arg(component_index)
        start = 0
        if component_index == start_component:
            start = start_position
        for tuple_index in range(start, len(word_tuple_component)):
            if isinstance(word_tuple_component[tuple_index], LCFRS_var):
                if word_tuple_component[tuple_index].mem() != current_mem:
                    terminals += 1
            elif isinstance(word_tuple_component[tuple_index], terminal_type):
                terminals += 1

    return terminals


def do_all_terminals_occur_in_input(rule, start_component, input, input_index, end_component=None):
    if end_component is None or end_component > rule.lhs().fanout():
        end_component = rule.lhs().fanout()

    for component_index in range(start_component, end_component):
        word_tuple_component = rule.lhs().arg(component_index)
        for tuple_index in range(0, len(word_tuple_component)):
            if isinstance(word_tuple_component[tuple_index], LCFRS_var):
                input_index += 1
            elif isinstance(word_tuple_component[tuple_index], terminal_type):
                while input_index < len(input) and input[input_index] != word_tuple_component[tuple_index]:
                    input_index += 1
                input_index += 1

            if input_index > len(input):
                return False
    return True