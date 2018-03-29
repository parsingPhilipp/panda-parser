__author__ = 'kilian'

from grammar.lcfrs import LCFRS_var
from grammar.lcfrs_derivation import LCFRSDerivation
from parser.active.parse_items import PassiveItem, Range
import collections


class DerivationItem(PassiveItem):
    def __init__(self, rule, variables):
        PassiveItem.__init__(self, rule, variables)
        self.__children = []

    def add_child(self, child):
        self.__children.append(child)

    def children(self):
        return self.__children

    def copy(self):
        item = DerivationItem(self._rule, self._variables)
        for child in self.children():
            item.add_child(child)
        return item


class Derivation(LCFRSDerivation):
    def weight(self):
        if self.__weight is None:
            self.__weight = 0
            for derivation_item in self.__derivationItems.values():
                assert isinstance(derivation_item, DerivationItem)
                self.__weight += derivation_item.rule().weight()
        return self.__weight

    def child_id(self, id, i):
        return (self.child_ids(id))[i]

    def position_relative_to_parent(self, id):
        parent = self.__parent[id]
        ith_child = self.child_ids(parent).index(id)
        assert parent is not None and ith_child is not None
        return parent, ith_child

    def __init__(self):
        self.__derivationItems = collections.defaultdict(list)
        self.__parent = collections.defaultdict()
        self.__children = collections.defaultdict()
        self.__counter = 0
        self.__weight = None
        self.__root = None

    def ids(self):
        return range(self.__counter)

    def root_id(self):
        return self.__root

    def getRule(self, id):
        passive_item = self.__derivationItems[id]
        assert isinstance(passive_item, DerivationItem)
        return passive_item.rule()

    def terminal_positions(self, id):
        passive_item = self.__derivationItems[id]
        assert isinstance(passive_item, DerivationItem)

        spanned_input_positions = self.__spanned_input_by(id)
        spanned_input_positions_of_children = [pos for child in self.child_ids(id) for pos in
                                               self.__spanned_input_by(child)]

        # FIXME: pos + 1 due to strange position counting in the hybridtree_generator (from derivation)
        return [pos + 1 for pos in spanned_input_positions if pos not in spanned_input_positions_of_children]

    def __spanned_input_by(self, id):
        passive_item = self.__derivationItems[id]
        assert isinstance(passive_item, DerivationItem)

        spanned_input_positions = []
        for component in range(passive_item.fanout()):
            r = passive_item.range(LCFRS_var(-1, component))
            assert isinstance(r, Range)
            spanned_input_positions += range(r.left, r.right)

        return spanned_input_positions

    def child_ids(self, id):
        return self.__children.get(id, [])

    def add_derivation_item(self, passive_item, parent=None):
        """
        :param passive_item:
        :type passive_item: DerivationItem
        :param parent:
        :type parent: int
        :return:
        """
        id = self.__counter
        self.__counter += 1

        if parent is not None:
            self.__parent[id] = parent
            if parent not in self.__children.keys():
                self.__children[parent] = [id]
            else:
                self.__children[parent].append(id)
        else:
            self.__root = id

        self.__derivationItems[id] = passive_item
        return id

    def __str__(self):
        return self.der_to_str_rec(self.root_id(), 0)

    # return: string
    def der_to_str_rec(self, id, indent):
        s = ' ' * indent * 2 + str(self.getRule(id)) + '\t(' + str(self.__derivationItems[id]) + ')\n'
        for child in self.child_ids(id):
            s += self.der_to_str_rec(child, indent + 1)
        return s


__all__ = ["Derivation", "DerivationItem"]