__author__ = 'kilian'

from parser.derivation_interface import AbstractDerivation
from parse_items import *
import collections


class Derivation(AbstractDerivation):
    def child_id(self, id, i):
        return (self.child_ids(id))[i]

    def position_relative_to_parent(self, id):
        parent = self.__parent[id]
        ith_child = self.child_ids(parent).index(id)
        assert parent is not None and ith_child is not None
        return parent, ith_child

    def __init__(self):
        self.__passiveItems = collections.defaultdict(list)
        self.__parent = collections.defaultdict()
        self.__children = collections.defaultdict()
        self.__counter = 0
        self.__root = None

    def ids(self):
        return range(self.__counter)

    def root_id(self):
        return self.__root

    def getRule(self, id):
        passive_item = self.__passiveItems[id]
        assert isinstance(passive_item, PassiveItem)
        return passive_item.rule()

    def terminal_positions(self, id):
        passive_item = self.__passiveItems[id]
        assert isinstance(passive_item, PassiveItem)

        spanned_input_positions = self.__spanned_input_by(id)
        spanned_input_positions_of_children = [pos for child in self.child_ids(id) for pos in
                                               self.__spanned_input_by(child)]

        # FIXME: pos + 1 due to strange position counting in the hybridtree_generator (from derivation)
        return [pos + 1 for pos in spanned_input_positions if pos not in spanned_input_positions_of_children]

    def __spanned_input_by(self, id):
        passive_item = self.__passiveItems[id]
        assert isinstance(passive_item, PassiveItem)

        spanned_input_positions = []
        for component in range(passive_item.fanout()):
            r = passive_item.range(LCFRS_var(-1, component))
            assert isinstance(r, Range)
            spanned_input_positions += range(r.left(), r.right())

        return spanned_input_positions

    def child_ids(self, id):
        return self.__children.get(id, [])

    def add_passive_item(self, passive_item, parent=None):
        """
        :param passive_item:
        :type passive_item: PassiveItem
        :param parent:
        :type parent: int
        :return:
        """
        id = self.__counter
        self.__counter += 1

        if not parent is None:
            self.__parent[id] = parent
            if not parent in self.__children.keys():
                self.__children[parent] = [id]
            else:
                self.__children[parent].append(id)
        else:
            self.__root = id

        self.__passiveItems[id] = passive_item
        return id

    def __str__(self):
        return self.der_to_str_rec(self.root_id(), 0)

    # return: string
    def der_to_str_rec(self, id, indent):
        s = ' ' * indent * 2 + str(self.getRule(id)) + '\t(' + str(self.__passiveItems[id]) + ')\n'
        for child in self.child_ids(id):
            s += self.der_to_str_rec(child, indent + 1)
        return s