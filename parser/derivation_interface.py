__author__ = 'kilian'

from abc import ABCMeta, abstractmethod
from general_hybrid_tree import GeneralHybridTree

class AbstractDerivation:
    @abstractmethod
    def root_id(self):
        """
        :rtype: str
        """
        pass

    @abstractmethod
    def add_rule(self, id, rule, weight):
        pass

    @abstractmethod
    def getRule(self, id):
        pass

    @abstractmethod
    def child_ids(self, id):
        pass

    @abstractmethod
    def children(self, id):
        pass

    @abstractmethod
    def terminal_positions(self):
        pass

    @abstractmethod
    def ids(self):
        """
        :rtype: list[T]
        """
        pass

    @abstractmethod
    def __str__(self):
        pass


# Turn a derivation tree into a hybrid tree.
# Assuming poss and ordered_labels to have equal length.
# der: Derivation
# poss: list of string (POS-tags)
# ordered_labels: list of words
# disconnected: list of positions in ordered_labels that are disconnected
# return: GeneralHybridTree
def derivation_to_hybrid_tree(der, poss, ordered_labels, disconnected = []):
    """
    :param der:
    :type der: AbstractDerivation
    :param poss:
    :param ordered_labels:
    :param disconnected:
    :return:
    """
    tree = GeneralHybridTree()
    j = 1
    for i in range(len(ordered_labels)):
        if i in disconnected:
            tree.add_node("d" + str(i), ordered_labels[i], poss[i], True, False)
        else:
            tree.add_node("c" + str(j), ordered_labels[i], poss[i], True, True)
            j += 1
    for id in der.ids():
        tree.add_node(id, der.getRule(id).lhs().nont())
        for child in der.child_ids(id):
            tree.add_child(id,child)
        for position in der.terminal_positions(id):
            tree.add_child(id, "c" + str(position))
    tree.set_root(der.root_id())
    tree.reorder()
    return tree