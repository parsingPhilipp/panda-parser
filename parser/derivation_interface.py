__author__ = 'kilian'

from abc import ABCMeta, abstractmethod

from hybridtree.general_hybrid_tree import GeneralHybridTree


class AbstractDerivation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def root_id(self):
        """
        :rtype: str
        """
        pass


    @abstractmethod
    def getRule(self, id):
        pass

    @abstractmethod
    def child_ids(self, id):
        """
        :param id:
        :rtype: list[object]
        """
        pass


    @abstractmethod
    def child_id(self, id, i):
        pass

    @abstractmethod
    def position_relative_to_parent(self, id):
        """
        :param id:
        :type id: object
        :rtype: (object, int)
        """
        pass

    @abstractmethod
    def terminal_positions(self, id):
        """
        :param id:
        :type id: object
        :return:
        :rtype: list[object]
        """
        pass

    @abstractmethod
    def ids(self):
        """
        :rtype: list[object]
        """
        pass

    @abstractmethod
    def __str__(self):
        pass


def derivation_to_hybrid_tree(der, poss, ordered_labels, disconnected=None):
    """
    :param der:
    :type der: AbstractDerivation
    :param poss: list of POS-tags
    :type poss: list[str]
    :param ordered_labels: list of words
    :type ordered_labels: list[str]
    :param disconnected: list of positions in ordered_labels that are disconnected
    :type disconnected: list[object]
    :rtype: GeneralHybridTree
    Turn a derivation tree into a hybrid tree. Assuming poss and ordered_labels to have equal length.
    """
    if not disconnected:
        disconnected = []
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
            tree.add_child(id, child)
        for position in der.terminal_positions(id):
            tree.add_child(id, "c" + str(position))
    tree.set_root(der.root_id())
    tree.reorder()
    return tree