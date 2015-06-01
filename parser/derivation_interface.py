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


def derivation_to_hybrid_tree(der, poss, ordered_labels, construct_token, disconnected=None):
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
        token = construct_token(ordered_labels[i], poss[i], True)
        if i in disconnected:
            tree.add_node("d" + str(i), token, True, False)
        else:
            tree.add_node("c" + str(j), token, True, True)
            j += 1
    for id in der.ids():
        token = construct_token(der.getRule(id).lhs().nont(), '_', False)
        tree.add_node(id, token)
        for child in der.child_ids(id):
            tree.add_child(id, child)
        for position in der.terminal_positions(id):
            tree.add_child(id, "c" + str(position))
    tree.add_to_root(der.root_id())
    tree.reorder()
    return tree