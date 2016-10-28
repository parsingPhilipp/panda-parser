__author__ = 'kilian'

from abc import ABCMeta, abstractmethod
from hybridtree.general_hybrid_tree import HybridTree
from grammar.lcfrs import LCFRS_rule, LCFRS_var
from collections import defaultdict


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
        """
        :param id:
        :type id:
        :return:
        :rtype: LCFRS_rule
        """
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

    def check_integrity_recursive(self, id, nonterminal=None):
        rule = self.getRule(id)
        if nonterminal is not None and not rule.lhs().nont() == nonterminal:
            return False
        if len(self.child_ids(id)) != len(rule.rhs()):
            return False
        for i,child in enumerate(self.child_ids(id)):
            if not self.check_integrity_recursive(child, rule.rhs_nont(i)):
                return False
        return True

    def _compute_spans(self):
        self.spans = defaultdict(list)
        self.spans[self.root_id()] = [[0, None]]
        self._compute_spans_recursive(self.root_id(), 0)

    def _compute_spans_recursive(self, id, k):
        arg = self.getRule(id).lhs().arg(k)
        try:
            pos = self.spans[id][k][0]
        except Exception:
            pass
        for elem in arg:
            if isinstance(elem, LCFRS_var):
                while len(self.spans[self.child_id(id, elem.mem)]) <= elem.arg:
                    self.spans[self.child_id(id, elem.mem)] += [[None, None]]
                self.spans[self.child_id(id, elem.mem)][elem.arg][0] = pos
                pos = self._compute_spans_recursive(self.child_id(id, elem.mem), elem.arg)
            else:
                pos += 1
        self.spans[id][k][1] = pos
        return pos


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
    tree = HybridTree()
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
