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

    def terminal_positions(self, id):
        if not self.spans:
            self._compute_spans()

        def spanned_positions(id_):
            return [x + 1 for (l, r) in self.spans[id_] for x in range(l, r)]

        own = spanned_positions(id)
        children = [x for cid in self.child_ids(id) for x in spanned_positions(cid)]
        return [x for x in own if not x in children]

    @abstractmethod
    def ids(self):
        """
        :rtype: list[object]
        """
        pass

    def __str__(self):
        if not self.spans:
            self._compute_spans()
        return self.der_to_str_rec(self.root_id(), 0)

    def der_to_str_rec(self, item, indentation):
        s = ' ' * indentation * 2 + str(self.getRule(item)) + '\t(' + str(self.spans[item]) + ')\n'
        for child in self.child_ids(item):
            s += self.der_to_str_rec(child, indentation + 1)
        return s

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

    def compute_yield(self):
        return self._compute_yield_recursive(self.root_id(), 0)

    def _compute_yield_recursive(self, id, k):
        the_yield = []
        arg = self.getRule(id).lhs().arg(k)
        for elem in arg:
            if isinstance(elem, LCFRS_var):
                child_id = self.child_id(id, elem.mem)
                the_yield += self._compute_yield_recursive(child_id, elem.arg)
            else:
                the_yield.append(elem)
        return the_yield

    def __eq__(self, other):
        if not isinstance(other, AbstractDerivation):
            return False
        return self.__compare_recursive(self.root_id(), other, other.root_id())

    def __compare_recursive(self, id_self, other, id_other):
        if self.getRule(id_self) != other.getRule(id_other):
            return False
        if len(self.child_ids(id_self)) != len(other.child_ids(id_other)):
            return False
        for child_self, child_other in zip(self.child_ids(id_self), other.child_ids(id_other)):
            if not self.__compare_recursive(child_self, other, child_other):
                return False
        return True

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
