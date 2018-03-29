from abc import abstractmethod
from collections import defaultdict

from grammar.lcfrs import LCFRS_var
from hybridtree.general_hybrid_tree import HybridTree
from grammar.derivation_interface import AbstractDerivation


class LCFRSDerivation(AbstractDerivation):
    @abstractmethod
    def root_id(self):
        pass

    @abstractmethod
    def getRule(self, id):
        pass

    @abstractmethod
    def child_ids(self, id):
        pass

    @abstractmethod
    def child_id(self, id, i):
        pass

    @abstractmethod
    def position_relative_to_parent(self, id):
        pass

    @abstractmethod
    def ids(self):
        pass

    def spanned_ranges(self, id):
        if not self.spans:
            self._compute_spans()
        return [tuple(s) for s in self.spans[id]]

    def spanned_positions(self, id):
        if not self.spans:
            self._compute_spans()
        return [x + 1 for (l, r) in self.spans[id] for x in range(l, r)]

    def terminal_positions(self, id):
        if not self.spans:
            self._compute_spans()

        own = self.spanned_positions(id)
        children = [x for cid in self.child_ids(id) for x in self.spanned_positions(cid)]
        return [x for x in own if not x in children]

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


class LCFRSDerivationWrapper(LCFRSDerivation):
    def root_id(self):
        return self.__base_derivation.root_id()

    def getRule(self, id):
        return self.__base_derivation.getRule(id)

    def child_ids(self, id):
        return self.__base_derivation.child_ids(id)

    def child_id(self, id, i):
        return self.__base_derivation.child_id(id, i)

    def position_relative_to_parent(self, id):
        return self.__base_derivation.position_relative_to_parent(id)

    def ids(self):
        return self.__base_derivation.ids()

    def __init__(self, base_derivation):
        self.__base_derivation = base_derivation
        self.spans = None


def derivation_to_hybrid_tree(der, poss, ordered_labels, construct_token, disconnected=None):
    """
    :param der:
    :type der: LCFRSDerivation
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
