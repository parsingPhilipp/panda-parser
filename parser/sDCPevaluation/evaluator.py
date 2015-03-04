__author__ = 'kilian'

from grammar.sDCP.dcp import DCP_evaluator, DCP_term, DCP_pos, DCP_string
from parser.derivation_interface import AbstractDerivation
import re


class The_DCP_evaluator(DCP_evaluator):
    # der: Derivation
    def __init__(self, der):
        """
        :param der:
        :type der: AbstractDerivation
        :return:
        """
        self.__der = der
        # self.__evaluate(der.root_id())

    def getEvaluation(self):
        return self.__evaluate(self.__der.root_id(), -1, 0)

    # General DCP evaluation.
    # id : position in derivation tree
    # mem: int
    # arg: int
    # return: list of DCP_term
    def __evaluate(self, id, mem, arg):
        rule = self.__der.getRule(id)
        for dcp_rule in rule.dcp():
            lhs = dcp_rule.lhs()
            rhs = dcp_rule.rhs()
            if lhs.mem() == mem and lhs.arg() == arg:
                # return [t for term in rhs \
                # for t in self.__eval_dcp_term(term, id)]
                result = []
                for term in rhs:
                    evaluation = self.__eval_dcp_term(term, id)
                    result += evaluation
                return result

    # term: DCP_term/DCP_var
    # der: 'derivation'
    # return: list of DCP_term/DCP_pos
    def __eval_dcp_term(self, term, id):
        return term.evaluateMe(self, id)

    # Evaluation Methods for term-heads
    # s: DCP_string
    def evaluateString(self, s, id):
        return s

    # index: DCP_index
    def evaluateIndex(self, index, id):
        i = index.index()
        pos = sorted(self.__der.terminal_positions(id))[i]
        return DCP_pos(pos, index.dep_label())

    # term: DCP_term
    def evaluateTerm(self, term, id):
        head = term.head()
        arg = term.arg()
        evaluated_head = head.evaluateMe(self, id)
        ground = [t for arg_term in arg for t in self.__eval_dcp_term(arg_term, id)]
        return [DCP_term(evaluated_head, ground)]

    def evaluateVariable(self, var, id):
        mem = var.mem()
        arg = var.arg()
        if mem >= 0:
            return self.__evaluate(self.__der.child_id(id, mem), -1, arg)
        else:
            parent, ith_child = self.__der.position_relative_to_parent(id)
            return self.__evaluate(parent, ith_child, arg)


# Turn DCP value into hybrid tree.
# dcp: list of DCP_term/DCP_pos
# poss: list of string
# words: list of string
def dcp_to_hybridtree(tree, dcp, poss, words, ignore_punctuation):
    if len(dcp) != 1:
        raise Exception('DCP has multiple roots')
    j = 0
    for (i, (pos, word)) in enumerate(zip(poss, words)):
        # tree.add_leaf(str(i), pos, word)
        if ignore_punctuation and re.search('^\$.*$', pos):
            tree.add_node(str(i) + 'p', word, pos, True, False)
        elif ignore_punctuation:
            tree.add_node(str(j), word, pos, True, True)
            j += 1
        else:
            tree.add_node(str(i), word, pos, True, True)
    (id, _) = dcp_to_hybridtree_recur(dcp[0], tree, len(poss))
    tree.set_root(id)
    tree.reorder()
    return tree


# As above, recur, with identifiers starting at next_id.
# Return id of root node and next id.
# dcp: list of DCP_term/DCP_pos
# tree: GeneralHybridTree
# next_id: string
# return: pair of string
def dcp_to_hybridtree_recur(dcp, tree, next_id):
    head = dcp.head()
    if isinstance(head, DCP_pos):
        # FIXME : inconsistent counting of positions in hybrid tree requires -1
        id = str(head.pos() - 1)
    elif isinstance(head, DCP_string):
        label = head
        id = str(next_id)
        next_id += 1
        tree.add_node(id, label)
        tree.set_label(id, label)
    else:
        raise Exception
    tree.set_dep_label(id, head.dep_label())
    for child in dcp.arg():
        (tree_child, next_id) = \
            dcp_to_hybridtree_recur(child, tree, next_id)
        tree.add_child(id, tree_child)
    return id, next_id