__author__ = 'kilian'

from corpora.conll_parse import is_punctuation
from grammar.dcp import DCP_evaluator, DCP_term, DCP_position, DCP_string
from grammar.lcfrs_derivation import LCFRSDerivation


class The_DCP_evaluator(DCP_evaluator):
    # der: Derivation
    def __init__(self, der):
        """
        :param der:
        :type der: LCFRSDerivation
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
        position = sorted(self.__der.terminal_positions(id))[i]
        return DCP_position(position, index.edge_label())

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
# dcp: list of DCP_term/DCP_position
# poss: list of string
# words: list of string
def dcp_to_hybridtree(tree, dcp, tokens, ignore_punctuation, construct_token, reorder=True, punct_positions=None):
    # if len(dcp) != 1:
    # raise Exception('DCP has multiple roots')
    j = 0
    for (i, token) in enumerate(tokens):
        # TODO: better punctuation detection
        if (ignore_punctuation and is_punctuation(token.form())) \
                or (punct_positions is not None and (i + 1) in punct_positions):
            tree.add_node(str(i) + 'p', token, True, False)
        elif ignore_punctuation or (punct_positions is not None):
            tree.add_node(str(j), token, True, True)
            j += 1
        else:
            tree.add_node(str(i), token, True, True)
    for root_term in dcp:
        (id, _) = dcp_to_hybridtree_recur(root_term, tree, len(tokens), construct_token)
        tree.add_to_root(id)
    if reorder:
        tree.reorder()
    return tree


# As above, recur, with identifiers starting at next_id.
# Return id of root node and next id.
# dcp: list of DCP_term/DCP_pos
# tree: GeneralHybridTree
# next_id: string
# return: pair of string
def dcp_to_hybridtree_recur(dcp, tree, next_id, construct_token):
    head = dcp.head()
    if isinstance(head, DCP_position):
        # FIXME : inconsistent counting of positions in hybrid tree requires -1
        id = str(head.position() - 1)
    elif isinstance(head, DCP_string):
        label = head
        id = str(next_id)
        next_id += 1
        tree.add_node(id, construct_token(label, None, False))
        # tree.set_label(id, label)
    else:
        raise Exception
    if head.edge_label() is not None:
        tree.node_token(id).set_edge_label(head.edge_label())
    for child in dcp.arg():
        (tree_child, next_id) = \
            dcp_to_hybridtree_recur(child, tree, next_id, construct_token)
        tree.add_child(id, tree_child)
    return id, next_id
