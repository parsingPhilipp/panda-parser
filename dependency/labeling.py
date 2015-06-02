__author__ = 'kilian'

from abc import ABCMeta, abstractmethod
from hybridtree.general_hybrid_tree import GeneralHybridTree
from types import FunctionType, MethodType


class AbstractLabeling:
    __metaclass__ = ABCMeta

    def __init__(self, name):
        """
        :type name: str
        """
        self.__name = name

    def label_nonterminal(self, tree, node_ids, t_max, b_max, fanout):
        """
        :type tree: GeneralHybridTree
        :param node_ids: Node Ids
        :param t_max:  top_max
        :param b_max:  bottom_max
        :type fanout: int
        :return: str
        """
        arg_dep = argument_dependencies(tree, b_max + t_max)
        terminal_generating = len(node_ids) == 1
        if terminal_generating:
            assert len(t_max) == 1
        label = ','.join(
            [self._label_bottom_seq(tree, id_seq) for id_seq in b_max] + [
                self._label_top_seq(tree, id_seq, terminal_generating) for id_seq
                in t_max])
        return ('{' + str(fanout) + ':' + label + ',' + arg_dep + '}').replace(' ', '').replace('*', '')

    @abstractmethod
    def _label_bottom_seq(self, tree, id_seq):
        """
        :type tree: GeneralHybridTree
        :param id_seq:
        :rtype: str
        """
        pass

    @abstractmethod
    def _label_top_seq(self, tree, id_seq, terminal_generating):
        """
        :type tree: GeneralHybridTree
        :param id_seq:
        :type terminal_generating: bool
        :rtype: str
        """
        pass

    @abstractmethod
    def _bottom_node_name(self, token):
        """
        :type tree: GeneralHybridTree
        :type id: str
        :rtype: str
        """
        pass

    @abstractmethod
    def _top_node_name(self, token, terminal_generating):
        """
        :type tree: GeneralHybridTree
        :type id: str
        """
        pass

    def __str__(self):
        return self.__name


class StrictLabeling(AbstractLabeling):
    def _label_bottom_seq(self, tree, id_seq):
        return '#'.join(map(lambda id: self._bottom_node_name(tree.node_token(id)), id_seq))

    # @abstractmethod
    def _top_node_name(self, token, terminal_generating):
        pass

    # @abstractmethod
    def _bottom_node_name(self, token):
        pass

    def _label_top_seq(self, tree, id_seq, terminal_generating):
        return '#'.join(map(lambda id: self._top_node_name(tree.node_token(id), terminal_generating), id_seq))


class ChildLabeling(AbstractLabeling):
    def _label_bottom_seq(self, tree, id_seq):
        if len(id_seq) == 1:
            return self._bottom_node_name(tree.node_token(id_seq[0]))
        elif len(id_seq) > 1:
            # assuming that id_seq are siblings in tree, and thus also not at root level
            return 'children-of(' + self._bottom_node_name(tree.node_token(tree.parent(id_seq[0]))) + ')'
        else:
            raise Exception('Empty components in top_max/ bottom_max!')

    # @abstractmethod
    def _top_node_name(self, token, terminal_generating):
        pass

    # @abstractmethod
    def _bottom_node_name(self, token):
        pass

    def _label_top_seq(self, tree, id_seq, terminal_generating):
        if len(id_seq) == 1:
            return self._top_node_name(tree.node_token(id_seq[0]), terminal_generating)
        elif len(id_seq) > 1:
            if id_seq[0] in tree.root:
                # only in case of multi-rooted hybrid trees
                return 'children-of(VIRTUAL-ROOT)'
            else:
                # assuming that id_seq are siblings in tree, and not at root level
                return 'children-of(' + self._top_node_name(tree.node_token(tree.parent(id_seq[0])), False) + ')'
        else:
            raise Exception('Empty components in top_max/ bottom_max!')


def argument_dependencies(tree, id_seqs):
    """
    Compute a string that represents, how the arguments of some dcp-nonterminal
    depend on one another.
    :rtype: str
    :param tree: GeneralHybridTree
    :param id_seqs: list of list of string (Concatenation of top_max and bottom_max)
    :return: string
        (of the form "1.4(0).2(3(5))": 1, 4 and 2 are independent, 4 depends on 0, etc.)
    """
    ancestor = {}
    descendants = {}

    # Build a table with the dependency relation of arguments.
    # The table holds the indices of a node in name_seqs.
    for i in range(len(id_seqs)):
        name_seq = id_seqs[i]
        for j in range(len(id_seqs)):
            name_seq2 = id_seqs[j]
            if name_seq[0] in [descendant for id in name_seq2 for descendant in tree.descendants(id)]:
                ancestor[i] = j
                if j not in descendants.keys():
                    descendants[j] = [i]
                else:
                    descendants[j].append(i)

    # compute the set of nodes that have no ancestors
    topmost = [i for i in range(len(id_seqs)) if i not in ancestor.keys()]

    # recursively compute the dependency string
    return argument_dependencies_rec(tree, id_seqs, descendants, topmost)


def argument_dependencies_rec(tree, id_seqs, descendants, arg_indices):
    """
    Recursively compute the string for the argument dependencies.
    :type tree: GeneralHybridTree
    :param id_seqs: list of list of string (concatenation of top_max and bottom_max)
    :param descendants: map from (indices of id_seqs) to (list of (indices of id_seqs))
    :param arg_indices: list of (indices of id_seqs)
    :rtype: str
    """

    # skip nodes that are descendants of some other node in arg_indices
    skip = [i for j in arg_indices for i in arg_indices if j in descendants.keys() and i in descendants[j]]
    arg_indices = [i for i in arg_indices if i not in skip]

    # sort indices according to position in yield
    arg_indices = sorted(arg_indices,
                         cmp=lambda i, j: cmp(tree.node_index(id_seqs[i][0]),
                                              tree.node_index(id_seqs[j][0])))
    term = []
    for i in arg_indices:
        t = str(i)
        if i in descendants.keys():
            t += '(' + argument_dependencies_rec(tree, id_seqs, descendants, descendants[i]) + ')'
        term.append(t)

    return '.'.join(term)


def token_to_pos(token, terminal_generating=False):
    if terminal_generating:
        return token.pos() + ':T'
    else:
        return token.pos()


def token_to_fine_grained_pos(token, terminal_generating=False):
    if terminal_generating:
        return token.fine_grained_pos() + ':T'
    else:
        return token.fine_grained_pos()


def token_to_pos_and_deprel(token, terminal_generating=False):
    if terminal_generating:
        return token.pos() + ':' + token.deprel() + ':T'
    else:
        return token.pos() + ':' + token.deprel()


def token_to_fine_grained_pos_and_deprel(token, terminal_generating=False):
    if terminal_generating:
        return token.fine_grained_pos() + ':' + token.deprel() + ':T'
    else:
        return token.fine_grained_pos() + ':' + token.deprel()


def token_to_form(token, terminal_generating=False):
    if terminal_generating:
        return token.form() + ':T'
    else:
        return token.form()


def token_to_deprel(token, terminal_generating=False):
    if terminal_generating:
        return token.deprel() + ':T'
    else:
        return token.deprel()


class LabelingStrategyFactory:
    def __init__(self):
        self.__top_level_strategies = {}
        self.__node_to_string_strategies = {}

    def register_top_level_strategy(self, name, strategy):
        self.__top_level_strategies[name] = strategy

    def register_node_to_string_strategy(self, name, strategy):
        self.__node_to_string_strategies[name] = strategy

    def create_simple_labeling_strategy(self, top_level, node_to_string):
        name = ('-'.join([top_level, node_to_string]))
        if not self.__top_level_strategies.has_key(top_level):
            s = 'Unknown top-level strategy ' + top_level + '\n'
            s += 'I know the following top-level strategies: \n'
            for name in self.__top_level_strategies.keys():
                s += '\t' + name + '\n'
            raise Exception(s)
        labeling_strategy = self.__top_level_strategies[top_level](name)
        assert (isinstance(labeling_strategy, AbstractLabeling))

        node_strategy = self.__node_to_string_strategies[node_to_string]
        if not isinstance(node_strategy, FunctionType):
            s = 'Unknown top-level strategy ' + node_strategy + '\n'
            s += 'I know the following node-level strategies: \n'
            for name in self.__node_to_string_strategies.keys():
                s += '\t' + name + '\n'
            raise Exception(s)

        labeling_strategy._top_node_name = node_strategy
        labeling_strategy._bottom_node_name = node_strategy
        return labeling_strategy

    def create_complex_labeling_strategy(self, args):
        raise Exception('Not implemented!')


def the_labeling_factory():
    factory = LabelingStrategyFactory()
    factory.register_top_level_strategy('strict', StrictLabeling)
    factory.register_top_level_strategy('child', ChildLabeling)

    factory.register_node_to_string_strategy('pos', token_to_pos)
    factory.register_node_to_string_strategy('fine_grained_pos', token_to_fine_grained_pos)
    factory.register_node_to_string_strategy('deprel', token_to_deprel)
    factory.register_node_to_string_strategy('pos+deprel', token_to_pos_and_deprel)
    factory.register_node_to_string_strategy('fine_grained_pos+deprel', token_to_fine_grained_pos_and_deprel)
    return factory
