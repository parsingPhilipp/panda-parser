__author__ = 'kilian'

from abc import ABCMeta, abstractmethod
from hybridtree.general_hybrid_tree import GeneralHybridTree


class AbstractLabeling:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

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
    def _bottom_node_name(self, tree, id):
        """
        :type tree: GeneralHybridTree
        :type id: str
        :rtype: str
        """
        pass

    @abstractmethod
    def _top_node_name(self, tree, id, terminal_generating):
        """
        :type tree: GeneralHybridTree
        :type id: str
        """
        pass

    @abstractmethod
    def __str__(self):
        pass


class StrictLabeling(AbstractLabeling):
    @abstractmethod
    def __str__(self):
        pass

    def __init__(self):
        super(StrictLabeling, self).__init__()

    def _label_bottom_seq(self, tree, id_seq):
        return '#'.join(map(lambda id: self._bottom_node_name(tree, id), id_seq))

    @abstractmethod
    def _top_node_name(self, tree, id, terminal_generating):
        pass

    @abstractmethod
    def _bottom_node_name(self, tree, id):
        pass

    def _label_top_seq(self, tree, id_seq, terminal_generating):
        return '#'.join(map(lambda id: self._top_node_name(tree, id, terminal_generating), id_seq))


class ChildLabeling(AbstractLabeling):
    @abstractmethod
    def __str__(self):
        pass

    def __init__(self):
        super(ChildLabeling, self).__init__()

    def _label_bottom_seq(self, tree, id_seq):
        if len(id_seq) == 1:
            return self._bottom_node_name(tree, id_seq[0])
        elif len(id_seq) > 1:
            # assuming that id_seq are siblings in tree, and thus also not at root level
            return 'children-of(' + self._bottom_node_name(tree, tree.parent(id_seq[0])) + ')'
        else:
            raise Exception('Empty components in top_max/ bottom_max!')

    @abstractmethod
    def _top_node_name(self, tree, id, terminal_generating):
        pass

    @abstractmethod
    def _bottom_node_name(self, tree, id):
        pass

    def _label_top_seq(self, tree, id_seq, terminal_generating):
        if len(id_seq) == 1:
            return self._top_node_name(tree, id_seq[0], terminal_generating)
        elif len(id_seq) > 1:
            # assuming that id_seq are siblings in tree, and thus also not at root level
            return 'children-of(' + self._top_node_name(tree, tree.parent(id_seq[0]), False) + ')'
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
                if not j in descendants.keys():
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


# Nonterminal labeling strategies
class StrictPOSLabeling(StrictLabeling):
    def __init__(self):
        super(StrictPOSLabeling, self).__init__()

    def _top_node_name(self, tree, id, terminal_generating):
        return tree.node_pos(id)

    def _bottom_node_name(self, tree, id):
        return tree.node_pos(id)

    def __str__(self):
        return 'strict_pos'


class StrictPOSdepAtLeafLabeling(StrictLabeling):
    def __init__(self):
        super(StrictPOSdepAtLeafLabeling, self).__init__()

    def _top_node_name(self, tree, id, terminal_generating):
        if not tree.descendants(id):
            return tree.node_pos(id) + ':' + tree.node_dep_label(id)
        else:
            return tree.node_pos(id)

    def _bottom_node_name(self, tree, id):
        return tree.node_pos(id)

    def __str__(self):
        return 'strict_pos_dep'


class StrictPOSdepLabeling(StrictLabeling):
    def __init__(self):
        super(StrictPOSdepLabeling, self).__init__()

    def _top_node_name(self, tree, id, terminal_generating):
        label = tree.node_pos(id) + ':' + tree.node_dep_label(id)
        if terminal_generating:
            return label + ':T'
        else:
            return label

    def _bottom_node_name(self, tree, id):
        return tree.node_pos(id) + ':' + tree.node_dep_label(id)

    def __str__(self):
        return 'strict_pos_dep_overall'


class StrictFormLabeling(StrictLabeling):
    def __init__(self):
        super(StrictFormLabeling, self).__init__()

    def _top_node_name(self, tree, id, terminal_generating):
        return tree.node_label(id)

    def _bottom_node_name(self, tree, id):
        return tree.node_label(id)

    def __str__(self):
        return 'strict_word'


class ChildPOSLabeling(ChildLabeling):
    def __init__(self):
        super(ChildPOSLabeling, self).__init__()

    def _top_node_name(self, tree, id, terminal_generating):
        return tree.node_pos(id)

    def _bottom_node_name(self, tree, id):
        return tree.node_pos(id)

    def __str__(self):
        return 'child_pos'


class ChildPOSdepAtLeafLabeling(ChildLabeling):
    def __init__(self):
        super(ChildPOSdepAtLeafLabeling, self).__init__()

    def _top_node_name(self, tree, id, terminal_generating):
        if terminal_generating:
            return tree.node_pos(id) + ':' + tree.node_dep_label(id)
        else:
            return tree.node_pos(id)

    def _bottom_node_name(self, tree, id):
        return tree.node_pos(id)

    def __str__(self):
        return 'child_pos_dep'


class ChildPOSdepLabeling(ChildLabeling):
    def __init__(self):
        super(ChildPOSdepLabeling, self).__init__()

    def _top_node_name(self, tree, id, terminal_generating):
        label = tree.node_pos(id) + ':' + tree.node_dep_label(id)
        if terminal_generating:
            return label + ':T'
        else:
            return label

    def _bottom_node_name(self, tree, id):
        return tree.node_pos(id) + ':' + tree.node_dep_label(id)

    def __str__(self):
        return 'child_pos_dep_overall'


class ChildFormLabeling(ChildLabeling):
    def __init__(self):
        super(ChildFormLabeling, self).__init__()

    def _top_node_name(self, tree, id, terminal_generating):
        return tree.node_label(id)

    def _bottom_node_name(self, tree, id):
        return tree.node_label(id)

    def __str__(self):
        return 'child_word'