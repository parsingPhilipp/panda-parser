# Hybrid Tree
# A directed acyclic graph, where a (not necessarily strict) subset of the nodes is linearly ordered.

from collections import defaultdict
from decomposition import *

class GeneralHybridTree:

    # Constructor
    # sent_label: string
    def __init__(self, sent_label = None):
        # label of sentence (identifier in corpus)
        self.__sent_label = sent_label
        # id of root node
        self.__root = None
        # list of node ids
        self.__nodes = []
        # maps node id to list of ids of children
        self.__id_to_child_ids = {}
        # maps node id to node label
        self.__id_to_label = {}
        # maps node id to part-of-speech tag
        self.__id_to_pos = {}
        # list of node ids in ascending order
        self.__ordered_ids = []
        # list of node ids in ascending order, including disconnected nodes
        self.__full_yield = []
        # maps node id to position in the ordering
        self.__id_to_node_index = {}
        # maps node_index (position in ordering) to node id
        self.__node_index_to_id = {}
        # number of nodes in ordering
        self.__n_ordered_nodes = 0

     # Get label of sentence.
    # return: string
    def sent_label(self):
        return self.__sent_label

    # Set root to node given by id.
    # id: string
    def set_root(self, id):
        self.__root = id

    # Has root been determined?
    # return: bool
    def rooted(self):
        return self.__root is not None

    # Get id of root.
    # return: string
    def root(self):
        return self.__root

    # Add next node. Order is significant for ordering
    # id: string
    # pos: string (part of speech)
    # label: string (word, syntactic category)
    # order: bool (include node in linear ordering)
    # connected: bool (node is connected to other tree nodes)
    # Set order = True and connected = False to include some token (e.g. punctuation)
    # that appears in the yield but shall be ignored during tree operations.
    def add_node(self, id, label, pos = None, order = False, connected = True):
        self.__nodes += [id]
        self.__id_to_label[id] = label
        if (pos is not None):
            self.__id_to_pos[id] = pos
        if (order is True):
            if (connected is True):
                self.__ordered_ids += [id]
                self.__id_to_node_index[id] = self.__n_ordered_nodes
                self.__n_ordered_nodes += 1
                self.__node_index_to_id[self.__n_ordered_nodes] = id
            self.__full_yield += [id]

    # Add a pair of node ids in the parent-child relation.
    # parent: string
    # child: string
    def add_child(self, parent, child):
        if not parent in self.__id_to_child_ids:
            self.__id_to_child_ids[parent] = []
        self.__id_to_child_ids[parent] += [child]

    # Get id of parent node, or None.
    # id: string
    # return: string
    def parent(self, id):
        return self.__parent_recur(id, self.root())
    # child: the node, whose parent is searched
    # id: potential parent
    def __parent_recur(self, child, id):
        if child in self.children(id):
            return id
        else:
            for next_id in self.children(id):
                parent = self.__parent_recur(child, next_id)
                if parent is not None:
                    return parent
        return None

    # Is there node that is child of two nodes?
    # return: bool
    def reentrant(self):
        parent = defaultdict(list)
        for id in self.__id_to_child_ids:
            for child in self.children(id):
                parent[child] += [id]
        for id in parent:
            if len(parent[id]) > 1:
                return True
        return False

    # Get the list of node ids of child nodes, or the empty list.
    # id: string
    # return: list of string
    def children(self, id):
        if id in self.__id_to_child_ids:
            return self.__id_to_child_ids[id]
        else:
            return []

    # Is the node in the ordering?
    # id: string
    # return: bool
    def in_ordering(self, id):
        return id in self.__ordered_ids

    # Get node id at index in ordering
    # index: int
    # return: string (id)
    def index_node(self, index):
        return self.__node_index_to_id[index]

    # Get index of node in ordering
    # id: string
    # return: int
    def node_index(self, id):
        return self.__id_to_node_index[id]

    # Reorder children according to smallest node in subtree
    def reorder(self):
        self.__reorder(self.root())
    # id: string
    # return: int (index or -1)
    def __reorder(self, id):
        min_indices = {}
        if self.children(id).__len__() > 0:
            for child in self.children(id):
                min_indices[child] = self.__reorder(child)
            self.__id_to_child_ids[id] = sorted(self.children(id), key = lambda i : min_indices[i])
        if self.in_ordering(id):
            min_indices[id] = self.__id_to_node_index[id]
        min_index = -1
        for index in min_indices.values():
            if min_index < 0 or index < min_index:
                min_index = index
        return min_index

    # indices (w.r.t. ordering) of all nodes under some node
    # cf. \Pi^{-1} in paper
    # id: string
    # return: list of int
    def fringe(self, id):
        y = []
        if self.in_ordering(id):
            y = [self.__id_to_node_index[id]]
        for child in self.children(id):
            y += self.fringe(child)
        return y

    # Number of contiguous spans of node.
    # id: string
    # return: int
    def n_spans(self, id):
        return len(join_spans(self.fringe(id)))

    # Maximum number of spans of any node.
    # return: int
    def max_n_spans(self):
        nums = [self.n_spans(id) for id in self.nodes()]
        if len(nums) > 0:
            return max(nums)
        else:
            return 1

    # Total number of gaps in any node.
    # return: int
    def n_gaps(self):
        return self.__n_gaps_below(self.root())
    # id: string
    # return: int
    def __n_gaps_below(self, id):
        n_gaps = self.n_spans(id) - 1
        for child in self.children(id):
            n_gaps += self.__n_gaps_below(child)
        return n_gaps

    # Create unlabelled structure, only in terms of breakup of yield
    # return: pair consisting of (root and list of child nodes)
    def unlabelled_structure(self):
        return self.unlabelled_structure_recur(self.root())
    def unlabelled_structure_recur(self, id):
        head = set(self.fringe(id))
        tail = [self.unlabelled_structure_recur(child) for \
                child in self.children(id)]
        # remove useless step
        if len(tail) == 1 and head == tail[0][0]:
            return tail[0]
        else:
            return (head, tail)

    # Labelled spans.
    # return: list of spans (each of which is string plus an even
    # number of (integer) positions)
    def labelled_spans(self):
        spans = []
        for id in [n for n in self.nodes() if not n in self.full_yield()]:
            span = [self.node_label(id)]
            for (low, high) in join_spans(self.fringe(id)):
                span += [low, high]
            # TODO: this if-clause allows to handle trees, that have nodes with empty fringe
            if len(span) >= 3:
                spans += [span]
        return sorted(spans, \
                      cmp=lambda x, y: cmp([x[1]] + [-x[2]] + x[3:] + [x[0]], \
                                           [y[1]] + [-y[2]] + y[3:] + [y[0]]))

    # Get yield as list of all labels of nodes, that are in the ordering
    # return: list of string
    def labelled_yield(self):
        return [self.node_label(id) for id in self.__ordered_ids]

    #Get full yield (including disconnected nodes) as list of labels
    def full_labelled_yield(self):
        return [self.node_label(id) for id in self.__full_yield]

    #Get full yield (including disconnected nodes) as list of ids
    def full_yield(self):
        return self.__full_yield

    # Get ids of all nodes.
    # return: list of string
    def nodes(self):
        return self.__nodes

    # Get the label of some node.
    # id: string
    # return: string
    def node_label(self, id):
        return self.__id_to_label[id]

    # Does yield cover whole string?
    # return: bool
    def complete(self):
        return self.rooted() and \
               len(self.fringe(self.root())) == self.__n_ordered_nodes

    # Get POS of node
    # id: string
    def node_pos(self, id):
        return self.__id_to_pos[id]

    # Get POS-yield (omitting disconnected nodes)
    def pos_yield(self):
        return [self.node_pos(id) for id in self.__ordered_ids]

    # Number of nodes in total tree (omitting disconnected nodes)
    def n_nodes(self):
        return self.__n_nodes_below(self.root()) + 1
    # Number of nodes below node
    # id: string
    # return: int
    def __n_nodes_below(self, id):
        n = len(self.children(id))
        for child in self.children(id):
            n += self.__n_nodes_below(child)
        return n

    # Is there any non-ordered node without children?
    # Includes the case the root has no children.
    # return: bool
    def empty_fringe(self):
        for id in self.nodes():
            if len(self.children(id)) == 0 and not id in self.full_yield():
                return True
        return self.rooted() and len(self.fringe(self.root())) == 0


#
def test():
    print "Start"
    tree = GeneralHybridTree()
    tree.add_node("v1","Piet","NP",True)
    tree.add_node("v21","Marie","N",True)
    tree.add_node("v","helpen","V",True)
    tree.add_node("v2","lezen", "V", True)
    tree.add_child("v","v2")
    tree.add_child("v","v1")
    tree.add_child("v2","v21")
    tree.add_node("v3",".","Punc",True,False)
    tree.set_root("v")
    print tree.children("v")
    tree.reorder()
    print tree.children("v")

    print "fringe v", tree.fringe("v")
    print "fringe v2", tree.fringe("v2")

    print "n spans v", tree.n_spans("v")
    print "n spans v2", tree.n_spans("v2")

    print "n_gaps", tree.n_gaps()
    print "ids:", tree.nodes()
    print "complete", tree.complete()
    print "unlabeled structure", tree.unlabelled_structure()

    print "max n spans", tree.max_n_spans()

    print "labelled yield", tree.labelled_yield()
    print "full labelled yield", tree.full_labelled_yield()

    print "full yield", tree.full_yield()

    print "labelled spans", tree.labelled_spans()
    print "POS yield", tree.pos_yield()
# test()