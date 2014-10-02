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
        # maps node id to position in the ordering
        self.__id_to_node_index = {}
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
    def add_node(self, id, label, pos = None, order = False):
        self.__nodes += [id]
        self.__id_to_label[id] = label
        if (pos is not None):
            self.__id_to_pos[id] = pos
        if (order is True):
            self.__ordered_ids += [id]
            self.__n_ordered_nodes += 1
            self.__id_to_node_index[id] = self.__n_ordered_nodes

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

    # Get yield as list of all labels of nodes, that are in the ordering
    # return: list of string
    def label_yield(self):
	    return [self.node_label(id) for id in self.__ordered_ids]

    # Get ids of all nodes.
    # return: list of string
    def nodes(self):
        return self.__nodes

    # Get the label of some node.
    # id: string
    # return: string
    def node_label(self, id):
        return self.__id_to_label[id]
#
def test():
    print "Start"
    tree = GeneralHybridTree()
    tree.add_node("v1","Piet",None,True)
    tree.add_node("v21","Marie",None,True)
    tree.add_node("v","helpen",None,True)
    tree.add_node("v2","lezen", None, True)
    tree.add_child("v","v2")
    tree.add_child("v","v1")
    tree.add_child("v2","v21")
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

    print "label yield: ", tree.label_yield()

test()