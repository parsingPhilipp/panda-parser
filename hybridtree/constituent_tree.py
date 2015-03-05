__author__ = 'kilian'

from hybridtree.general_hybrid_tree import GeneralHybridTree

#
class HybridTree(GeneralHybridTree):
    def __init__(self, sent_label=None):
       GeneralHybridTree.__init__(self, sent_label)

    # Add next leaf. Order of adding is significant.
    # id: string
    # pos: string (part of speech)
    # word: string
    def add_leaf(self, id, pos, word):
        self.add_node(id, word, pos, True, True)

        # Add punctuation (has no parent).

    # id: string
    # pos: string (part of speech)
    # word: string
    def add_punct(self, id, pos, word):
        self.add_node(id, word, pos, True, False)

    # Add label of non-leaf. If it has no children, give it empty list of
    # children.
    # id: string
    # label: string
    def set_label(self, id, label):
        self.add_node(id, label, None, False, True)

    # All leaves of tree.
    # return: list of triples.
    def leaves(self):
        return [(id, self.node_pos(id), self.node_label(id)) for id in self.full_yield()]

    # Is leaf? (This is, the id occurs in the list of leaves.)
    # id: string
    # return: bool
    def is_leaf(self, id):
        return id in self.full_yield()

    # Get leaf for index.
    # index: int
    # return: triple
    def index_leaf(self, index):
        return self.index_node(index)

    # Get index for id of leaf.
    # id: string
    # return: int
    def leaf_index(self, id):
        return self.node_index(id)

    # Get part of speech of node.
    # id: string
    # return: string
    def leaf_pos(self, id):
        return self.node_pos(id)

    # Get word of node.
    # id: string
    # return: string
    def leaf_word(self, id):
        return self.node_label(id)


    # Get yield as list of words, omitting punctuation.
    # return: list of string
    def word_yield(self):
        return self.labelled_yield()

    # Get label of (non-leaf) node.
    # id: string
    # return: string
    def label(self, id):
        return self.node_label(id)

    # Get ids of all internal nodes.
    # return: list of string
    def ids(self):
        return [n for n in self.nodes() if n not in self.full_yield()]

    def n_nodes(self):
        return GeneralHybridTree.n_nodes(self) + 1

def test():
    tree = HybridTree("s1")
    tree.add_leaf("f1","VP","hat")
    tree.add_leaf("f2","ADV","schnell")
    tree.add_leaf("f3","VP","gearbeitet")
    tree.add_punct("f4","PUNC",".")

    tree.add_child("V","f1")
    tree.add_child("V","f3")
    tree.add_child("ADV","f2")

    tree.add_child("VP","V")
    tree.add_child("VP","ADV")

    print "rooted", tree.rooted()
    tree.set_root("VP")
    print "rooted", tree.rooted()
    tree.set_label("V","V")
    tree.set_label("VP","VP")
    tree.set_label("ADV","ADV")

    print "sent label", tree.sent_label()

    print "leaves", tree.leaves()

    print "is leaf (leaves)", [(x, tree.is_leaf(x)) for (x,_,_) in tree.leaves()]
    print "is leaf (internal)", [(x, tree.is_leaf(x)) for x in tree.ids()]
    print "leaf index",  [(x, tree.leaf_index(x)) for x in ["f1","f2","f3"]]

    print "pos yield", tree.pos_yield()
    print "word yield", tree.word_yield()

    # reentrant
    # parent

    print "ids", tree.ids()

    # reorder
    print "n nodes", tree.n_nodes()
    print "n gaps", tree.n_gaps()

    print "fringe VP", tree.fringe("VP")
    print "fringe V", tree.fringe("V")

    print "empty fringe", tree.empty_fringe()

    print "complete?", tree.complete()

    print "max n spans", tree.max_n_spans()

    print "unlabelled structure", tree.unlabelled_structure()

    print "labelled spans", tree.labelled_spans()


# test()