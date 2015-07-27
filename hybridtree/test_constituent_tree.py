__author__ = 'kilian'

import unittest
from hybridtree.constituent_tree import ConstituentTree


class ConstituentTreeTest(unittest.TestCase):
    def test_something(self):
        tree = self.tree
        print "rooted", tree.root
        tree.add_to_root("VP")
        print "rooted", tree.root
        tree.set_label("V", "V")
        tree.set_label("VP", "VP")
        tree.set_label("ADV", "ADV")

        print "sent label", tree.sent_label()

        print "leaves", tree.leaves()

        print "is leaf (leaves)", [(x, tree.is_leaf(x)) for (x, _, _) in tree.leaves()]
        print "is leaf (internal)", [(x, tree.is_leaf(x)) for x in tree.ids()]
        print "leaf index", [(x, tree.leaf_index(x)) for x in ["f1", "f2", "f3"]]

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

    def setUp(self):
        tree = ConstituentTree("s1")
        tree.add_leaf("f1", "VP", "hat")
        tree.add_leaf("f2", "ADV", "schnell")
        tree.add_leaf("f3", "VP", "gearbeitet")
        tree.add_punct("f4", "PUNC", ".")

        tree.add_child("V", "f1")
        tree.add_child("V", "f3")
        tree.add_child("ADV", "f2")

        tree.add_child("VP", "V")
        tree.add_child("VP", "ADV")
        self.tree = tree


if __name__ == '__main__':
    unittest.main()
