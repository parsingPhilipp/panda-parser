__author__ = 'kilian'

import unittest
from hybridtree.general_hybrid_tree import GeneralHybridTree
from hybridtree.biranked_tokens import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        tree = GeneralHybridTree("multi")
        tree.add_node('1.1', ConstituencyTerminal('A', 'pA'), True, True)
        tree.add_node('2.1', ConstituencyTerminal('B', 'pB'), True, True)
        tree.add_node('1.2', ConstituencyTerminal('C', 'pC'), True, True)
        tree.add_node('2.2', ConstituencyTerminal('D', 'pD'), True, True)
        tree.add_node('1', ConstituencyCategory('E'), False, True)
        tree.add_node('2', ConstituencyCategory('F'), False, True)
        for p in ['2', '1']:
            tree.add_to_root(p)
            for c in ['1', '2']:
                tree.add_child(p, p + '.' + c)

        print tree
        tree.reorder()
        print tree

        self.assertEqual(tree.recursive_partitioning(), (set([0, 1, 2, 3]),
                                                         [(set([0, 2]), [(set([0]), []), (set([2]), [])]),
                                                          (set([1, 3]), [(set([1]), []), (set([3]), [])])]))


if __name__ == '__main__':
    unittest.main()
