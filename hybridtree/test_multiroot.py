__author__ = 'kilian'

import unittest
from hybridtree.general_hybrid_tree import GeneralHybridTree
from hybridtree.monadic_tokens import *


class MultiRootedHybridTreeTest(unittest.TestCase):
    def test_recursive_partitioning(self):
        tree = multi_const_tree()

        print tree
        tree.reorder()
        print tree

        self.assertEqual(tree.recursive_partitioning(), ({0, 1, 2, 3},
                                                         [({0, 2}, [({0}, []), ({2}, [])]),
                                                          ({1, 3}, [({1}, []), ({3}, [])])]))

        tree2 = multi_dep_tree()
        print tree2
        print tree2.recursive_partitioning()
        self.assertEqual(tree2.recursive_partitioning(), ({0, 1, 2, 3, 4},
                                                          [({0, 2}, [({0}, []), ({2}, [])]), (
                                                              {1, 3, 4},
                                                              [({3}, []), ({1}, []), ({4}, [])])]))


if __name__ == '__main__':
    unittest.main()


def multi_const_tree():
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
    return tree


def multi_const_tree_ordered():
    tree = multi_const_tree()
    tree.reorder()
    return tree


def multi_dep_tree():
    tree = GeneralHybridTree('multi')
    tree.add_node('1', CoNLLToken('A', '_', 'pA', '_', '_', 'dA'), True)
    tree.add_node('211', CoNLLToken('B', '_', 'pB', '_', '_', 'dB'), True)
    tree.add_node('11', CoNLLToken('C', '_', 'pC', '_', '_', 'dC'), True)
    tree.add_node('2', CoNLLToken('D', '_', 'pD', '_', '_', 'dD'), True)
    tree.add_node('21', CoNLLToken('E', '_', 'pE', '_', '_', 'dE'), True)
    tree.add_to_root('2')
    tree.add_to_root('1')
    for c in ['21', '211']:
        tree.add_child('2', c)
    tree.add_child('1', '11')
    tree.reorder()
    return tree
