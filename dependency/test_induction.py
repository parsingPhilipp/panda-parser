__author__ = 'kilian'

import unittest
from hybridtree.general_hybrid_tree import GeneralHybridTree
from hybridtree.biranked_tokens import CoNLLToken, construct_dependency_token
from induction import induce_grammar, term_pos, direct_extraction, fanout_1, left_branching
from parser.naive.parsing import LCFRS_parser
from dependency.labeling import ChildPOSdepLabeling, StrictPOSdepLabeling
from grammar.sDCP.dcp import DCP_string
from hybridtree.test_multiroot import multi_dep_tree


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    def test_single_root_induction(self):
        tree = hybrid_tree_1()
        # print tree.children("v")
        # print tree
        #
        # for id_set in ['v v1 v2 v21'.split(' '), 'v1 v2'.split(' '),
        # 'v v21'.split(' '), ['v'], ['v1'], ['v2'], ['v21']]:
        # print id_set, 'top:', top(tree, id_set), 'bottom:', bottom(tree, id_set)
        # print id_set, 'top_max:', max(tree, top(tree, id_set)), 'bottom_max:', max(tree, bottom(tree, id_set))
        #
        # print "some rule"
        # for mem, arg in [(-1, 0), (0,0), (1,0)]:
        #     print create_DCP_rule(mem, arg, top_max(tree, ['v','v1','v2','v21']), bottom_max(tree, ['v','v1','v2','v21']),
        #                           [(top_max(tree, l), bottom_max(tree, l)) for l in [['v1', 'v2'], ['v', 'v21']]])
        #
        #
        # print "some other rule"
        # for mem, arg in [(-1,1),(1,0)]:
        #     print create_DCP_rule(mem, arg, top_max(tree, ['v1','v2']), bottom_max(tree, ['v1','v2']),
        #                           [(top_max(tree, l), bottom_max(tree, l)) for l in [['v1'], ['v2']]])
        #
        # print 'strict:' , strict_labeling(tree, top_max(tree, ['v','v21']), bottom_max(tree, ['v','v21']))
        # print 'child:' , child_labeling(tree, top_max(tree, ['v','v21']), bottom_max(tree, ['v','v21']))
        # print '---'
        # print 'strict: ', strict_labeling(tree, top_max(tree, ['v1','v21']), bottom_max(tree, ['v1','v21']))
        # print 'child: ', child_labeling(tree, top_max(tree, ['v1','v21']), bottom_max(tree, ['v1','v21']))
        # print '---'
        # print 'strict:' , strict_labeling(tree, top_max(tree, ['v','v1', 'v21']), bottom_max(tree, ['v','v1', 'v21']))
        # print 'child:' , child_labeling(tree, top_max(tree, ['v','v1', 'v21']), bottom_max(tree, ['v','v1', 'v21']))

        tree2 = hybrid_tree_2()

        # print tree2.children("v")
        # print tree2
        #
        # print 'siblings v211', tree2.siblings('v211')
        # print top(tree2, ['v','v1', 'v211'])
        # print top_max(tree2, ['v','v1', 'v211'])
        #
        # print '---'
        # print 'strict:' , strict_labeling(tree2, top_max(tree2, ['v','v1', 'v211']), bottom_max(tree2, ['v','v11', 'v211']))
        # print 'child:' , child_labeling(tree2, top_max(tree2, ['v','v1', 'v211']), bottom_max(tree2, ['v','v11', 'v211']))

        # rec_par = ('v v1 v2 v21'.split(' '),
        #            [('v1 v2'.split(' '), [(['v1'],[]), (['v2'],[])])
        #                ,('v v21'.split(' '), [(['v'],[]), (['v21'],[])])
        #            ])
        #
        # grammar = LCFRS(nonterminal_str(tree, top_max(tree, rec_par[0]), bottom_max(tree, rec_par[0]), 'strict'))
        #
        # add_rules_to_grammar_rec(tree, rec_par, grammar, 'child')
        #
        # grammar.make_proper()
        # print grammar

        print tree.recursive_partitioning()

        (_, grammar) = induce_grammar([tree, tree2], ChildPOSdepLabeling(), term_pos, direct_extraction, 'START')
        print max([grammar.fanout(nont) for nont in grammar.nonts()])
        print grammar

        parser = LCFRS_parser(grammar, 'NP N V V'.split(' '))
        print parser.best_derivation_tree()

        hybrid_tree = GeneralHybridTree()
        hybrid_tree = parser.dcp_hybrid_tree_best_derivation(hybrid_tree, 'P M h l'.split(' '),
                                                             'Piet Marie helpen lezen'.split(' '), True,
                                                             construct_dependency_token)
        print map(str, hybrid_tree.full_token_yield())
        print hybrid_tree

        string = "foo"
        dcp_string = DCP_string(string)
        dcp_string.set_dep_label("bar")
        print dcp_string, dcp_string.dep_label()

    def test_multiroot(self):
        tree = multi_dep_tree()
        for labeling_strategy in [StrictPOSdepLabeling(), ChildPOSdepLabeling()]:
            for recursive_partitioning in [direct_extraction, fanout_1, left_branching]:
                (_, grammar) = induce_grammar([tree], labeling_strategy, term_pos, recursive_partitioning, 'START')
                print grammar

                parser = LCFRS_parser(grammar, 'pA pB pC pD pE'.split(' '))
                print parser.best_derivation_tree()

                hybrid_tree = GeneralHybridTree()
                hybrid_tree = parser.dcp_hybrid_tree_best_derivation(hybrid_tree, 'pA pB pC pD pE'.split(' '),
                                                                     'A B C D E'.split(' '), True,
                                                                     construct_dependency_token)
                print hybrid_tree
                self.assertEqual(tree, hybrid_tree)


def hybrid_tree_1():
    tree = GeneralHybridTree()
    tree.add_node("v1", CoNLLToken('Piet', '_', "NP", 'SBJ'), True)
    tree.add_node("v21", CoNLLToken('Marie', '_', "N", 'OBJ'), True)
    tree.add_node("v", CoNLLToken('helpen', '_', "V", 'ROOT'), True)
    tree.add_node("v2", CoNLLToken('lezen', '_', "V", 'VBI'), True)
    tree.add_child("v", "v2")
    tree.add_child("v", "v1")
    tree.add_child("v2", "v21")
    tree.add_to_root("v")
    tree.reorder()
    return tree


def hybrid_tree_2():
    tree2 = GeneralHybridTree()
    tree2.add_node("v1", CoNLLToken('Piet', '_', "NP", 'SBJ'), True)
    tree2.add_node("v211", CoNLLToken('Marie', '_', 'N', 'OBJ'), True)
    tree2.add_node("v", CoNLLToken('helpen', '_', "V", 'ROOT'), True)
    tree2.add_node("v2", CoNLLToken('leren', '_', "V", 'VBI'), True)
    tree2.add_node("v21", CoNLLToken('lezen', '_', "V", 'VFIN'), True)
    tree2.add_child("v", "v2")
    tree2.add_child("v", "v1")
    tree2.add_child("v2", "v21")
    tree2.add_child("v21", "v211")
    tree2.add_to_root("v")
    tree2.reorder()
    return tree2


if __name__ == '__main__':
    unittest.main()
