#-*- coding: iso-8859-15 -*-
from __future__ import print_function
__author__ = 'kilian'

import copy
import sys
import unittest
from collections import defaultdict
from math import e

from dependency.induction import induce_grammar
from grammar.induction.recursive_partitioning import left_branching, right_branching, cfg, \
    the_recursive_partitioning_factory, direct_extraction
from grammar.induction.terminal_labeling import the_terminal_labeling_factory
from dependency.labeling import the_labeling_factory
from grammar.linearization import linearize
from grammar.dcp import DCP_string
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import CoNLLToken, construct_conll_token
from parser.cpp_cfg_parser.parser_wrapper import CFGParser
from parser.derivation_interface import derivation_to_hybrid_tree
try:
    from parser.fst.fst_export import compile_wfst_from_right_branching_grammar, fsa_from_list_of_symbols, compose, shortestpath, shortestdistance, retrieve_rules, PolishDerivation, ReversePolishDerivation, compile_wfst_from_left_branching_grammar, local_rule_stats, paths, LeftBranchingFSTParser
    test_pynini = True
except ModuleNotFoundError:
    test_pynini = False

from parser.sDCPevaluation.evaluator import The_DCP_evaluator, dcp_to_hybridtree
from parser.viterbi.viterbi import ViterbiParser as LCFRS_parser
from tests.test_multiroot import multi_dep_tree


class InductionTest(unittest.TestCase):
    def test_recursive_partitioning_transformation(self):
        tree = HybridTree("mytree")
        ids = ['a', 'b', 'c', 'd']
        for f in ids:
            tree.add_node(f, CoNLLToken(f, '_', '_', '_', '_', '_'), True, True)
            if f != 'a':
                tree.add_child('a', f)
        tree.add_to_root('a')

        print(tree)
        self.assertEqual([token.form() for token in tree.token_yield()], ids)
        self.assertEqual(tree.recursive_partitioning(), (set([0, 1, 2, 3]), [(set([0]), []), (set([1]), []), (set([2]), []), (set([3]), [])]))
        print(tree.recursive_partitioning())

        [fanout_1] = the_recursive_partitioning_factory().getPartitioning('fanout-1')

        print(fanout_1(tree))



        # self.assertEqual(True, True)


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
        # print create_DCP_rule(mem, arg, top_max(tree, ['v','v1','v2','v21']), bottom_max(tree, ['v','v1','v2','v21']),
        # [(top_max(tree, l), bottom_max(tree, l)) for l in [['v1', 'v2'], ['v', 'v21']]])
        #
        #
        # print "some other rule"
        # for mem, arg in [(-1,1),(1,0)]:
        # print create_DCP_rule(mem, arg, top_max(tree, ['v1','v2']), bottom_max(tree, ['v1','v2']),
        # [(top_max(tree, l), bottom_max(tree, l)) for l in [['v1'], ['v2']]])
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
        # [('v1 v2'.split(' '), [(['v1'],[]), (['v2'],[])])
        #                ,('v v21'.split(' '), [(['v'],[]), (['v21'],[])])
        #            ])
        #
        # grammar = LCFRS(nonterminal_str(tree, top_max(tree, rec_par[0]), bottom_max(tree, rec_par[0]), 'strict'))
        #
        # add_rules_to_grammar_rec(tree, rec_par, grammar, 'child')
        #
        # grammar.make_proper()
        # print grammar

        print(tree.recursive_partitioning())

        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty','pos'),
                                      # the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel'),
                                      terminal_labeling.token_label, [direct_extraction], 'START')
        print(max([grammar.fanout(nont) for nont in grammar.nonts()]))
        print(grammar)

        parser = LCFRS_parser(grammar, 'NP N V V'.split(' '))
        print(parser.best_derivation_tree())

        tokens = [construct_conll_token(form, pos) for form, pos in
                  zip('Piet Marie helpen lezen'.split(' '), 'NP N V V'.split(' '))]
        hybrid_tree = HybridTree()
        hybrid_tree = parser.dcp_hybrid_tree_best_derivation(hybrid_tree, tokens, True,
                                                             construct_conll_token)
        print(list(map(str, hybrid_tree.full_token_yield())))
        print(hybrid_tree)

        string = "foo"
        dcp_string = DCP_string(string)
        dcp_string.set_edge_label("bar")
        print(dcp_string, dcp_string.edge_label())

        linearize(grammar, the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel'), the_terminal_labeling_factory().get_strategy('pos'), sys.stdout)

    def test_multiroot(self):
        tree = multi_dep_tree()
        term_pos = the_terminal_labeling_factory().get_strategy('pos').token_label
        fanout_1 = the_recursive_partitioning_factory().getPartitioning('fanout-1')
        for top_level_labeling_strategy in ['strict', 'child']:
            labeling_strategy = the_labeling_factory().create_simple_labeling_strategy(top_level_labeling_strategy,
                                                                                       'pos+deprel')
            for recursive_partitioning in [[direct_extraction], fanout_1, [left_branching]]:
                (_, grammar) = induce_grammar([tree], labeling_strategy, term_pos, recursive_partitioning, 'START')
                print(grammar)

                parser = LCFRS_parser(grammar, 'pA pB pC pD pE'.split(' '))
                print(parser.best_derivation_tree())

                cleaned_tokens = copy.deepcopy(tree.full_token_yield())
                for token in cleaned_tokens:
                    token.set_edge_label('_')
                hybrid_tree = HybridTree()
                hybrid_tree = parser.dcp_hybrid_tree_best_derivation(hybrid_tree, cleaned_tokens, True,
                                                                     construct_conll_token)
                print(hybrid_tree)
                self.assertEqual(tree, hybrid_tree)

    def test_fst_compilation_right(self):
        if not test_pynini:
            return
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [right_branching], 'START')

        a, rules = compile_wfst_from_right_branching_grammar(grammar)

        print(repr(a))

        symboltable = a.input_symbols()

        string = 'NP N V V V'.split(' ')

        token_sequence = [construct_conll_token(form, lemma) for form, lemma in
                          zip('Piet Marie helpen leren lezen'.split(' '), string)]


        fsa = fsa_from_list_of_symbols(string, symboltable)
        self.assertEqual('0\t1\tNP\tNP\n1\t2\tN\tN\n2\t3\tV\tV\n3\t4\tV\tV\n4\t5\tV\tV\n5\n', fsa.text().decode('utf-8'))

        b = compose(fsa, a)

        print(b.input_symbols())
        for i in b.input_symbols():
            print(i)


        print("Input Composition")
        print(b.text(symboltable, symboltable).decode('utf-8'))

        i = 0
        for path in paths(b):
            print(i, "th path:", path, end=' ')
            r = list(map(rules.index_object, path))
            d = PolishDerivation(r[1::])
            dcp = The_DCP_evaluator(d).getEvaluation()
            h = HybridTree()
            dcp_to_hybridtree(h, dcp, token_sequence, False, construct_conll_token)
            h.reorder()
            if h == tree2:
                print("correct")
            else:
                print("incorrect")
            i += 1

        stats = defaultdict(lambda: 0)
        local_rule_stats(b, stats, 15)

        print(stats)

        print("Shortest path probability")
        best = shortestpath(b)
        best.topsort()
        self.assertAlmostEquals(pow(e, -float(shortestdistance(best)[-1])), 1.80844898756e-05)
        print(best.text())

        polish_rules = retrieve_rules(best)
        self.assertSequenceEqual(polish_rules, [8, 7, 1, 6, 2, 5, 3, 10, 3, 3])

        polish_rules = list(map(rules.index_object, polish_rules))

        print(polish_rules)

        der = PolishDerivation(polish_rules[1::])

        print(der)

        print(derivation_to_hybrid_tree(der, string, "Piet Marie helpen lezen leren".split(), construct_conll_token))

        dcp = The_DCP_evaluator(der).getEvaluation()

        h_tree_2 = HybridTree()
        dcp_to_hybridtree(h_tree_2, dcp, token_sequence, False,
                          construct_conll_token)

        print(h_tree_2)

    def test_fst_compilation_left(self):
        if not test_pynini:
            return
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [left_branching], 'START')

        fst, rules = compile_wfst_from_left_branching_grammar(grammar)

        print(repr(fst))

        symboltable = fst.input_symbols()

        string = ["NP", "N", "V", "V", "V"]

        fsa = fsa_from_list_of_symbols(string, symboltable)
        self.assertEqual(fsa.text().decode('utf-8'), '0\t1\tNP\tNP\n1\t2\tN\tN\n2\t3\tV\tV\n3\t4\tV\tV\n4\t5\tV\tV\n5\n')

        b = compose(fsa, fst)

        print(b.text(symboltable, symboltable))

        print("Shortest path probability", end=' ')
        best = shortestpath(b)
        best.topsort()
        # self.assertAlmostEquals(pow(e, -float(shortestdistance(best)[-1])), 1.80844898756e-05)
        print(best.text())

        polish_rules = retrieve_rules(best)
        self.assertSequenceEqual(polish_rules, [1, 2, 3, 4, 5, 4, 9, 4, 7, 8])

        polish_rules = list(map(rules.index_object, polish_rules))

        for rule in polish_rules:
            print(rule)
        print()

        der = ReversePolishDerivation(polish_rules[0:-1])
        self.assertTrue(der.check_integrity_recursive(der.root_id()))

        print(der)

        LeftBranchingFSTParser.preprocess_grammar(grammar)
        parser = LeftBranchingFSTParser(grammar, string)
        der_ = parser.best_derivation_tree()

        print(der_)
        self.assertTrue(der_.check_integrity_recursive(der_.root_id()))

        print(derivation_to_hybrid_tree(der, string, "Piet Marie helpen lezen leren".split(), construct_conll_token))

        print(derivation_to_hybrid_tree(der_, string, "Piet Marie helpen lezen leren".split(), construct_conll_token))

        dcp = The_DCP_evaluator(der).getEvaluation()

        h_tree_2 = HybridTree()
        token_sequence = [construct_conll_token(form, lemma) for form, lemma in
                          zip('Piet Marie helpen lezen leren'.split(' '), 'NP N V V V'.split(' '))]
        dcp_to_hybridtree(h_tree_2, dcp, token_sequence, False,
                          construct_conll_token)

        print(h_tree_2)

    def test_cfg_parser(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')



        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for parser_class in [LCFRS_parser, CFGParser]:

            parser_class.preprocess_grammar(grammar)

            string = ["NP", "N", "V", "V", "V"]

            parser = parser_class(grammar, string)

            self.assertTrue(parser.recognized())

            der = parser.best_derivation_tree()
            self.assertTrue(der.check_integrity_recursive(der.root_id(), grammar.start()))

            print(der)

            print(derivation_to_hybrid_tree(der, string, "Piet Marie helpen lezen leren".split(), construct_conll_token))

            dcp = The_DCP_evaluator(der).getEvaluation()

            h_tree_2 = HybridTree()
            token_sequence = [construct_conll_token(form, lemma) for form, lemma in
                              zip('Piet Marie helpen lezen leren'.split(' '), 'NP N V V V'.split(' '))]
            dcp_to_hybridtree(h_tree_2, dcp, token_sequence, False,
                              construct_conll_token)

            print(h_tree_2)



def hybrid_tree_1():
    tree = HybridTree()
    tree.add_node('v1', CoNLLToken('Piet', '_', 'NP', 'NP', '_', 'SBJ'), True)
    tree.add_node('v21', CoNLLToken('Marie', '_', 'N', 'N', '_', 'OBJ'), True)
    tree.add_node('v', CoNLLToken('helpen', '_', 'V', 'V', '_', 'ROOT'), True)
    tree.add_node('v2', CoNLLToken('lezen', '_', 'V', 'V', '_', 'VBI'), True)
    tree.add_child('v', 'v2')
    tree.add_child('v', 'v1')
    tree.add_child('v2', 'v21')
    tree.add_to_root('v')
    tree.reorder()
    return tree


def hybrid_tree_2():
    tree2 = HybridTree()
    tree2.add_node('v1', CoNLLToken('Piet', '_', 'NP', 'NP', '_', 'SBJ'), True)
    tree2.add_node('v211', CoNLLToken('Marie', '_', 'N', 'N', '_', 'OBJ'), True)
    tree2.add_node('v', CoNLLToken('helpen', '_', 'V', 'V', '_', 'ROOT'), True)
    tree2.add_node('v2', CoNLLToken('leren', '_', 'V', 'V', '_', 'VBI'), True)
    tree2.add_node('v21', CoNLLToken('lezen', '_', 'V', 'V', '_', 'VFIN'), True)
    tree2.add_child('v', 'v2')
    tree2.add_child('v', 'v1')
    tree2.add_child('v2', 'v21')
    tree2.add_child('v21', 'v211')
    tree2.add_to_root('v')
    tree2.reorder()
    return tree2


if __name__ == '__main__':
    unittest.main()
