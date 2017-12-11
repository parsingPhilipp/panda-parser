from __future__ import print_function, division

__author__ = 'kilian'

import unittest

from dependency.induction import induce_grammar
from grammar.induction.recursive_partitioning import direct_extraction
from grammar.induction.terminal_labeling import the_terminal_labeling_factory
from dependency.labeling import the_labeling_factory
from grammar.lcfrs import *
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import *
from parser.active.derivation import Derivation
from parser.active.parsing import *
from parser.derivation_interface import derivation_to_hybrid_tree
from parser.sDCPevaluation.evaluator import The_DCP_evaluator, dcp_to_hybridtree
from tests.test_induction import hybrid_tree_1, hybrid_tree_2


class ActiveParserTest(unittest.TestCase):
    def setUp(self):
        self.grammar_ab_copy = create_copy_grammar()
        self.assertEqual(self.grammar_ab_copy.ordered()[0], True)
        self.grammar_ab_copy_2 = create_copy_grammar_2()
        self.assertEqual(self.grammar_ab_copy_2.ordered()[0], True)

    def test_a4(self):
        word = ['a'] * 4
        parser = Parser(self.grammar_ab_copy, word)
        print("Parse", word)
        counter = 0
        print("Found items:")

        for passive_item in parser.successful_root_items():
            print(passive_item)
            derivation = print_derivation_tree(passive_item)
            print(derivation)
            poss = ['P' + str(i) for i in range(1, len(word) + 1)]
            tree = derivation_to_hybrid_tree(derivation, poss, word, construct_constituent_token)
            print(tree)
            counter += 1
        self.assertEqual(counter, 2)
        print()

    def test_aabaab(self):
        word = ['a', 'a', 'b'] * 2
        parser = Parser(self.grammar_ab_copy, word)
        print("Parse", word)
        counter = 0
        print("Found items:")
        for passive_item in parser.successful_root_items():
            print(passive_item)
            counter += 1
        self.assertEqual(counter, 2)
        print()

    def test_abba(self):
        word = ['a', 'b', 'b', 'a']
        parser = Parser(self.grammar_ab_copy, word)
        print("Parse", word)
        counter = 0
        print("Found items:")
        for passive_item in parser.successful_root_items():
            print(passive_item)
            counter += 1

        self.assertEqual(counter, 0)
        print()

    def test_a4_2(self):
        word = ['a'] * 4
        parser = Parser(self.grammar_ab_copy_2, word)
        print("Parse", word)
        counter = 0
        print("Found items:")
        for passive_item in parser.successful_root_items():
            print(passive_item)
            counter += 1
        self.assertEqual(counter, 2)
        print()

    def test_aabaab_2(self):
        word = ['a', 'a', 'b'] * 2
        parser = Parser(self.grammar_ab_copy_2, word)
        print("Parse", word)
        counter = 0
        print("Found items:")
        for passive_item in parser.successful_root_items():
            print(passive_item)
            counter += 1
        self.assertEqual(counter, 2)
        print()

    def test_baabbaab_2(self):
        word = ['b', 'a', 'a', 'b'] * 2
        parser = Parser(self.grammar_ab_copy_2, word)
        print("Parse", word)
        counter = 0
        print("Found items:")
        for passive_item in parser.successful_root_items():
            print(passive_item)
            counter += 1
        self.assertEqual(counter, 2)
        print()

    def test_abba_2(self):
        word = ['a', 'b', 'b', 'a']
        parser = Parser(self.grammar_ab_copy_2, word)
        print("Parse", word)
        counter = 0
        print("Found items:")
        for passive_item in parser.successful_root_items():
            print(passive_item)
            counter += 1
        self.assertEqual(counter, 0)
        print()

    def test_remaining_terminal_function(self):
        x1 = LCFRS_var(0, 0)
        x2 = LCFRS_var(0, 1)
        lhs = LCFRS_lhs('S')
        lhs.add_arg(['a', x1, 'a', x2])
        rule = LCFRS_rule(lhs)
        n = number_of_consumed_terminals(rule, 0, 3, 0)
        self.assertEqual(n, 0)

    def test_ambncmdn(self):
        m = 6
        n = 3
        word = (['a'] * m + ['b'] * n + ['c'] * m + ['d'] * n)
        print("Parse", word)
        parser = Parser(kaeshammer_grammar(), word)
        counter = 0
        print("Found items:")
        for passive_item in parser.successful_root_items():
            print(passive_item)
            counter += 1
            derivation = print_derivation_tree(passive_item)
            print(derivation)
            hybrid_tree = derivation_to_hybrid_tree(derivation, word, word, construct_constituent_token)
            # print hybrid_tree
        self.assertEqual(counter, 1)
        print()

    def test_ambncmdn_fail(self):
        m = 6
        n = 3
        word = (['a'] * m + ['b'] * n + ['c'] * (m + 1) + ['d'] * n)
        print("Parse", word)
        parser = Parser(kaeshammer_grammar(), word)
        counter = 0
        print("Found items:")
        for passive_item in parser.successful_root_items():
            print(passive_item)
            counter += 1
        self.assertEqual(counter, 0)
        print()

    def test_kallmayer_pos(self):
        for n in range(4):
            for m in range(4):
                word = ((['c'] + ['a'] * 2 * n + ['b'] * m) * 2 + ['c'] + ['a'] * 2 * n)
                counter = self._kallmayer(word)
                if n > 0:
                    self.assertEqual(counter, 1)
                else:
                    self.assertEqual(counter, 0)

    def test_kallmayer_neg(self):
        word = 'c a a a b c a a a b c a a a'.split(' ')
        counter = self._kallmayer(word)
        self.assertEqual(counter, 0)

    def _kallmayer(self, word):
        """
        :return:
        """
        print("Parse", word)
        parser = Parser(kallmeyer_grammar(), word)
        counter = 0
        print("Found items:")
        for passive_item in parser.successful_root_items():
            print(passive_item)
            counter += 1
            # print_derivation_tree(passive_item)
        print()
        return counter

    def test_dcp_evaluation_with_induced_dependency_grammar(self):
        tree = hybrid_tree_1()

        print(tree)

        tree2 = hybrid_tree_2()

        print(tree2)
        # print tree.recursive_partitioning()

        labeling = the_labeling_factory().create_simple_labeling_strategy('child', 'pos')
        term_pos = the_terminal_labeling_factory().get_strategy('pos').token_label
        (_, grammar) = induce_grammar([tree, tree2], labeling, term_pos, [direct_extraction], 'START')

        # print grammar

        self.assertEqual(grammar.well_formed(), None)
        self.assertEqual(grammar.ordered()[0], True)
        # print max([grammar.fanout(nont) for nont in grammar.nonts()])
        print(grammar)

        parser = Parser(grammar, 'NP N V V'.split(' '))

        self.assertEqual(parser.recognized(), True)

        for item in parser.successful_root_items():
            der = Derivation()
            derivation_tree(der, item, None)
            print(der)

            hybrid_tree = derivation_to_hybrid_tree(der, 'NP N V V'.split(' '), 'Piet Marie helpen lezen'.split(' '),
                                                    construct_constituent_token)
            print(hybrid_tree)

            dcp = The_DCP_evaluator(der).getEvaluation()
            h_tree_2 = HybridTree()
            token_sequence = [construct_conll_token(form, lemma) for form, lemma in
                              zip('Piet Marie helpen lezen'.split(' '), 'NP N V V'.split(' '))]
            dcp_to_hybridtree(h_tree_2, dcp, token_sequence, False,
                              construct_conll_token)

            # correct = h_tree_2.__eq__(tree) or h_tree_2.__eq__(tree2)
            # self.assertEqual(correct, True)

    def test_ambiguous_copy_grammar(self):
        grammar = ambiguous_copy_grammar()
        self.assertEqual(None, grammar.well_formed())
        self.assertTrue(grammar.ordered()[0])

        word = ['a'] * 16
        parser = Parser(grammar, word)
        counter = 0

        # for passive_item in test_parser.query_passive_items('A', [0]):
        # if passive_item.range(LCFRS_var(-1, 0)) != Range(0, 4):
        # continue
        # else:
        # print passive_item
        # print print_derivation_tree(passive_item)
        #         print
        # print "###############"

        for passive_item in parser.successful_root_items():
            # print passive_item
            derivation = print_derivation_tree(passive_item)
            # print derivation
            # poss = ['P' + str(i) for i in range(1, len(word) + 1)]
            # tree = derivation_to_hybrid_tree(derivation, poss, word)
            # print tree
            counter += 1
        self.assertEqual(counter, number_of_ambiguous_trees(len(word) // 2))
        print(counter)


def number_of_ambiguous_trees(n):
    assert (n >= 1)
    if n == 1:
        return 1
    c = 0
    for i in range(1, n):
        c += number_of_ambiguous_trees(i) * number_of_ambiguous_trees(n - i)
    return c


if __name__ == '__main__':
    unittest.main()


def ambiguous_copy_grammar():
    grammar = LCFRS('S')

    x1 = LCFRS_var(0, 0)
    x2 = LCFRS_var(0, 1)
    y1 = LCFRS_var(1, 0)
    y2 = LCFRS_var(1, 1)

    lhs1 = LCFRS_lhs('S')
    lhs1.add_arg([x1, x2])
    grammar.add_rule(lhs1, ['A'])

    lhs2 = LCFRS_lhs('A')
    lhs2.add_arg(['a'])
    lhs2.add_arg(['a'])
    grammar.add_rule(lhs2, [])

    lhs3 = LCFRS_lhs('A')
    lhs3.add_arg(['b'])
    lhs3.add_arg(['b'])
    grammar.add_rule(lhs3, [])

    lhs4 = LCFRS_lhs('A')
    lhs4.add_arg([x1, y1])
    lhs4.add_arg([x2, y2])
    grammar.add_rule(lhs4, ['A', 'A'])

    return grammar


def create_copy_grammar():
    grammar = LCFRS('S')

    x1 = LCFRS_var(0, 0)
    x2 = LCFRS_var(0, 1)

    lhs1 = LCFRS_lhs('S')
    lhs1.add_arg([x1, x2])
    grammar.add_rule(lhs1, ['A'])

    lhs2 = LCFRS_lhs('S')
    lhs2.add_arg(['a', x1, 'a', x2])
    grammar.add_rule(lhs2, ['A'])

    lhs3 = LCFRS_lhs('S')
    lhs3.add_arg(['b', x1, 'b', x2])
    grammar.add_rule(lhs3, ['A'])

    lhs4 = LCFRS_lhs('A')
    lhs4.add_arg(['a', x1])
    lhs4.add_arg(['a', x2])
    grammar.add_rule(lhs4, ['A'])

    lhs5 = LCFRS_lhs('A')
    lhs5.add_arg(['b', x1])
    lhs5.add_arg(['b', x2])
    grammar.add_rule(lhs5, ['A'])

    lhs6 = LCFRS_lhs('A')
    lhs6.add_arg(['a'])
    lhs6.add_arg(['a'])
    grammar.add_rule(lhs6, [])

    lhs7 = LCFRS_lhs('A')
    lhs7.add_arg(['b'])
    lhs7.add_arg(['b'])
    grammar.add_rule(lhs7, [])

    return grammar


def create_copy_grammar_2():
    grammar = LCFRS('S')

    x1 = LCFRS_var(0, 0)
    x2 = LCFRS_var(0, 1)

    lhs1 = LCFRS_lhs('S')
    lhs1.add_arg([x1, x2])
    grammar.add_rule(lhs1, ['A'])

    lhs2 = LCFRS_lhs('S')
    lhs2.add_arg([x1, 'a', x2, 'a'])
    grammar.add_rule(lhs2, ['A'])

    lhs3 = LCFRS_lhs('S')
    lhs3.add_arg([x1, 'b', x2, 'b'])
    grammar.add_rule(lhs3, ['A'])

    lhs4 = LCFRS_lhs('A')
    lhs4.add_arg([x1, 'a'])
    lhs4.add_arg([x2, 'a'])
    grammar.add_rule(lhs4, ['A'])

    lhs5 = LCFRS_lhs('A')
    lhs5.add_arg([x1, 'b'])
    lhs5.add_arg([x2, 'b'])
    grammar.add_rule(lhs5, ['A'])

    lhs6 = LCFRS_lhs('A')
    lhs6.add_arg(['a'])
    lhs6.add_arg(['a'])
    grammar.add_rule(lhs6, [])

    lhs7 = LCFRS_lhs('A')
    lhs7.add_arg(['b'])
    lhs7.add_arg(['b'])
    grammar.add_rule(lhs7, [])

    return grammar


def kaeshammer_grammar():
    grammar = LCFRS('S')

    x1 = LCFRS_var(0, 0)
    x2 = LCFRS_var(0, 1)

    y1 = LCFRS_var(1, 0)
    y2 = LCFRS_var(1, 1)

    lhs1 = LCFRS_lhs('A')
    lhs1.add_arg(['a'])
    lhs1.add_arg(['c'])
    grammar.add_rule(lhs1, [])

    lhs2 = LCFRS_lhs('B')
    lhs2.add_arg(['b'])
    lhs2.add_arg(['d'])
    grammar.add_rule(lhs2, [])

    lhs3 = LCFRS_lhs('A')
    lhs3.add_arg(['a', x1])
    lhs3.add_arg(['c', x2])
    grammar.add_rule(lhs3, ['A'])

    lhs4 = LCFRS_lhs('B')
    lhs4.add_arg(['b', x1])
    lhs4.add_arg(['d', x2])
    grammar.add_rule(lhs4, ['B'])

    lhs5 = LCFRS_lhs('S')
    lhs5.add_arg([y1, x1, y2, x2])
    grammar.add_rule(lhs5, ['B', 'A'])
    assert grammar.ordered()[0]
    return grammar


def kallmeyer_grammar():
    # p. 174/175, Problem 8.2 Grammar G_1
    # made epsilon free
    grammar = LCFRS('S')

    x1 = LCFRS_var(0, 0)
    x2 = LCFRS_var(0, 1)
    x3 = LCFRS_var(0, 2)

    y1 = LCFRS_var(1, 0)
    y2 = LCFRS_var(1, 1)

    lhs1 = LCFRS_lhs('S')
    lhs1.add_arg(['c', x1, 'c', x2, 'c', x3])
    grammar.add_rule(lhs1, ['A'])

    lhs2 = LCFRS_lhs('S')
    lhs2.add_arg(['c', x1, y1, 'c', x2, y2, 'c', x3])
    grammar.add_rule(lhs2, ['A', 'B'])

    lhs3 = LCFRS_lhs('A')
    for i in range(3):
        lhs3.add_arg(['a', LCFRS_var(0, i), 'a'])
    grammar.add_rule(lhs3, ['A'])

    lhs4 = LCFRS_lhs('A')
    for _ in range(3):
        lhs4.add_arg(['a', 'a'])
    grammar.add_rule(lhs4, [])

    lhs5 = LCFRS_lhs('B')
    lhs5.add_arg(['b', x1])
    lhs5.add_arg(['b', x2])
    grammar.add_rule(lhs5, ['B'])

    lhs6 = LCFRS_lhs('B')
    lhs6.add_arg(['b'])
    lhs6.add_arg(['b'])
    grammar.add_rule(lhs6, [])

    assert (grammar.ordered()[0])
    return grammar


def print_derivation_tree(root_element):
    derivation = Derivation()
    derivation_tree(derivation, root_element, None)
    return derivation
