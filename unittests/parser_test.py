__author__ = 'kilian'

import unittest
from parser.active.lcfrs_parser_new import *
from lcfrs import *


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.grammar_ab_copy = create_copy_grammar()
        self.grammar_ab_copy_2 = create_copy_grammar_2()

    def test_a4(self):
        word = ['a'] * 4
        parser = Parser(self.grammar_ab_copy, word)
        print "Parse", word
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        self.assertEqual(counter, 2)
        print

    def test_aabaab(self):
        word = ['a', 'a', 'b'] * 2
        parser = Parser(self.grammar_ab_copy, word )
        print "Parse", word
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        self.assertEqual(counter, 2)
        print

    def test_abba(self):
        word = ['a', 'b', 'b', 'a']
        parser = Parser(self.grammar_ab_copy, word )
        print "Parse", word
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        self.assertEqual(counter, 0)
        print

    def test_a4_2(self):
        word = ['a'] * 4
        parser = Parser(self.grammar_ab_copy_2, word )
        print "Parse", word
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        self.assertEqual(counter, 2)
        print

    def test_aabaab_2(self):
        word = ['a', 'a', 'b'] * 2
        parser = Parser(self.grammar_ab_copy_2, word)
        print "Parse", word
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        self.assertEqual(counter, 2)
        print

    def test_baabbaab_2(self):
        word = ['b', 'a', 'a', 'b'] * 2
        parser = Parser(self.grammar_ab_copy_2, word)
        print "Parse", word
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        self.assertEqual(counter, 2)
        print

    def test_abba_2(self):
        word = ['a', 'b', 'b', 'a']
        parser = Parser(self.grammar_ab_copy_2, word )
        print "Parse", word
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        self.assertEqual(counter, 0)
        print

    def test_remaining_terminal_function(self):
        x1 = LCFRS_var(1,0)
        x2 = LCFRS_var(1,1)
        lhs = LCFRS_lhs('S')
        lhs.add_arg(['a', x1, 'a', x2])
        rule = LCFRS_rule(lhs)
        n = number_of_consumed_terminals(rule, 0, 3)
        self.assertEqual(n, 0)

    def test_ambncmdn(self):
        m = 6
        n = 3
        word = (['a'] * m + ['b'] * n + ['c'] * m + ['d'] * n)
        print "Parse", word
        parser = Parser(kaeshhammer_grammar(), word)
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        self.assertEqual(counter, 1)
        print

    def test_ambncmdn_fail(self):
        m = 6
        n = 3
        word = (['a'] * m + ['b'] * n + ['c'] * (m + 1) + ['d'] * n)
        print "Parse", word
        parser = Parser(kaeshhammer_grammar(), word)
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        self.assertEqual(counter, 0)
        print

    def test_kallmayer_pos(self):
        for n in range(4):
            for m in range(4):
                word = ((['c'] + ['a'] * 2 * n + ['b'] * m) * 2 + ['c'] + ['a'] * 2 * n )
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
        print "Parse", word
        parser = Parser(kallmayar_grammar(), word)
        counter = 0
        print "Found items:"
        for passive_item in parser.query_passive_items('S', [0]):
            if passive_item.range(0) == Range(0, len(word)):
                print passive_item
                counter += 1
        print
        return counter


if __name__ == '__main__':
    unittest.main()

def create_copy_grammar():
    grammar = LCFRS('S')

    x1 = LCFRS_var(1,0)
    x2 = LCFRS_var(1,1)

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

    x1 = LCFRS_var(1,0)
    x2 = LCFRS_var(1,1)

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


def kaeshhammer_grammar():
    grammar = LCFRS('S')

    x1 = LCFRS_var(1,0)
    x2 = LCFRS_var(1,1)

    y1 = LCFRS_var(2,0)
    y2 = LCFRS_var(2,1)

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
    lhs5.add_arg([x1, y1, x2, y2])
    grammar.add_rule(lhs5, ['A', 'B'])

    return grammar


def kallmayar_grammar():
    # p. 174/175, Problem 8.2 Grammar G_1
    # made epsilon free
    grammar = LCFRS('S')

    x1 = LCFRS_var(1,0)
    x2 = LCFRS_var(1,1)
    x3 = LCFRS_var(1,2)

    y1 = LCFRS_var(2,0)
    y2 = LCFRS_var(2,1)

    lhs1 = LCFRS_lhs('S')
    lhs1.add_arg(['c', x1, 'c', x2, 'c', x3])
    grammar.add_rule(lhs1, ['A'])

    lhs2 = LCFRS_lhs('S')
    lhs2.add_arg(['c', x1, y1, 'c', x2, y2, 'c', x3])
    grammar.add_rule(lhs2, ['A', 'B'])

    lhs3= LCFRS_lhs('A')
    for i in range(3):
        lhs3.add_arg(['a', LCFRS_var(1, i), 'a'])
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

    return grammar