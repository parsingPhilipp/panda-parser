from __future__ import print_function

__author__ = 'kilian'

import unittest

from parser.naive.parsing import *
from tests.test_parser.test_active_parser import ambiguous_copy_grammar


class PassiveParserTest(unittest.TestCase):
    def test_LHS_instance(self):
        lhs = LHS_instance("A")
        lhs.add_arg()
        for mem in [Span(0, 2), "foo", Span(3, 4)]:
            lhs.add_mem(mem)
        self.assertEqual(lhs.consistent(), True)

        lhs = LHS_instance("A")
        lhs.add_arg()
        for mem in [Span(0, 2), Span(3, 4)]:
            lhs.add_mem(mem)
        self.assertEqual(lhs.consistent(), False)
        self.assertEqual(lhs.next_member_bounds(0, 3), (0, 3))

        lhs = LHS_instance("A")
        lhs.add_arg()
        for mem in [Span(0, 2), Span(2, 4)]:
            lhs.add_mem(mem)
        lhs.add_arg()

        for mem in [Span(6, 9), Span(9, 12)]:
            lhs.add_mem(mem)

        self.assertEqual(lhs.consistent(), True)

        self.assertEqual(str(lhs),
                         "A(Span(low=0, high=2) Span(low=2, high=4); Span(low=6, high=9) Span(low=9, high=12))")

        lhs.collapse()

        self.assertEqual(str(lhs), "A(Span(low=0, high=4); Span(low=6, high=12))")

    def test_naive_parser(self):
        grammar = ambiguous_copy_grammar()
        self.assertEqual(grammar.well_formed(), None)
        self.assertEqual(grammar.ordered()[0], True)

        word = ['a'] * 18

        parser2 = LCFRS_parser(grammar, word)
        derivation = parser2.best_derivation_tree()
        print(derivation)


if __name__ == '__main__':
    unittest.main()
