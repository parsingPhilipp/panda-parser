__author__ = 'kilian'

import unittest
from lcfrs_parser_new import *
from lcfrs import *

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.grammar_ab_copy = create_grammar()

    def test_a4(self):
        parser = Parser(self.grammar_ab_copy, ['a']* 4 )
        print parser.query_passive_items('S', [0])

if __name__ == '__main__':
    unittest.main()

def create_grammar():
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