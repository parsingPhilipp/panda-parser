import unittest
from parser.sDCP_parser.sdcp_parser_wrapper import print_grammar
from tests.test_induction import hybrid_tree_1, hybrid_tree_2
from dependency.induction import the_terminal_labeling_factory, induce_grammar, cfg
from dependency.labeling import the_labeling_factory
from sys import stderr

class MyTestCase(unittest.TestCase):
    def test_something(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for rule in grammar.rules():
            print >>stderr, rule

        # x = input("Press any key")

        print_grammar(grammar)



if __name__ == '__main__':
    unittest.main()
