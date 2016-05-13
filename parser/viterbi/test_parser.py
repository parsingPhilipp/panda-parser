import unittest

from parser.viterbi.viterbi import ViterbiParser as Parser
from parser.viterbi.viterbi import ViterbiDerivation as Derivation
from parser.derivation_interface import derivation_to_hybrid_tree
from dependency.test_induction import hybrid_tree_1, hybrid_tree_2
from dependency.labeling import the_labeling_factory
from dependency.induction import induce_grammar, direct_extraction, the_terminal_labeling_factory
from hybridtree.monadic_tokens import *
from parser.sDCPevaluation.evaluator import *
from hybridtree.general_hybrid_tree import HybridTree


class MyTestCase(unittest.TestCase):
    def test_dcp_evaluation_with_induced_dependency_grammar(self):
        tree = hybrid_tree_1()

        print tree

        tree2 = hybrid_tree_2()

        print tree2
        # print tree.recursive_partitioning()

        labeling = the_labeling_factory().create_simple_labeling_strategy('child', 'pos')
        term_pos = the_terminal_labeling_factory().get_strategy('pos').token_label
        (_, grammar) = induce_grammar([tree, tree2], labeling, term_pos, direct_extraction, 'START')

        # print grammar

        self.assertEqual(grammar.well_formed(), None)
        self.assertEqual(grammar.ordered()[0], True)
        # print max([grammar.fanout(nont) for nont in grammar.nonts()])
        print grammar

        parser = Parser(grammar, 'NP N V V'.split(' '))

        self.assertEqual(parser.recognized(), True)


        der = parser.best_derivation_tree()
        print der

        hybrid_tree = derivation_to_hybrid_tree(der, 'NP N V V'.split(' '), 'Piet Marie helpen lezen'.split(' '),
                                                construct_constituent_token)
        print hybrid_tree

        dcp = The_DCP_evaluator(der).getEvaluation()
        h_tree_2 = HybridTree()
        token_sequence = [construct_conll_token(form, lemma) for form, lemma in
                          zip('Piet Marie helpen lezen'.split(' '), 'NP N V V'.split(' '))]
        dcp_to_hybridtree(h_tree_2, dcp, token_sequence, False,
                          construct_conll_token)

            # correct = h_tree_2.__eq__(tree) or h_tree_2.__eq__(tree2)
            # self.assertEqual(correct, True)

if __name__ == '__main__':
    unittest.main()
