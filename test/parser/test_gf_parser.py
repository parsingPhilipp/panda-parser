import unittest
from dependency.test_induction import hybrid_tree_1, hybrid_tree_2
from dependency.induction import the_terminal_labeling_factory, induce_grammar, direct_extraction
from dependency.labeling import the_labeling_factory
from hybridtree.monadic_tokens import construct_conll_token
from hybridtree.general_hybrid_tree import HybridTree

from parser.gf_parser.gf_export import *
from parser.gf_parser.gf_interface import GFParser
from parser.derivation_interface import derivation_to_hybrid_tree
from parser.sDCPevaluation.evaluator import The_DCP_evaluator, dcp_to_hybridtree


class MyTestCase(unittest.TestCase):
    def test_grammar_export(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')



        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty','pos'),
                                      # the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel'),
                                      terminal_labeling.token_label, [direct_extraction], 'START')
        print max([grammar.fanout(nont) for nont in grammar.nonts()])
        print grammar

        prefix = '/tmp/'
        name = 'tmpGrammar'

        export(grammar, prefix, name)

        self.assertEqual(0, compile_gf_grammar(prefix, name))

        GFParser.preprocess_grammar(grammar)

        string = ["NP", "N", "V", "V", "V"]

        parser = GFParser(grammar, string)

        self.assertTrue(parser.recognized())

        der = parser.best_derivation_tree()
        self.assertTrue(der.check_integrity_recursive(der.root_id(), grammar.start()))

        print der

        print derivation_to_hybrid_tree(der, string, "Piet Marie helpen lezen leren".split(), construct_conll_token)

        dcp = The_DCP_evaluator(der).getEvaluation()

        h_tree_2 = HybridTree()
        token_sequence = [construct_conll_token(form, lemma) for form, lemma in
                          zip('Piet Marie helpen lezen leren'.split(' '), 'NP N V V V'.split(' '))]
        dcp_to_hybridtree(h_tree_2, dcp, token_sequence, False,
                          construct_conll_token)

        print h_tree_2

        #self.assertEqual(True, False)



if __name__ == '__main__':
    unittest.main()
