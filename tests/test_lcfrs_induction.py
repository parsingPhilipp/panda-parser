import unittest
from corpora.negra_parse import sentence_names_to_hybridtrees
from constituent.induction import direct_extract_lcfrs_from_prebinarized_corpus
from grammar.induction.terminal_labeling import FormTerminals
from parser.naive.parsing import LCFRS_parser
from hybridtree.constituent_tree import HybridTree
from hybridtree.monadic_tokens import construct_constituent_token
from copy import deepcopy
from parser.sDCP_parser.sdcp_parser_wrapper import PysDCPParser
from parser.sDCPevaluation.evaluator import The_DCP_evaluator, dcp_to_hybridtree

class MyTestCase(unittest.TestCase):
    def test_something(self):
        limit = 50000
        # limit = 30
        corpus_bin = sentence_names_to_hybridtrees({str(x) for x in range(limit)}, "/tmp/tiger-bin.export",
                                               disconnect_punctuation=False, add_vroot=True, mode="DISCO-DOP")

        corpus = sentence_names_to_hybridtrees({str(x) for x in range(limit)}, "/tmp/tiger.export",
                                               disconnect_punctuation=False, add_vroot=True, mode="DISCO-DOP")
        terminal_labeling = FormTerminals()

        for htree, htree_bin in zip(corpus, corpus_bin):
            # print(htree_bin)

            try:
                htree_grammar = direct_extract_lcfrs_from_prebinarized_corpus(htree_bin, term_labeling=terminal_labeling)
            except Exception as e:
                print(e)
                print(htree_bin)
                print(htree_bin.nodes())
                print(htree_bin.word_yield())
                raise e
            # print(htree_grammar)

            parser_input = terminal_labeling.prepare_parser_input(htree.token_yield())

            p = PysDCPParser(htree_grammar, terminal_labelling=terminal_labeling)
            p.set_input(htree)
            p.parse()
            # p = LCFRS_parser(htree_grammar, parser_input)
            self.assertTrue(p.recognized())

            derivs = list(p.all_derivation_trees())
            print("derivations:", len(derivs))

            dcp = The_DCP_evaluator(derivs[0]).getEvaluation()
            sys_tree = HybridTree(htree.sent_label())

            sys_tree = dcp_to_hybridtree(sys_tree, dcp, deepcopy(htree.token_yield()), ignore_punctuation=False,
                                         construct_token=construct_constituent_token)
            # print(sys_tree)
            if htree != sys_tree:
                print(htree.sent_label())
                print(htree)
                print(sys_tree)

            self.assertEqual(htree, sys_tree)



if __name__ == '__main__':
    unittest.main()

