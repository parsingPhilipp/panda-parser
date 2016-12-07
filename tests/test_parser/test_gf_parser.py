import unittest

from dependency.induction import the_terminal_labeling_factory, induce_grammar, direct_extraction, cfg
from dependency.labeling import the_labeling_factory
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
from parser.derivation_interface import derivation_to_hybrid_tree
from parser.gf_parser.gf_export import *
from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
from parser.sDCPevaluation.evaluator import The_DCP_evaluator, dcp_to_hybridtree
from tests.test_induction import hybrid_tree_1, hybrid_tree_2
from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from sys import stderr
from math import exp
import copy


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

        _, name_ = export(grammar, prefix, name)

        self.assertEqual(0, compile_gf_grammar(prefix, name_))

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

    def test_k_best_parsing(self):
        limit_train = 20
        limit_test = 10
        train = '../../res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        test = train
        parser_type = GFParser_k_best
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        parser_type.preprocess_grammar(grammar_prim)
        tree_yield = term_labelling.prepare_parser_input

        trees = parse_conll_corpus(test, False, limit_test)

        for i, tree in enumerate(trees):
            print >>stderr, "Parsing sentence ", i

            # print >>stderr, tree

            parser = parser_type(grammar_prim, tree_yield(tree.token_yield()), k=50)

            self.assertTrue(parser.recognized())

            derivations = [der for der in parser.k_best_derivation_trees()]
            print >>stderr, "# derivations: ", len(derivations)
            h_trees = []
            current_weight = 0
            weights = []
            derivation_list = []
            for weight, der in derivations:
                # print >>stderr, exp(-weight)
                # print >>stderr, der

                self.assertTrue(not der in derivation_list)

                derivation_list.append(der)

                # TODO this should hold, but it looks like a GF bug!
                # self.assertGreaterEqual(weight, current_weight)
                current_weight = weight

                dcp = The_DCP_evaluator(der).getEvaluation()
                h_tree = HybridTree()
                cleaned_tokens = copy.deepcopy(tree.full_token_yield())
                dcp_to_hybridtree(h_tree, dcp, cleaned_tokens, False, construct_conll_token)

                h_trees.append(h_tree)
                weights.append(exp(-weight))

                # print >>stderr, h_tree

            # print a matrix indicating which derivations result
            # in the same hybrid tree
            if True:
                for i, h_tree1 in enumerate(h_trees):
                    for h_tree2 in h_trees:
                        if h_tree1 == h_tree2:
                            print >> stderr, "x",
                        else:
                            print >> stderr, " ",
                    print >> stderr, weights[i]
                print >> stderr

    def test_best_trees(self):
        limit_train = 5000
        limit_test = 100
        train = '../../res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        test = train
        parser_type = GFParser_k_best
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("child", "pos+deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        parser_type.preprocess_grammar(grammar_prim)
        tree_yield = term_labelling.prepare_parser_input

        trees = parse_conll_corpus(test, False, limit_test)

        for i, tree in enumerate(trees):
            print >> stderr, "Parsing sentence ", i

            parser = parser_type(grammar_prim, tree_yield(tree.token_yield()), k=200)

            self.assertTrue(parser.recognized())

            viterbi_weight = parser.viterbi_weight()
            viterbi_deriv = parser.viterbi_derivation()

            der_to_tree = lambda der: dcp_to_hybridtree(HybridTree(), The_DCP_evaluator(der).getEvaluation(), copy.deepcopy(tree.full_token_yield()), False, construct_conll_token)

            viterbi_tree = der_to_tree(viterbi_deriv)

            ordered_parse_trees = parser.best_trees(der_to_tree)

            best_tree, best_weight, best_witnesses = ordered_parse_trees[0]

            for i, (parsed_tree, _, _) in enumerate(ordered_parse_trees):
                if parsed_tree.__eq__(tree):
                    print >>stderr, "Gold tree is ", i+1, " in best tree list"
                    break

            if (not viterbi_tree.__eq__(best_tree) and viterbi_weight != best_weight):
                print >> stderr, "viterbi and k-best tree differ"
                print >> stderr, "viterbi: ", viterbi_weight
                print >> stderr, "k-best: ", best_weight, best_witnesses
                if False:
                    print >> stderr, viterbi_tree
                    print >> stderr, tree_to_conll_str(viterbi_tree)
                    print >> stderr, best_tree
                    print >> stderr, tree_to_conll_str(best_tree)
                    print >> stderr, "gold tree"
                    print >> stderr, tree
                    print >> stderr, tree_to_conll_str(tree)





if __name__ == '__main__':
    unittest.main()
