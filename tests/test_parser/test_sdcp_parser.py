import unittest
from parser.sDCP_parser.sdcp_parser_wrapper import print_grammar, PysDCPParser, LCFRS_sDCP_Parser, SDCPDerivation
from parser.sDCP_parser.sm_trainer import compute_reducts, PyEMTrainer, PySDCPTraceManager, split_merge_training
from tests.test_induction import hybrid_tree_1, hybrid_tree_2
from dependency.induction import the_terminal_labeling_factory, induce_grammar, cfg
from dependency.labeling import the_labeling_factory
from parser.sDCPevaluation.evaluator import dcp_to_hybridtree, The_DCP_evaluator
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
from corpora.conll_parse import parse_conll_corpus
from sys import stderr
import cPickle as pickle

class MyTestCase(unittest.TestCase):
    def test_basic_sdcp_parsing(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for rule in grammar.rules():
            print >>stderr, rule

        parser_type = LCFRS_sDCP_Parser

        print >>stderr, "preprocessing grammar"

        parser_type.preprocess_grammar(grammar)

        print >>stderr, "invoking parser"

        parser = parser_type(grammar, tree)

        print >>stderr, "listing derivations"

        for der in parser.all_derivation_trees():
            print der
            output_tree = HybridTree()
            tokens = tree.token_yield()
            dcp_to_hybridtree(output_tree, The_DCP_evaluator(der).getEvaluation(), tokens, False, construct_conll_token)
            print tree
            print output_tree

        print >>stderr, "completed test"

    def test_basic_em_training(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for rule in grammar.rules():
            print >>stderr, rule

        print >>stderr, "compute reducts"

        trace = compute_reducts(grammar, [tree, tree2])

        print >>stderr, "call em Training"
        emTrainer = PyEMTrainer(trace)
        emTrainer.em_training(grammar, n_epochs=10)

        print >>stderr, "finished em Training"

        for rule in grammar.rules():
            print >>stderr, rule

    def test_corpus_em_training(self):
        train = '../../res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        limit_train = 200
        test = train
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        # for rule in grammar.rules():
        #    print >>stderr, rule

        trees = parse_conll_corpus(train, False, limit_train)

        print >>stderr, "compute reducts"

        trace = compute_reducts(grammar_prim, trees)

        print >>stderr, "call em Training"
        emTrainer = PyEMTrainer(trace)
        emTrainer.em_training(grammar_prim, 20, tie_breaking=True, init="equal", sigma=0.05, seed=50)

        print >>stderr, "finished em Training"

        # for rule in grammar.rules():
        #     print >>stderr, rule


    def test_corpus_sdcp_parsing(self):
        # parser_type = PysDCPParser
        print >>stderr, "testing (plain) sDCP parser"
        self.generic_parsing_test(PysDCPParser, 50, 20, False)

    def test_corpus_lcfrs_sdcp_parsing(self):
        print >> stderr, "testing LCFRS/sDCP hybrid parser"
        self.generic_parsing_test(LCFRS_sDCP_Parser, 2000, 100, True)

    def generic_parsing_test(self, parser_type, limit_train, limit_test, compare_order):
        def filter_by_id(n, trees):
            j = 0
            for tree in trees:
                if j in n:
                    yield tree
                j += 1
        #params
        train = '../../res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        test = train
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                     recursive_partitioning, start)

        parser_type.preprocess_grammar(grammar_prim)

        trees = parse_conll_corpus(test, False, limit_test)

        count_derivs = {}
        no_complete_match = 0

        for i, tree in enumerate(trees):
            print >>stderr, "Parsing tree for ", i

            print >>stderr, tree

            parser = parser_type(grammar_prim, tree)
            self.assertTrue(parser.recognized())
            count_derivs[i] = 0

            print >>stderr, "Found derivations for ", i
            j = 0

            derivations = []

            for der in parser.all_derivation_trees():
                self.assertTrue(der.check_integrity_recursive(der.root_id(), start))

                print >>stderr, count_derivs[i]
                print >>stderr, der

                output_tree = HybridTree()
                tokens = tree.token_yield()

                the_yield = der.compute_yield()
                # print >>stderr, the_yield
                tokens2 = map(lambda pos: construct_conll_token('_', pos), the_yield)

                dcp_to_hybridtree(output_tree, The_DCP_evaluator(der).getEvaluation(), tokens2, False, construct_conll_token, reorder=False)
                print >>stderr, tree
                print >>stderr, output_tree

                self.compare_hybrid_trees(tree, output_tree, compare_order)
                count_derivs[i] += 1
                derivations.append(der)

            self.assertTrue(MyTestCase.pairwise_different(derivations, MyTestCase.compare_derivations))
            self.assertEqual(len(derivations), count_derivs[i])

            if count_derivs[i] == 0:
                no_complete_match += 1

        for key in count_derivs:
            print key, count_derivs[key]

        print "# trees with no complete match:", no_complete_match


    def compare_hybrid_trees(self, tree1, tree2, compare_order=False):
        self.assertTrue(isinstance(tree1, HybridTree))
        self.assertTrue(isinstance(tree2, HybridTree))
        self.compare_hybrid_trees_rec(tree1, tree1.root, tree2, tree2.root, compare_order)

    def compare_hybrid_trees_rec(self, tree1, ids1, tree2, ids2, compare_order):
        self.assertEqual(len(ids1), len(ids2))
        for id1, id2 in zip(ids1, ids2):
            token1 = tree1.node_token(id1)
            token2 = tree2.node_token(id2)
            self.assertEqual(token1.pos(), token2.pos())
            self.assertEqual(token1.deprel(), token2.deprel())
            if (compare_order):
                self.assertEqual(tree1.in_ordering(id1), tree2.in_ordering(id2))
                if (tree1.in_ordering(id1)):
                    self.assertEqual(tree1.node_index(id1), tree2.node_index(id2))
            self.compare_hybrid_trees_rec(tree1, tree1.children(id1), tree2, tree2.children(id2), compare_order)

    @staticmethod
    def compare_derivations(der1, der2):
        assert isinstance(der1, SDCPDerivation)
        assert isinstance(der2, SDCPDerivation)
        # return str(der1) == str(der2)

        id1 = der1.root_id()
        id2 = der2.root_id()

        MyTestCase.compare_derivation_recursive(der1, id1, der2, id2)

    @staticmethod
    def compare_derivation_recursive(der1, id1, der2, id2):
        """
        :param der1:
        :type der1: SDCPDerivation
        :param id1:
        :type id1:
        :param der2:
        :type der2: SDCPDerivation
        :param id2:
        :type id2:
        :return:
        :rtype:
        """
        if not der1.getRule(id1) == der2.getRule(id2):
            return False
        if not len(der1.child_ids(id1)) == len(der2.child_ids(id2)):
            return False
        for i in range(len(der1.child_ids(id1))):
            if not MyTestCase.compare_derivation_recursive(der1, der1.child_id(id1, i), der2, der2.child_id(id2, i)):
                return False
        return True

    @staticmethod
    def pairwise_different(l, comparator):
        for i in range(len(l)):
            for j in range(i + 1, len(l)):
                if comparator(l[i], l[j]):
                    return False
        return True

    def test_trace_serialization(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for rule in grammar.rules():
            print >>stderr, rule

        trace = compute_reducts(grammar, [tree, tree2])
        trace.serialize("/tmp/reducts.p")

        grammar_load = grammar
        trace2 = PySDCPTraceManager(grammar_load)
        trace2.load_traces_from_file("/tmp/reducts.p")
        trace2.serialize("/tmp/reducts2.p")

        for e1, e2 in zip(open("/tmp/reducts.p", "r"), open("/tmp/reducts2.p", "r")):
            self.assertEqual(e1, e2)

    def test_basic_split_merge(self):
        tree = hybrid_tree_1()
        tree2 = hybrid_tree_2()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

        (_, grammar) = induce_grammar([tree, tree2],
                                      the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                      terminal_labeling.token_label, [cfg], 'START')

        for rule in grammar.rules():
            print >>stderr, rule

        print >>stderr, "call S/M Training"

        new_grammars = split_merge_training(grammar, [tree, tree2], 3, 5, merge_threshold=0.5, debug=False)

        for new_grammar in new_grammars:
            for i, rule in enumerate(new_grammar.rules()):
                print >>stderr, i, rule
            print >> stderr

        print >>stderr, "finished S/M Training"


    def test_corpus_split_merge_training(self):
        train = '../../res/dependency_conll/german/tiger/train/german_tiger_train.conll'
        limit_train = 100
        test = train
        # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
        trees = parse_conll_corpus(train, False, limit_train)
        primary_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
        term_labelling = the_terminal_labeling_factory().get_strategy('pos')
        start = 'START'
        recursive_partitioning = [cfg]

        (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)

        # for rule in grammar.rules():
        #    print >>stderr, rule

        trees = parse_conll_corpus(train, False, limit_train)
        print >> stderr, "call S/M Training"

        new_grammars = split_merge_training(grammar_prim, trees, 4, 10, tie_breaking=True, init="equal", sigma=0.05, seed=50, merge_threshold=0.1)

        print >> stderr, "finished S/M Training"

        for new_grammar in new_grammars:
            for i, rule in enumerate(new_grammar.rules()):
                print >>stderr, i, rule
            print >>stderr

if __name__ == '__main__':
    unittest.main()
