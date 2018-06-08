import unittest
import corpora.negra_parse as np
import corpora.tiger_parse as tp
from hybridtree.general_hybrid_tree import HybridDag
from hybridtree.monadic_tokens import construct_constituent_token
from constituent.dag_induction import direct_extract_lcfrs_from_prebinarized_corpus, top, bottom
from parser.naive.parsing import LCFRS_parser
from parser.discodop_parser.parser import DiscodopKbestParser
from parser.sDCPevaluation.evaluator import DCP_evaluator, dcp_to_hybriddag
import tempfile
import copy
import subprocess
import os
from grammar.lcfrs import LCFRS


class MyTestCase(unittest.TestCase):
    def test_negra_to_dag_parsing(self):
        pass
        names = list(map(str, [26954]))

        fd_, primary_file = tempfile.mkstemp(suffix='.export')
        with open(primary_file, mode='w') as pf:

            for s in names:
                dsg = tp.sentence_names_to_deep_syntax_graphs([s], "res/tiger/tiger_s%s.xml" % s, hold=False,
                                                              ignore_puntcuation=False)[0]
                dsg.set_label(dsg.label[1:])
                lines = np.serialize_hybrid_dag_to_negra([dsg], 0, 500, use_sentence_names=True)
                print(''.join(lines), file=pf)

        _, binarized_file = tempfile.mkstemp(suffix='.export')
        subprocess.call(["discodop", "treetransforms", "--binarize", "-v", "1", "-h", "1", primary_file, binarized_file])

        print(primary_file)
        print(binarized_file)

        corpus = np.sentence_names_to_hybridtrees(names, primary_file, secedge=True)
        corpus2 = np.sentence_names_to_hybridtrees(names, binarized_file, secedge=True)
        dag = corpus[0]
        print(dag)
        assert isinstance(dag, HybridDag)
        self.assertEqual(8, len(dag.token_yield()))
        for token in dag.token_yield():
            print(token.form() + '/' + token.pos(), end=' ')
        print()

        dag_bin = corpus2[0]
        print(dag_bin)

        for token in dag_bin.token_yield():
            print(token.form() + '/' + token.pos(), end=' ')
        print()
        self.assertEqual(8, len(dag_bin.token_yield()))

        for node, token in zip(dag_bin.nodes(), list(map(str, map(dag_bin.node_token, dag_bin.nodes())))):
            print(node, token)

        print()
        print(top(dag_bin, {'500', '101', '102'}))
        self.assertSetEqual({'101', '500'}, top(dag_bin, {'500', '101', '102'}))
        print(bottom(dag_bin, {'500', '101', '102'}))
        self.assertSetEqual({'502'}, bottom(dag_bin, {'500', '101', '102'}))
        grammar = direct_extract_lcfrs_from_prebinarized_corpus(dag_bin)
        print(grammar)

        parser = LCFRS_parser(grammar)

        poss = list(map(lambda x: x.pos(), dag_bin.token_yield()))
        print(poss)
        parser.set_input(poss)

        parser.parse()

        self.assertTrue(parser.recognized())

        der = parser.best_derivation_tree()
        print(der)

        dcp_term = DCP_evaluator(der).getEvaluation()

        print(dcp_term[0])

        dag_eval = HybridDag(dag_bin.sent_label())
        dcp_to_hybriddag(dag_eval, dcp_term, copy.deepcopy(dag_bin.token_yield()), False, construct_token=construct_constituent_token)

        print(dag_eval)
        for node in dag_eval.nodes():
            token = dag_eval.node_token(node)
            if token.type() == "CONSTITUENT-CATEGORY":
                label = token.category()
            elif token.type() == "CONSTITUENT-TERMINAL":
                label = token.form(), token.pos()

            print(node, label, dag_eval.children(node), dag_eval.sec_children(node), dag_eval.sec_parents(node))

        lines = np.serialize_hybridtrees_to_negra([dag_eval], 1, 500, use_sentence_names=True)
        for line in lines:
            print(line, end='')

        print()

        with open(primary_file) as pcf:
            for line in pcf:
                print(line, end='')

    def test_negra_dag_small_grammar(self):
        DAG_CORPUS = 'res/tiger/tiger_full_with_sec_edges.export'
        DAG_CORPUS_BIN = 'res/tiger/tiger_full_with_sec_edges_bin_h1_v1.export'
        names = list([str(i) for i in range(1, 101)])
        if not os.path.exists(DAG_CORPUS):
            print('run the following command to create an export corpus with dags:')
            print('\tPYTHONPATH=. util/tiger_dags_to_negra.py ' +
                  'res/tiger/tiger_release_aug07.corrected.16012013.xml '
                   + DAG_CORPUS + ' 1 50474')
        self.assertTrue(os.path.exists(DAG_CORPUS))

        if not os.path.exists(DAG_CORPUS_BIN):
            print('run the following command to binarize the export corpus with dags:')
            print("discodop treetransforms --binarize -v 1 -h 1 " + DAG_CORPUS + " " + DAG_CORPUS_BIN)
            # _, DAG_CORPUS_BIN = tempfile.mkstemp(prefix='corpus_bin_', suffix='.export')
            # subprocess.call(["discodop", "treetransforms", "--binarize", "-v", "1", "-h", "1", DAG_CORPUS, DAG_CORPUS_BIN])
        self.assertTrue(os.path.exists(DAG_CORPUS_BIN))
        corpus = np.sentence_names_to_hybridtrees(names, DAG_CORPUS, secedge=True)
        corpus_bin = np.sentence_names_to_hybridtrees(names, DAG_CORPUS_BIN, secedge=True)

        grammar = LCFRS(start="START")

        for hybrid_dag, hybrid_dag_bin in zip(corpus, corpus_bin):
            self.assertEqual(len(hybrid_dag.token_yield()), len(hybrid_dag_bin.token_yield()))

            dag_grammar = direct_extract_lcfrs_from_prebinarized_corpus(hybrid_dag_bin)
            grammar.add_gram(dag_grammar)

        grammar.make_proper()
        print("Extracted LCFRS/DCP-hybrid grammar with %i nonterminals and %i rules"
              % (len(grammar.nonts()), len(grammar.rules())))

        parser = DiscodopKbestParser(grammar, k=1)

        _, RESULT_FILE = tempfile.mkstemp(prefix='parser_results_', suffix='.export')

        with open(RESULT_FILE, 'w') as results:
            for hybrid_dag in corpus:

                poss = list(map(lambda x: x.pos(), hybrid_dag.token_yield()))
                parser.set_input(poss)
                parser.parse()
                self.assertTrue(parser.recognized())
                der = parser.best_derivation_tree()

                dcp_term = DCP_evaluator(der).getEvaluation()
                dag_eval = HybridDag(hybrid_dag.sent_label())
                dcp_to_hybriddag(dag_eval, dcp_term, copy.deepcopy(hybrid_dag.token_yield()), False,
                                 construct_token=construct_constituent_token)
                lines = np.serialize_hybridtrees_to_negra([dag_eval], 1, 500, use_sentence_names=True)
                for line in lines:
                    print(line, end='', file=results)
                parser.clear()

        print("Wrote results to %s" % RESULT_FILE)


if __name__ == '__main__':
    unittest.main()
