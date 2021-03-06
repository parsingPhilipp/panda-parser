# -*- coding: iso-8859-15 -*-
from __future__ import print_function
import grammar.induction.recursive_partitioning
import grammar.induction.terminal_labeling

__author__ = 'kilian'

import unittest
import copy
import dependency.induction as d_i
from dependency.labeling import the_labeling_factory
from grammar.induction.terminal_labeling import the_terminal_labeling_factory
from grammar.induction.recursive_partitioning import direct_extraction
from hybridtree.monadic_tokens import construct_conll_token
from hybridtree.dependency_tree import disconnect_punctuation, HybridTree
from parser.naive.parsing import LCFRS_parser
from corpora.conll_parse import *
import subprocess
import os
import re

TEST_FILE = 'res/tests/Dependency_Corpus.conll'
TEST_FILE_MODIFIED = 'res/tests/Dependency_Corpus_modified.conll'
SLOVENE = 'res/tests/slovene_multi_root.conll'

CONLL_TEST = 'res/dependency_conll/german/tiger/test/german_tiger_test.conll'
CONLL_TRAIN = 'res/dependency_conll/german/tiger/train/german_tiger_train.conll'

HYPOTHESIS_PREFIX = '.tmp/sys-output'
EVAL_PL = 'util/eval.pl'
TEST_FILE_PATH = '/tmp/no-punctuation.conll'
TEST_FILE_PATH_LE_20 = '/tmp/sent-leq-20.conll'

GLOBAL_SAMPLE = """1       Viele   _       PIAT    PIAT    _       4       NK      4       NK
2       G�ttinger       _       ADJA    ADJA    _       4       NK      4       NK
3       ``      _       $(      $(      _       4       PUNC    4       PUNC
4       Autonome        _       NN      NN      _       6       SB      6       SB
5       ''      _       $(      $(      _       6       PUNC    6       PUNC
6       laufen  _       VVFIN   VVFIN   _       0       ROOT    0       ROOT
7       zur     _       APPRART APPRART _       6       MO      6       MO
8       Zeit    _       NN      NN      _       7       NK      7       NK
9       mit     _       APPR    APPR    _       6       MO      6       MO
10      einem   _       ART     ART     _       9       NK      9       NK
11      unguten _       ADJA    ADJA    _       9       NK      9       NK
12      Gef�hl  _       NN      NN      _       9       NK      9       NK
13      durch   _       APPR    APPR    _       6       MO      6       MO
14      die     _       ART     ART     _       13      NK      13      NK
15      Stadt   _       NN      NN      _       13      NK      13      NK
16      .       _       $.      $.      _       6       PUNC    6       PUNC"""


class CoNLLParserTest(unittest.TestCase):
    @staticmethod
    def test_conll_grammar_induction():
        ignore_punctuation = True
        trees = parse_conll_corpus(TEST_FILE, False)
        trees = disconnect_punctuation(trees)
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')
        nonterminal_labeling = the_labeling_factory().create_simple_labeling_strategy('child', 'pos')
        (_, grammar) = d_i.induce_grammar(trees, nonterminal_labeling, terminal_labeling.token_label,
                                          [direct_extraction], 'START')

        trees2 = parse_conll_corpus(TEST_FILE_MODIFIED, False)
        trees2 = disconnect_punctuation(trees2)

        for tree in trees2:
            parser = LCFRS_parser(grammar, terminal_labeling.prepare_parser_input(tree.token_yield()))
            cleaned_tokens = copy.deepcopy(tree.full_token_yield())
            for token in cleaned_tokens:
                token.set_edge_label('_')
            h_tree = HybridTree()
            h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, cleaned_tokens, ignore_punctuation,
                                                            construct_conll_token)
            # print h_tree
            print('input -> hybrid-tree -> output')
            print(tree_to_conll_str(tree))
            print('parsed tokens')
            print(list(map(str, h_tree.full_token_yield())))
            print('test_parser output')
            print(tree_to_conll_str(h_tree))

    def test_multi_root_parsing(self):
        trees = parse_conll_corpus(SLOVENE, False)

        counter = 0

        for tree in trees:
            print(tree_to_conll_str(tree))
            print(tree)
            counter += 1

        self.assertEqual(counter, 1)

    def test_conll_parse(self):
        trees = parse_conll_corpus(CONLL_TEST, True)
        test_trees = parse_conll_corpus(CONLL_TEST, True)

        # for i in range (len(trees)):
        # if i < len(test_trees):
        #         print compare_dependency_trees(trees[i], test_trees[i])
        #         print score_cmp_dep_trees(trees[i], test_trees[i])
        try:
            while True:
                t1 = next(trees)
                t2 = next(test_trees)
                self.assertEqual(t1.sent_label(), t2.sent_label())
                # tuple2 = score_cmp_dep_trees(t1, t2)
                # self.assertEqual(tuple2[0], tuple2)
                tuple1 = compare_dependency_trees(t1, t2)
                self.assertEqual(tuple1[0], tuple1[1])
                self.assertEqual(tuple1[2], 1)
                self.assertEqual(tuple1[3], 1)
                self.assertEqual(tuple1[4], tuple1[0])
        except StopIteration:
            pass

            # print score_cmp_dep_trees(trees[i], test_trees[i])
            # print tree
            # print tree_to_conll_str(tree), '\n '
            # print node_to_conll_str(trees[0], trees[0].root())

            # print tree_to_conll_str(trees[0])

    def test_conll_generation(self):
        test_trees = disconnect_punctuation(parse_conll_corpus(CONLL_TEST, True))
        CoNLL_strings = []
        for tree in test_trees:
            CoNLL_strings.append(tree_to_conll_str(tree))

        CoNLL_strings.append('')

        # Remove file if exists
        try:
            os.remove(TEST_FILE_PATH)
        except OSError:
            pass

        test_file = open(TEST_FILE_PATH, 'a+')
        test_file.write('\n\n'.join(CoNLL_strings))
        test_file.close()

        eval_pl_call_strings = ["-g {!s}".format(CONLL_TEST), "-s {!s}".format(TEST_FILE_PATH), ""]
        # print(eval_pl_call_strings)
        p = subprocess.Popen(['perl', EVAL_PL] + eval_pl_call_strings, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        self.assertEqual(p.returncode, 0)

        out = str(out, encoding='utf8')
        lines = str(out).splitlines()
        # for line in lines:
        #     print(line)
        uas = 0.0
        las = 0.0
        la = 0.0
        for line in lines:
            m = re.search(r'^\s*Labeled\s*attachment\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
            if m:
                las = float(m.group(1)) / 100
            m = re.search(r'^\s*Unlabeled\s*attachment\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
            if m:
                uas = float(m.group(1)) / 100
            m = re.search(r'^\s*Label\s*accuracy\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
            if m:
                la = float(m.group(1)) / 100

        # print(uas, las, la)
        self.assertEqual(1.0, uas)
        self.assertEqual(1.0, las)
        self.assertEqual(1.0, la)

    def test_conll_generation_2(self):
        test_trees = parse_conll_corpus(CONLL_TEST, True)
        dis = disconnect_punctuation(parse_conll_corpus(CONLL_TEST, True))
        dis_labs = []
        CoNLL_strings = []
        for tree in dis:
            if len(tree.token_yield()) <= 20:
                dis_labs.append(tree.sent_label())

        for tree in test_trees:
            if tree.sent_label() in dis_labs:
                CoNLL_strings.append(tree_to_conll_str(tree))

        CoNLL_strings.append('')

        # Remove file if exists
        try:
            os.remove(TEST_FILE_PATH_LE_20)
        except OSError:
            pass

        test_file = open(TEST_FILE_PATH_LE_20, 'a+')
        test_file.write('\n\n'.join(CoNLL_strings))
        test_file.close()

    def test_compare_rec_par(self):
        test_trees = disconnect_punctuation(parse_conll_corpus(CONLL_TRAIN, True))
        mylist = []
        factory = grammar.induction.recursive_partitioning.the_recursive_partitioning_factory()
        [leftb, cfg] = factory.get_partitioning('left-branching,fanout-1')
        i = 0
        for tree in test_trees:
            i = i + 1
            if leftb(tree) == cfg(tree):
                mylist.append(tree.sent_label())
            if i == 1000:
                break
        print(mylist)


if __name__ == '__main__':
    unittest.main()
