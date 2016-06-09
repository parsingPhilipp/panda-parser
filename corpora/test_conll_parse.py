# -*- coding: iso-8859-15 -*-
__author__ = 'kilian'

import unittest
import copy
import dependency.induction as d_i
from dependency.labeling import the_labeling_factory
from hybridtree.monadic_tokens import construct_conll_token
from hybridtree.dependency_tree import disconnect_punctuation
from parser.naive.parsing import LCFRS_parser
from corpora.conll_parse import *
import subprocess32 as subprocess
import os

test_file = '../examples/Dependency_Corpus.conll'
test_file_modified = '../examples/Dependency_Corpus_modified.conll'
slovene = 'slovene_multi_root.conll'

conll_test = '../../dependency_conll/german/tiger/test/german_tiger_test.conll'
conll_train = '../../dependency_conll/german/tiger/train/german_tiger_train.conll'

hypothesis_prefix = '.tmp/sys-output'
eval_pl = '../util/eval.pl'
test_file_path = '/tmp/no-punctuation.conll'

global_s = """1       Viele   _       PIAT    PIAT    _       4       NK      4       NK
2       Göttinger       _       ADJA    ADJA    _       4       NK      4       NK
3       ``      _       $(      $(      _       4       PUNC    4       PUNC
4       Autonome        _       NN      NN      _       6       SB      6       SB
5       ''      _       $(      $(      _       6       PUNC    6       PUNC
6       laufen  _       VVFIN   VVFIN   _       0       ROOT    0       ROOT
7       zur     _       APPRART APPRART _       6       MO      6       MO
8       Zeit    _       NN      NN      _       7       NK      7       NK
9       mit     _       APPR    APPR    _       6       MO      6       MO
10      einem   _       ART     ART     _       9       NK      9       NK
11      unguten _       ADJA    ADJA    _       9       NK      9       NK
12      Gefühl  _       NN      NN      _       9       NK      9       NK
13      durch   _       APPR    APPR    _       6       MO      6       MO
14      die     _       ART     ART     _       13      NK      13      NK
15      Stadt   _       NN      NN      _       13      NK      13      NK
16      .       _       $.      $.      _       6       PUNC    6       PUNC"""


class CoNLLParserTest(unittest.TestCase):
    def test_conll_grammar_induction(self):
        trees = parse_conll_corpus(test_file, True)
        terminal_labeling = d_i.the_terminal_labeling_factory().get_strategy('pos')
        nonterminal_labeling = the_labeling_factory().create_simple_labeling_strategy('child', 'pos')
        (_, grammar) = d_i.induce_grammar(trees, nonterminal_labeling, terminal_labeling.token_label,
                                          d_i.direct_extraction, 'START')

        trees2 = parse_conll_corpus(test_file_modified, True)

        for tree in trees2:
            parser = LCFRS_parser(grammar, terminal_labeling.prepare_parser_input(tree.token_yield()))
            cleaned_tokens = copy.deepcopy(tree.full_token_yield())
            for token in cleaned_tokens:
                token.set_deprel('_')
            h_tree = HybridTree()
            h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, cleaned_tokens, True,
                                                            construct_conll_token)
            # print h_tree
            print 'input -> hybrid-tree -> output'
            print tree_to_conll_str(tree)
            print 'parsed tokens'
            print map(str, h_tree.full_token_yield())
            print 'parser output'
            print tree_to_conll_str(h_tree)

    def test_multi_root_parsing(self):
        trees = parse_conll_corpus(slovene, False)

        counter = 0

        for tree in trees:
            print tree_to_conll_str(tree)
            print tree
            counter += 1

        self.assertEqual(counter, 1)

    def test_conll_parse(self):
        trees = parse_conll_corpus(conll_test, True)
        test_trees = parse_conll_corpus(conll_test, True)

        # for i in range (len(trees)):
        # if i < len(test_trees):
        #         print compare_dependency_trees(trees[i], test_trees[i])
        #         print score_cmp_dep_trees(trees[i], test_trees[i])
        try:
            while True:
                t1 = trees.next()
                t2 = test_trees.next()
                print t1.sent_label(), t2.sent_label()
                print compare_dependency_trees(t1, t2)
                print score_cmp_dep_trees(t1, t2)
                print compare_dependency_trees(t1, t1)
                print score_cmp_dep_trees(t1, t1)
        except StopIteration:
            pass

            # print score_cmp_dep_trees(trees[i], test_trees[i])
            # print tree
            # print tree_to_conll_str(tree), '\n '
            # print node_to_conll_str(trees[0], trees[0].root())

            # print tree_to_conll_str(trees[0])

    def test_conll_generation(self):
        test_trees = disconnect_punctuation(parse_conll_corpus(conll_test, True))
        CoNLL_strings = []
        for tree in test_trees:
            CoNLL_strings.append(tree_to_conll_str(tree))

        CoNLL_strings.append('')

        # Remove file if exists
        try:
            os.remove(test_file_path)
        except OSError:
            pass

        test_file = open(test_file_path, 'a+')
        test_file.write('\n\n'.join(CoNLL_strings))
        test_file.close()

        eval_pl_call_strings = ["-g {!s}".format(conll_test), "-s {!s}".format(test_file_path), ""]
        print eval_pl_call_strings
        p = subprocess.Popen(['perl', eval_pl] + eval_pl_call_strings, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        print out
        print err

        lines = out.split('\n')
        # print lines
        uas = 0.0
        las = 0.0
        la = 0.0
        # for line in lines:
        #     m = re.search(r'^\s*Labeled\s*attachment\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
        #     if m:
        #         las = float(m.group(1)) / 100
        #     m = re.search(r'^\s*Unlabeled\s*attachment\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
        #     if m:
        #         uas = float(m.group(1)) / 100
        #     m = re.search(r'^\s*Label\s*accuracy\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
        #     if m:
        #         la = float(m.group(1)) / 100

        print uas, las, la


if __name__ == '__main__':
    unittest.main()
