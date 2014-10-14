#!/usr/bin/python2.7
#  -*- coding: iso-8859-15 -*-
__author__ = 'kilian'

from general_hybrid_tree import GeneralHybridTree
import re


test_file = 'examples/Dependency_Corpus.conll'

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


def match_line(line):
    match = re.search(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)$', line)
    return match

# parses a conll file
# file: path to file
# return: list of GeneralHbyridTree
def parse_conll_corpus(file):
    file_content = open(file).readlines()

    trees = []

    i = 0;
    tree_count = 0

    while i < len(file_content):
        tree = None
        line = file_content[i]
        match = match_line(line)
        while match:
            if match.group(1) == '1':
                tree_count += 1
                tree = GeneralHybridTree('tree' + str(tree_count))

            id = match.group(1)
            label = match.group(2)
            pos = match.group(4)
            parent = match.group(7)
            deprel = match.group(8)

            tree.add_node(id, label, pos, True, True)
            tree.add_child(parent, id)

            if parent == '0':
                tree.set_root(id)

            if i < len(file_content):
                line = file_content[i]
                match = match_line(line)
                i += 1
            else:
                match = None

        # Assume empty line, otherwise raise exception
        match = re.search(r'^[^\s]*$', line)
        if not match:
            raise Exception

        if tree:
            # basic sanity checks
            if not tree.rooted():
                raise Exception
            elif tree.n_nodes() != len(tree.full_yield()):
                raise Exception

            trees.append(tree)
            # print tree
    return trees

def test_conll_parse():
    trees = parse_conll_corpus(test_file)
    for tree in trees:
        print tree
test_conll_parse()
