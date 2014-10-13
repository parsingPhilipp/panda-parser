#!/usr/bin/python2.7
#  -*- coding: iso-8859-15 -*-
__author__ = 'kilian'

from general_hybrid_tree import GeneralHybridTree
import re

def parse_word(s, grammar):
    #re.search
    return

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


def test_conll_parse():
    s = global_s
    ss = s.split('\n')
    tree = GeneralHybridTree("s1")
    tree.add_node("0", "ROOT", None, False, False)
    tree.set_root("0")

    for s in ss:
        match = re.search(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)$',s)
        if match:
            # s = match.group(11)
            id = match.group(1)
            label = match.group(2)
            pos = match.group(4)
            parent = match.group(7)
            deprel = match.group(8)
            tree.add_node(id, label, pos, True, True)
            tree.add_child(parent, id)
            # match = re.search(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)(.*$)',s)
            print "adding", label, "with", id,"to parent", parent
    print tree

test_conll_parse()