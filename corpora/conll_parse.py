#!/usr/bin/python2.7
#  -*- coding: iso-8859-15 -*-
__author__ = 'kilian'

import re
import sys

from hybridtree.general_hybrid_tree import GeneralHybridTree
from hybridtree.biranked_tokens import construct_dependency_token, CoNLLToken
import dependency.induction as d_i
import dependency.labeling as label
from parser.naive.parsing import LCFRS_parser


test_file = 'examples/Dependency_Corpus.conll'
test_file_modified = 'examples/Dependency_Corpus_modified.conll'

conll_test = '../dependency_conll/german/tiger/test/german_tiger_test.conll'
conll_train = '../dependency_conll/german/tiger/train/german_tiger_train.conll'


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
    match = re.search(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)'
                      r'\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)$', line)
    return match


def parse_conll_corpus(path, ignore_punctuation, limit=sys.maxint):
    """
    :param path: path to corpus
    :type: str
    :param ignore_punctuation: exclude punctuation from tree structure
    :type ignore_punctuation: bool
    :param limit: stop generation after limit trees
    :type: int
    :return: a series of hybrid trees read from file
    :rtype: __generator[GeneralHybridTree]
    :raise Exception: unexpected input in corpus file
    Lazily parses a dependency corpus (in CoNLL format) and generates GeneralHybridTrees.
    """

    # print path
    file_content = open(path)
    tree_count = 0

    while tree_count < limit:
        tree = None

        try:
            line = file_content.next()
        except StopIteration:
            break

        match = match_line(line)
        while match:
            if match.group(1) == '1':
                tree_count += 1
                tree = GeneralHybridTree('tree' + str(tree_count))
                root = 0

            node_id = match.group(1)
            label = match.group(2)
            pos = match.group(4)
            parent = match.group(7)
            deprel = match.group(8)

            if not ignore_punctuation or (not re.search(r'^\$.*$', pos)):
                tree.add_node(node_id, CoNLLToken(label, None, pos, deprel), True, True)
                tree.add_child(parent, node_id)
            else:
                tree.add_node(node_id, CoNLLToken(label, None, pos, deprel), True, False)

            if parent == '0':
                tree.set_root(node_id)
                root += 1

            try:
                line = file_content.next()
                match = match_line(line)
            except StopIteration:
                line = ''
                match = None

        # Assume empty line, otherwise raise exception
        match = re.search(r'^[^\s]*$', line)
        if not match:
            raise Exception("Unexpected input in CoNLL corpus file.")

        if tree:
            # basic sanity checks
            if not tree.root:
                # FIXME: ignoring punctuation may leads to malformed trees
                print "non-rooted"
                if ignore_punctuation:
                    continue
                raise Exception
            elif root > 1:
                # FIXME: turkish corpus contains trees with more than one root
                # FIXME: currently, they are ignored
                continue
            elif tree.n_nodes() != len(tree.id_yield()) or len(tree.nodes()) != len(tree.full_yield()):
                # FIXME: ignoring punctuation may leads to malformed trees
                if ignore_punctuation:
                    continue
                raise Exception('{4}: connected nodes: {0}, total nodes: {1}, full yield: {2}, connected yield: {3}'.format(
                     str(tree.n_nodes()), str(len(tree.nodes())), str(len(tree.full_yield())),
                     str(len(tree.id_yield()))), tree.sent_label())
            yield tree

def tree_to_conll_str(tree):
    """
    :param tree: hybrid tree
    :type tree: GeneralHybridTree
    :return: ConLL format of tree!
    :rtype: str
    Output a hybrid tree, that models the dependency structure of some sentence, in CoNLL format.
    """

    s = '\n'.join([node_to_conll_str(tree, id) for id in tree.full_yield()])
    return s


def node_to_conll_str(tree, id):
    """
    :type tree: GeneralHybridTree
    :type id:   str
    :param id: node id
    :return: line for this tree node in CoNLL format
    :rtype: str
    """
    delimiter = '\t'
    s = ''
    s += str(tree.node_index_full(id) + 1) + delimiter
    s += tree.node_token(id) + delimiter
    s += '_' + delimiter
    s += tree.node_token(id).pos() + delimiter
    s += tree.node_token(id).pos() + delimiter
    s += '_' + delimiter
    dependency_info = ''
    if tree.root == id:
        dependency_info += '0' + delimiter
    # Connect disconnected tokens (i.e. punctuation) to the root.
    elif tree.disconnected(id):
        dependency_info += str(tree.node_index_full(tree.root) + 1) + delimiter
    else:
        dependency_info += str(tree.node_index_full(tree.parent(id)) + 1) + delimiter
    if tree.disconnected(id):
        dependency_info += 'PUNC'
    else:
        dependency_info += tree.node_token(id).dep()
    s += dependency_info + delimiter + dependency_info
    return s


def compare_dependency_trees(reference, test):
    """
    :param reference: GeneralHybridTree
    :param test: GeneralHybridTree
    :return: 5-tuple of int :raise Exception:
    :rtype: int,int,int,int,int
    Compute UAS, LAS, UEM, LEM, length (of front) for the parsed dependency tree, given some reference tree.
    """
    UAS = 0
    LAS = 0
    UEM = 0
    LEM = 0

    # sanity check
    if reference.token_yield() != test.token_yield():
        raise Exception("yield of trees differs: \'{0}\' vs. \'{1}\'".format(' '.join(reference.token_yield()),
                                                                             ' '.join(test.token_yield())))

    for i in range(1, len(reference.token_yield()) + 1):
        ref_id = reference.index_node(i)
        test_id = test.index_node(i)
        if reference.root == ref_id:
            if test.root == test_id:
                UAS += 1
                if reference.node_dep_label(ref_id) == test.node_dep_label(test_id):
                    LAS += 1
        elif test.root != test_id:
            ref_parent_i = reference.node_index(reference.parent(ref_id))
            test_parent_i = test.node_index(test.parent(test_id))
            if ref_parent_i == test_parent_i:
                UAS += 1
                if reference.node_dep_label(ref_id) == test.node_dep_label(test_id):
                    LAS += 1

    if reference.n_nodes() == UAS:
        UEM = 1
        if reference.n_nodes() == LAS:
            LEM = 1

    return UAS, LAS, UEM, LEM, reference.n_nodes()


def score_cmp_dep_trees(reference, test):
    """
    :param reference: GeneralHybridTree
    :param test: GeneralHybridTree
    :rtype: float,float,float,float
    :raise Exception:
    Compute UAS, LAS, UEM, LEM for the parsed dependency tree, given some reference tree,
    normalized to length of front.
    """
    (UAS, LAS, UEM, LEM, length) = compare_dependency_trees(reference, test)
    return UAS * 1.0 / length, LAS * 1.0 / length, UEM, LEM


def test_conll_parse():
    trees = parse_conll_corpus(conll_test, True)
    test_trees = parse_conll_corpus(conll_test, True)

    # for i in range (len(trees)):
    #     if i < len(test_trees):
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


def test_conll_grammar_induction():
    trees = parse_conll_corpus(test_file, True)
    (_, grammar) = d_i.induce_grammar(trees, label.ChildFormLabeling(), d_i.term_pos, d_i.direct_extraction, 'START')

    trees2 = parse_conll_corpus(test_file, True)

    for tree in trees2:
        parser = LCFRS_parser(grammar, [token.pos() for token in tree.token_yield()])
        h_tree = GeneralHybridTree()
        h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, [token.pos() for token in tree.full_token_yield()], [token.form() for token in tree.full_token_yield()], True, construct_dependency_token)
        #print h_tree
        print h_tree.full_token_yield()
        print tree_to_conll_str(h_tree)



# test_conll_grammar_induction()
# test_conll_parse()
