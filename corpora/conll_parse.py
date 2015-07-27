#!/usr/bin/python2.7
# -*- coding: iso-8859-15 -*-
__author__ = 'kilian'

import re
import sys

from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import CoNLLToken


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
    :rtype: __generator[HybridTree]
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
                tree = HybridTree('tree' + str(tree_count))

            node_id = match.group(1)
            form = match.group(2)
            lemma = match.group(3)
            pos = match.group(4)
            fine_grained_pos = match.group(5)
            feats = match.group(6)
            parent = match.group(7)
            deprel = match.group(8)

            # TODO: Create a definition-conform punctuation filter
            # cf. http://ilk.uvt.nl/conll/software.html#eval
            # How do you exclude tokens from scoring?
            if not ignore_punctuation or (not re.search(r'^\$.*$', pos)):
                tree.add_node(node_id, CoNLLToken(form, lemma, pos, fine_grained_pos, feats, deprel), True, True)
                if parent != '0':
                    tree.add_child(parent, node_id)
            else:
                tree.add_node(node_id, CoNLLToken(form, lemma, pos, fine_grained_pos, feats, deprel), True, False)

            # TODO: If punctuation is ignored and the root is punctuation,
            # TODO: it is added to the tree anyhow.
            if parent == '0':
                tree.add_to_root(node_id)

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
                # elif root > 1:
                # FIXME: turkish corpus contains trees with more than one root
                # FIXME: currently, they are ignored
                # continue
            elif tree.n_nodes() != len(tree.id_yield()) or len(tree.nodes()) != len(tree.full_yield()):
                # FIXME: ignoring punctuation may leads to malformed trees
                if ignore_punctuation:
                    continue
                raise Exception(
                    '{4}: connected nodes: {0}, total nodes: {1}, full yield: {2}, connected yield: {3}'.format(
                        str(tree.n_nodes()), str(len(tree.nodes())), str(len(tree.full_yield())),
                        str(len(tree.id_yield()))), tree.sent_label())
            yield tree


def tree_to_conll_str(tree):
    """
    :param tree: hybrid tree
    :type tree: HybridTree
    :return: ConLL format of tree!
    :rtype: str
    Output a hybrid tree, that models the dependency structure of some sentence, in CoNLL format.
    """
    s = '\n'.join([node_to_conll_str(tree, id) for id in tree.full_yield()])
    return s


def node_to_conll_str(tree, id):
    """
    :type tree: HybridTree
    :type id:   str
    :param id: node id
    :return: line for this tree node in CoNLL format
    :rtype: str
    """
    token = tree.node_token(id)
    delimiter = '\t'
    s = ''
    s += str(tree.node_index_full(id) + 1) + delimiter
    s += token.form() + delimiter
    s += token.lemma() + delimiter
    s += token.cpos() + delimiter
    s += token.pos() + delimiter
    s += token.feats() + delimiter
    dependency_info = ''
    if id in tree.root:
        dependency_info += '0' + delimiter
    # Connect disconnected tokens (i.e. punctuation) to the root.
    elif tree.disconnected(id):
        dependency_info += str(tree.node_index_full(tree.root[0]) + 1) + delimiter
    else:
        dependency_info += str(tree.node_index_full(tree.parent(id)) + 1) + delimiter
    dependency_info += token.deprel()
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
    if not [token.form() for token in reference.token_yield()].__eq__([token.form() for token in test.token_yield()]):
        raise Exception("yield of trees differs: \'{0}\' vs. \'{1}\'".format(
            ' '.join([token.form() for token in reference.token_yield()])),
            ' '.join([token.form() for token in test.token_yield()]))

    for i in range(1, len(reference.token_yield()) + 1):
        ref_id = reference.index_node(i)
        test_id = test.index_node(i)
        if ref_id in reference.root:
            if test_id in test.root:
                UAS += 1
                if reference.node_token(ref_id).deprel() == test.node_token(test_id).deprel():
                    LAS += 1
        elif test_id not in test.root:
            ref_parent_i = reference.node_index(reference.parent(ref_id))
            test_parent_i = test.node_index(test.parent(test_id))
            if ref_parent_i == test_parent_i:
                UAS += 1
                if reference.node_token(ref_id).deprel() == test.node_token(test_id).deprel():
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
