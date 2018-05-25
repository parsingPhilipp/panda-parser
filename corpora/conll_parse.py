"""Parsing/Serialization between the CoNLL dependency tree format and HybridTrees."""
__author__ = 'kilian'

import re
import sys

from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import CoNLLToken

CONLL_LINE = re.compile(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)'
                        r'\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)$')
MULTI_TOKEN = re.compile(r'^[^\s]+-[^\s]+')
EMPTY_LINE = re.compile(r'^[^\s]*$')

DELETE_PUNCTUATION = str.maketrans("", "", '!"&()*+#,/-:.?;<=>@[\\]^_{|}~')


def is_punctuation(form):
    # this is string.punctuation with $, % removed (which are PMOD, NMOD, COORD, NMOD with dependents in WSJ)
    return not str(form).translate(DELETE_PUNCTUATION)
    # we allow the dollar sign $ and the quotation marks `` and ''


def parse_conll_corpus(path, ignore_punctuation, limit=sys.maxsize, start=0):
    """
    :param path: path to corpus
    :type: str
    :param ignore_punctuation: exclude punctuation from tree structure
    :type ignore_punctuation: bool
    :param limit: stop generation after limit trees
    :type: int
    :param start: start generation with start'th tree
    :type start: int
    :return: a series of hybrid trees read from file
    :rtype: __generator[HybridTree]
    :raise Exception: unexpected input in corpus file
    Lazily parses a dependency corpus (in CoNLL format) and generates GeneralHybridTrees.
    """

    # print path
    with open(path) as file_content:
        tree_count = 0

        while tree_count < limit:
            tree = None

            try:
                line = next(file_content)
                while line.startswith('#'):
                    line = next(file_content)
            except StopIteration:
                break

            match = CONLL_LINE.match(line)
            while match:
                if match.group(1) == '1':
                    tree_count += 1
                    tree = HybridTree('tree' + str(tree_count))

                node_id = match.group(1)
                form = match.group(2)
                lemma = match.group(3)
                cpos = match.group(4)
                pos = match.group(5)
                feats = match.group(6)
                parent = match.group(7)
                deprel = match.group(8)

                # We ignore information about multiple token's as present in the UD version of Prague Dep. TB
                if MULTI_TOKEN.search(node_id):
                  pass
                else:
                    # If punctuation is to be ignored, we
                    # remove it from the hybrid tree
                    # Punctuation according to definition
                    # cf. http://ilk.uvt.nl/conll/software.html#eval

                    # if not ignore_punctuation or form.translate(no_translation, string.punctuation):
                    tree.add_node(node_id, CoNLLToken(form, lemma, cpos, pos, feats, deprel), True, True)
                    if parent != '0':
                        tree.add_child(parent, node_id)
                    # else:
                    #    tree.add_node(node_id, CoNLLToken(form, lemma, pos, fine_grained_pos, feats, deprel), True, False)

                    # TODO: If punctuation is ignored and the root is punctuation,
                    # TODO: it is added to the tree anyhow.
                    if parent == '0':
                        tree.add_to_root(node_id)

                try:
                    line = next(file_content)
                    while line.startswith('#'):
                        line = next(file_content)
                    match = CONLL_LINE.search(line)
                except StopIteration:
                    line = ''
                    match = None

            # Assume empty line, otherwise raise exception
            match = EMPTY_LINE.match(line)
            if not match:
                raise Exception("Unexpected input in CoNLL corpus file.")

            if tree:
                # basic sanity checks
                if not tree.root:
                    # FIXME: ignoring punctuation may leads to malformed trees
                    print("non-rooted")
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
                            str(len(tree.id_yield())), tree.sent_label()))
                if tree_count > start:
                    yield tree


def tree_to_conll_str(tree):
    """
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
    line = []
    delimiter = '\t'
    line.append(str(tree.node_index_full(id) + 1))
    line.append(token.form())
    # TODO the database does not store these fields of a CoNLLToken yet,
    # TODO but eval.pl rejects to compare two tokens if they differ
    # TODO extend the database, then fix this
    line.append(token.lemma())
    line.append(token.cpos())
    line.append(token.pos())
    line.append(token.feats())

    if id in tree.root:
        line.append('0')
    # Connect disconnected tokens (i.e. punctuation) to the root.
    elif tree.disconnected(id):
        line.append(str(tree.node_index_full(tree.root[0]) + 1))
    else:
        line.append(str(tree.node_index_full(tree.parent(id)) + 1))
    line.append(token.deprel())
    return delimiter.join(line)


def compare_dependency_trees(reference, test):
    """
    :type reference: HybridTree
    :type test: HybridTree
    :return: 5-tuple of int
    :raise Exception:
    :rtype: int,int,int,int,int
    Compute UAS, LAS, UEM, LEM, length (of front) for the parsed dependency tree, given some reference tree.
    """

    uas = 0
    las = 0
    uem = 0
    lem = 0

    # sanity check
    if not [token.form() for token in reference.token_yield()].__eq__([token.form() for token in test.token_yield()]):
        raise Exception("yield of trees differs: \'{0}\' vs. \'{1}\'".format(
            ' '.join([token.form() for token in reference.token_yield()]),
            ' '.join([token.form() for token in test.token_yield()])))

    for i in range(1, len(reference.token_yield()) + 1):
        ref_id = reference.index_node(i)
        test_id = test.index_node(i)
        if ref_id in reference.root:
            if test_id in test.root:
                uas += 1
                if reference.node_token(ref_id).deprel() == test.node_token(test_id).deprel():
                    las += 1
        elif test_id not in test.root:
            ref_parent_i = reference.node_index(reference.parent(ref_id))
            test_parent_i = test.node_index(test.parent(test_id))
            if ref_parent_i == test_parent_i:
                uas += 1
                if reference.node_token(ref_id).deprel() == test.node_token(test_id).deprel():
                    las += 1

    if reference.n_nodes() == uas:
        uem = 1
        if reference.n_nodes() == las:
            lem = 1

    return uas, las, uem, lem, reference.n_nodes()


def score_cmp_dep_trees(reference, test):
    """
    :type reference: HybridTree
    :type test: HybridTree
    :rtype: float,float,float,float
    :raise Exception:
    Compute UAS, LAS, UEM, LEM for the parsed dependency tree, given some reference tree,
    normalized to length of front.
    """
    (uas, las, uem, lem, length) = compare_dependency_trees(reference, test)
    return uas * 1.0 / length, las * 1.0 / length, uem, lem


__all__ = ["parse_conll_corpus", "tree_to_conll_str", "score_cmp_dep_trees", "compare_dependency_trees"]
