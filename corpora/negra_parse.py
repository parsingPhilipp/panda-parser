# Parsing of the Negra corpus and capture of hybrid trees.
from __future__ import print_function, unicode_literals
from os.path import expanduser
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.general_hybrid_tree import HybridDag
from hybridtree.monadic_tokens import ConstituentTerminal, ConstituentCategory
from graphs.dog import DeepSyntaxGraph, DirectedOrderedGraph
from grammar.lcfrs import *
import re
import codecs
from util.enumerator import Enumerator
try:
    from __builtin__ import str as text
except ImportError:
    text = str

# Location of Negra corpus.
NEGRA_DIRECTORY = 'res/negra-corpus/downloadv2'

# The non-projective and projective versions of the negra corpus.
NEGRA_NONPROJECTIVE = NEGRA_DIRECTORY + '/negra-corpus.export'
NEGRA_PROJECTIVE = NEGRA_DIRECTORY + '/negra-corpus.cfg'

DISCODOP_HEADER = re.compile(r'^%%\s+word\s+lemma\s+tag\s+morph\s+edge\s+parent\s+secedge$')
BOS = re.compile(r'^#BOS\s+([0-9]+)')
EOS = re.compile(r'^#EOS\s+([0-9]+)$')

STANDARD_NONTERMINAL = re.compile(r'^#([0-9]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([0-9]+)((\s+[^\s]+\s+[0-9]+)*)\s*$')
STANDARD_TERMINAL = re.compile(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([0-9]+)((\s+[^\s]+\s+[0-9]+)*)\s*$')

DISCODOP_NONTERMINAL = re.compile(r'^#([0-9]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+'
                                  r'([^\s]+)\s+([0-9]+)((\s+[^\s]+\s+[0-9]+)*)\s*$')
DISCODOP_TERMINAL = re.compile(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+'
                               r'([0-9]+)((\s+[^\s]+\s+[0-9]+)*)\s*$')


# Sentence number to name.
# file_name: int
# return: string
def num_to_name(num):
    return str(num)


# Return trees for names:
# names: list of string
# file_name: string
# return: list of hybrid trees obtained
def sentence_names_to_hybridtrees(names,
                                  file_name,
                                  enc="utf-8",
                                  disconnect_punctuation=True,
                                  add_vroot=False,
                                  mode="STANDARD",
                                  secedge=False):
    negra = codecs.open(expanduser(file_name), encoding=enc)
    trees = []
    tree = None
    name = ''
    n_leaves = 0
    node_to_children = {}
    for line in negra:
        match_mode = DISCODOP_HEADER.match(line)
        if match_mode:
            mode = "DISCO-DOP"
            continue
        match_sent_start = BOS.search(line)
        match_sent_end = EOS.match(line)
        if mode == "STANDARD":
            match_nont = \
                STANDARD_NONTERMINAL.match(line)
            match_term = \
                STANDARD_TERMINAL.match(line)
        elif mode == "DISCO-DOP":
            match_nont = DISCODOP_NONTERMINAL.match(line)
            match_term = DISCODOP_TERMINAL.match(line)
        if match_sent_start:
            this_name = match_sent_start.group(1)
            if this_name in names:
                name = this_name
                if secedge:
                    tree = HybridDag(name)
                else:
                    tree = ConstituentTree(name)
                n_leaves = 0
                node_to_children = {}
                if add_vroot:
                    tree.set_label('0', 'VROOT')
                    tree.add_to_root('0')
        elif match_sent_end:
            this_name = match_sent_end.group(1)
            if name == this_name:
                tree.reorder()
                trees += [tree]
                tree = None
        elif tree:
            secedges = []
            if match_nont:
                id = match_nont.group(1)
                if mode == "STANDARD":
                    OFFSET = 0
                else:
                    OFFSET = 1
                nont = match_nont.group(2 + OFFSET)
                edge = match_nont.group(4 + OFFSET)
                parent = match_nont.group(5 + OFFSET)
                # print(match_nont.groups(), len(match_nont.groups()))
                secedges = [] if not secedge or match_nont.group(6 + OFFSET) is None else \
                    match_nont.group(6 + OFFSET).split()

                tree.add_node(id, ConstituentCategory(nont), False, True)

                tree.node_token(id).set_edge_label(edge)
                if parent == '0' and not add_vroot:
                    tree.add_to_root(id)
                else:
                    tree.add_child(parent, id)
                if secedge and secedges:
                    # print(secedges)
                    for sei in range(0, len(secedges) // 2, 2):
                        # sec_label = secedges[sei]
                        assert secedges[sei] == edge
                        sec_parent = secedges[sei + 1]
                        tree.add_sec_child(sec_parent, id)
            elif match_term:
                if mode == "STANDARD":
                    OFFSET = 0
                else:
                    OFFSET = 1

                word = match_term.group(1)
                pos = match_term.group(2 + OFFSET)
                edge = match_term.group(4 + OFFSET)
                parent = match_term.group(5 + OFFSET)
                # print(match_term.groups(), len(match_term.groups()))
                secedges = [] if not secedge or match_term.group(6 + OFFSET) is None else \
                    match_term.group(6 + OFFSET).split()

                n_leaves += 1
                leaf_id = str(100 + n_leaves)
                if parent == '0' and disconnect_punctuation:
                    tree.add_punct(leaf_id, pos, word)
                else:
                    if parent == '0' and not add_vroot:
                        tree.add_to_root(leaf_id)
                    else:
                        tree.add_child(parent, leaf_id)

                    token = ConstituentTerminal(word, pos, edge, None, '--')
                    tree.add_node(leaf_id, token, True, True)

                    tree.node_token(leaf_id).set_edge_label(edge)
                    if secedge and secedges:
                        # print(secedges)
                        for sei in range(0, len(secedges) // 2, 2):
                            # sec_label = secedges[sei]
                            assert secedges[sei] == edge
                            sec_parent = secedges[sei + 1]
                            tree.add_sec_child(sec_parent, leaf_id)
    negra.close()
    return trees


def generate_ids_for_inner_nodes(tree, node_id, idNum):
    """
    generates a dictionary which assigns each tree id an numeric id like specified in export format
    :param tree: parse tree
    :type: ConstituentTree
    :param node_id: id of current node
    :type: str
    :param idNum: current dictionary 
    :type: dict
    :return: nothing
    """
    count = 500+len(tree.ids())

    if len(idNum) is not 0:
        count = min(idNum.values())

    if node_id not in tree.id_yield():
        idNum[node_id] = count-1

    for child in tree.children(node_id):
        generate_ids_for_inner_nodes(tree, child, idNum)

    return


def hybridtree_to_sentence_name(tree, idNum):
    """
    generates lines for given tree in export format
    :param tree: parse tree
    :type: ConstituentTree
    :param idNum: dictionary mapping node id to a numeric id
    :type: dict
    :return: list of lines
    :rtype: list of str
    """
    lines = []

    for leaf in tree.full_yield():
        token = tree.node_token(leaf)
        morph = u'--'
        # if not isinstance(token.form(), str):
        #     print(token.form(), type(token.form()))
        #     assert isinstance(token.form(), str)
        line = [token.form(), token.pos(), morph, token.edge()]

        if leaf in tree.id_yield() and leaf not in tree.root:
            if tree.parent(leaf) is None or tree.parent(leaf) not in idNum:
                print(tree, leaf, tree.full_yield(), list(map(text, tree.full_token_yield())), tree.parent(leaf), tree.parent(leaf) in idNum)
            lines.append(u'\t'.join(line + [text(idNum[tree.parent(leaf)])]) + u'\n')
        else:
            lines.append(u'\t'.join(line + [u'0']) + u'\n')

    for id in tree.ids():
        token = tree.node_token(id)
        morph = u'--'

        line = [u'#' + text(idNum[id]), text(token.category()), morph, token.edge()]

        if id in tree.root:
            lines.append(u'\t'.join(line + [u'0']) + u'\n')
        elif id not in tree.root:
            lines.append(u'\t'.join(line + [text(idNum[tree.parent(id)])]) + u'\n')

    return lines


def hybridtrees_to_sentence_names(trees, counter, length):
    """
    converts a sequence of parse tree to the export format
    :param trees: list of parse trees
    :type: list of ConstituentTrees
    :return: list of export format lines
    :rtype: list of str
    """
    sentence_names = []

    for tree in trees:
        if len(tree.full_yield()) <= length:
            idNum = dict()
            for root in tree.root:
                generate_ids_for_inner_nodes(tree, root, idNum)
            sentence_names.append(u'#BOS ' + text(counter) + u'\n')
            sentence_names.extend(hybridtree_to_sentence_name(tree, idNum))
            sentence_names.append(u'#EOS ' + text(counter) + u'\n')
            counter += 1

    return sentence_names


def acyclic_syntax_graph_to_sentence_name(dsg, sec_edge_to_terminal=False):
    """
    :type dsg: DeepSyntaxGraph
    :type sec_edge_to_terminal: bool
    :param sec_edge_to_terminal: if true, exports secondary edges with terminals as target
    """
    assert not dsg.dog.cyclic()
    assert len(dsg.sentence) < 500

    enum = Enumerator(first_index=500)
    # NB: contrary to the export standard, we index words starting from 1 (and not starting from 0)
    # NB: because 0 also refers to the virtual root (important for sec_edge_to_terminal == True)
    # NB: see http://www.coli.uni-saarland.de/~thorsten/publications/Brants-CLAUS98.pdf
    # NB: only relevant for TiGer s22084, probably annotation error
    synced_idxs = {idx: i + 1 for i, l in enumerate(dsg.synchronization) for idx in l}

    def idNum(tree_idx):
        if tree_idx in synced_idxs:
            return str(synced_idxs[tree_idx])
        else:
            return str(enum.object_index(tree_idx))

    # NB: here we enforce the indices to be topologically ordered as required by the export standard
    for idx in dsg.dog.topological_order():
        if idx not in synced_idxs:
            idNum(idx)

    lines = []

    for idx, token in enumerate(dsg.sentence):
        assert isinstance(token, ConstituentTerminal)
        # if not isinstance(token.form(), str):
        #     print(token.form(), type(token.form()))
        #     assert isinstance(token.form(), str)
        morph_order = ['person', 'case', 'number', 'tense', 'mood', 'gender', 'degree']
        morph = sorted(token.morph_feats(), key=lambda x: morph_order.index(x[0]))
        morph = '.'.join([str(x[1]) for x in morph if str(x[1]) != '--'])
        if morph == '':
            morph = u'--'
        line = [token.form(), token.pos(), morph]
        tree_idx = dsg.get_graph_position(idx)
        assert len(tree_idx) == 1
        tree_idx = tree_idx[0]

        parents = []
        if tree_idx in dsg.dog.outputs:
            parents.append(u'--')
            parents.append(u'0')

        for parent_idx in dsg.dog.parents:
            if not sec_edge_to_terminal and parent_idx in synced_idxs:
                continue
            edge = dsg.dog.incoming_edge(parent_idx)
            for j, child_idx in enumerate(edge.inputs):
                if child_idx == tree_idx:
                    if j in edge.primary_inputs:
                        parents = [edge.get_function(j), idNum(parent_idx)] + parents
                    else:
                        parents.append(edge.get_function(j))
                        parents.append(idNum(parent_idx))
        line += parents
        lines.append(u'\t'.join(line) + u'\n')

    category_lines = []
    for tree_idx in dsg.dog.nodes:
        token = dsg.dog.incoming_edge(tree_idx).label
        if isinstance(token, ConstituentTerminal):
            continue
        morph = u'--'

        line = ['#' + text(idNum(tree_idx)), token, morph]

        parents = []
        if tree_idx in dsg.dog.outputs:
            parents.append(u'--')
            parents.append(u'0')

        for parent_idx in dsg.dog.parents:
            if not sec_edge_to_terminal and parent_idx in synced_idxs:
                continue
            edge = dsg.dog.incoming_edge(parent_idx)
            for j, child_idx in enumerate(edge.inputs):
                if child_idx == tree_idx:
                    if j in edge.primary_inputs:
                        parents = [edge.get_function(j), idNum(parent_idx)] + parents
                    else:
                        parents.append(edge.get_function(j))
                        parents.append(idNum(parent_idx))
        line += parents

        category_lines.append(line)

    category_lines = sorted(category_lines, key=lambda x: x[0])

    for line in category_lines:
        lines.append(u'\t'.join(line) + u'\n')

    return lines


def acyclic_graphs_to_sentence_names(dsgs, counter, length):
    """
    converts a sequence of parse tree to the export format
    :param trees: list of parse trees
    :type: list of ConstituentTrees
    :return: list of export format lines
    :rtype: list of str
    """
    sentence_names = []

    for dsg in dsgs:
        if len(dsg.sentence) <= length:
            sentence_names.append(u'#BOS ' + text(counter) + u'\n')
            sentence_names.extend(acyclic_syntax_graph_to_sentence_name(dsg))
            sentence_names.append(u'#EOS ' + text(counter) + u'\n')
            counter += 1

    return sentence_names


__all__ = ["sentence_names_to_hybridtrees", "hybridtrees_to_sentence_names", "hybridtree_to_sentence_name",
           "acyclic_syntax_graph_to_sentence_name", "acyclic_graphs_to_sentence_names"]
