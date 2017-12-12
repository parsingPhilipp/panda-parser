# Parsing of the Negra corpus and capture of hybrid trees.
from __future__ import print_function, unicode_literals
from os.path import expanduser
from hybridtree.constituent_tree import ConstituentTree
from grammar.lcfrs import *
try:
    from __builtin__ import str as text
except ModuleNotFoundError:
    text = str

# Location of Negra corpus.
negra_dir = 'res/negra-corpus/downloadv2'

# The non-projective and projective versions of the negra corpus.
negra_nonproj = negra_dir + '/negra-corpus.export'
negra_proj = negra_dir + '/negra-corpus.cfg'


# Sentence number to name.
# file_name: int
# return: string
def num_to_name(num):
    return str(num)


# Return trees for names:
# names: list of string
# file_name: string
# return: list of hybrid trees obtained
def sentence_names_to_hybridtrees(names, file_name):
    negra = codecs.open(expanduser(file_name), encoding='utf-8') #encoding='iso-8859-1')
    trees = []
    tree = None
    name = ''
    n_leaves = 0
    node_to_children = {}
    mode = "STANDARD"
    for line in negra:
        match_mode = re.search(r'^%%\s+word\s+lemma\s+tag\s+morph\s+edge\s+parent\s+secedge$', line)
        if match_mode:
            mode = "DISCO-DOP"
            continue
        match_sent_start = re.search(r'^#BOS\s+([0-9]+)', line)
        match_sent_end = re.search(r'^#EOS\s+([0-9]+)$', line)
        if mode == "STANDARD":
            match_nont = \
                re.search(r'^#([0-9]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)(\n)?$', line)
            match_term = \
                re.search(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)(\n)?$', line)
        elif mode == "DISCO-DOP":
            match_nont = \
                re.search(r'^#([0-9]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)(\s+([^\s]+))?(\n)?$', line)
            match_term = \
                re.search(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)(\s+([^\s]+))?(\n)?$', line)
        if match_sent_start:
            this_name = match_sent_start.group(1)
            if this_name in names:
                name = this_name
                tree = ConstituentTree(name)
                n_leaves = 0
                node_to_children = {}
        elif match_sent_end:
            this_name = match_sent_end.group(1)
            if name == this_name:
                tree.reorder()
                trees += [tree]
                tree = None
        elif tree:
            if match_nont:
                id = match_nont.group(1)
                if mode == "STANDARD":
                    nont = match_nont.group(2)
                    edge = match_nont.group(4)
                    parent = match_nont.group(5)
                elif mode == "DISCO-DOP":
                    nont = match_nont.group(3)
                    edge = match_nont.group(5)
                    parent = match_nont.group(6)
                tree.set_label(id, nont)
                tree.node_token(id).set_edge_label(edge)
                if parent == '0':
                    tree.add_to_root(id)
                else:
                    tree.add_child(parent, id)
            elif match_term:
                word = match_term.group(1)
                if mode == "STANDARD":
                    pos = match_term.group(2)
                    edge = match_term.group(4)
                    parent = match_term.group(5)
                elif mode == "DISCO-DOP":
                    pos = match_term.group(3)
                    edge = match_term.group(5)
                    parent = match_term.group(6)
                n_leaves += 1
                leaf_id = str(100 + n_leaves)
                if parent == '0':
                    tree.add_punct(leaf_id, pos, word)
                else:
                    tree.add_leaf(leaf_id, pos, word)
                    tree.add_child(parent, leaf_id)
                    tree.node_token(leaf_id).set_edge_label(edge)
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
                print(tree, leaf, tree.full_yield(), map(text, tree.full_token_yield()), tree.parent(leaf), tree.parent(leaf) in idNum)
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
