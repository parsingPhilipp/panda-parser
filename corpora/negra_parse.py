# Parsing of the Negra corpus and capture of hybrid trees.

from os.path import expanduser
from hybridtree.constituent_tree import ConstituentTree
from grammar.lcfrs import *

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
    negra = codecs.open(expanduser(file_name), encoding='iso-8859-1')
    trees = []
    tree = None
    name = ''
    n_leaves = 0
    node_to_children = {}
    for line in negra:
        match_sent_start = re.search(r'^#BOS ([0-9]+)', line)
        match_sent_end = re.search(r'^#EOS ([0-9]+)', line)
        match_nont = \
            re.search(r'^#([^\t]+)\t+([^\t]+)\t+([^\t]+)\t+([^\t]+)\t+([^\t\n]+)$', line)
        match_term = \
            re.search(r'^([^\t]+)\t+([^\t]+)\t+([^\t]+)\t+([^\t]+)\t+([^\t\n]+)$', line)
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
                nont = match_nont.group(2)
                parent = match_nont.group(5)
                tree.set_label(id, nont)
                if parent == '0':
                    tree.add_to_root(id)
                else:
                    tree.add_child(parent, id)
            elif match_term:
                word = match_term.group(1)
                pos = match_term.group(2)
                parent = match_term.group(5)
                n_leaves += 1
                leaf_id = str(100 + n_leaves)
                if parent == '0':
                    tree.add_punct(leaf_id, pos, word)
                else:
                    tree.add_leaf(leaf_id, pos, word)
                    tree.add_child(parent, leaf_id)
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
        line = str(token.form()) + ' ' + str(token.pos()) + ' -- -- '
        if leaf in tree.id_yield():
            lines.append(line + str(idNum[tree.parent(leaf)]) + '\n')
        else:
            lines.append(line + '0\n')

    for id in tree.ids():
        token = tree.node_token(id)
        line = '#' + str(idNum[id]) + ' ' + str(token.category()) + ' -- -- '

        if tree.parent(id) == tree.root[0] and tree.node_token(tree.parent(id)).category() == 'VROOT':
            lines.append(line + '0\n')
        elif token.category() != 'VROOT' and id == tree.root[0]:
            lines.append(line + '0\n')
        elif id != tree.root[0]:
            lines.append(line + str(idNum[tree.parent(id)]) + '\n')

    return lines


def hybridtrees_to_sentence_names(trees):
    """
    converts a sequence of parse tree to the export format
    :param trees: list of parse trees
    :type: list of ConstituentTrees
    :return: list of export format lines
    :rtype: list of str
    """
    sentence_names = []
    counter = 1

    for tree in trees:
        idNum = dict()
        generate_ids_for_inner_nodes(tree, tree.root[0], idNum)
        sentence_names.append('#BOS ' + str(counter) + '\n')
        sentence_names.extend(hybridtree_to_sentence_name(tree, idNum))
        sentence_names.append('#EOS ' + str(counter) + '\n')
        counter += 1

    return sentence_names
