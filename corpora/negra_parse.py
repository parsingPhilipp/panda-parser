# Parsing of the Negra corpus and capture of hybrid trees.

from os.path import expanduser

from hybridtree.constituent_tree import ConstituentTree
from grammar.LCFRS.lcfrs import *

# Location of Negra corpus.
negra_dir = '~/Data/Negra'

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
