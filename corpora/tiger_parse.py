# Parsing of the Tiger corpus and capture of hybrid trees.

import re
from os.path import expanduser

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# from hybridtree import HybridTree
from hybridtree.constituent_tree import HybridTree

# Location of Tiger corpus.
tiger_dir = '..'

# Uncomment depending on whether complete corpus is used or subsets.
# For testing purposes, smaller portions where manually extracted from the
# complete XML file, which takes a long time to parse.
# Results were put in tiger_8000.xml and test.xml.
# tiger = tiger_dir + '/tiger_release_aug07.corrected.16012013.xml'
tiger = tiger_dir + '/tiger.xml'
tiger_test = tiger_dir + '/tiger_8000.xml'


# To hold parsed XML file. Cached for efficiency.
xml_file = None


# Determine XML file holding data, given file name.
# file_name: string
def initialize(file_name):
    global xml_file
    if xml_file is None:
        xml_file = ET.parse(file_name)


# Sentence number to name.
# file_name: int
# return: string
def num_to_name(num):
    return 's' + str(num)


# Return trees for names.
# names: list of string
# file_name: string
# return: list of hybrid trees obtained
def sentence_names_to_hybridtrees(names, file_name):
    trees = []
    for name in names:
        tree = sentence_name_to_hybridtree(name, file_name)
        if tree is not None:
            trees += [tree]
        else:
            print 'missing', name
    return trees


# Return tree for name. Return None if none.
# name: string 
# file_name: string
# return: HybridTree
def sentence_name_to_hybridtree(name, file_name):
    initialize(expanduser(file_name))
    sent = xml_file.find('.//body/s[@id="%s"]' % name)
    if sent is not None:
        tree = HybridTree(name)
        graph = sent.find('graph')
        root = graph.get('root')
        tree.add_to_root(root)
        for term in graph.iterfind('terminals/t'):
            id = term.get('id')
            word = term.get('word')
            pos = term.get('pos')
            if is_word(pos, word):
                tree.add_leaf(id, pos, word)
            else:
                tree.add_punct(id, pos, word)
        for nont in graph.iterfind('nonterminals/nt'):
            id = nont.get('id')
            cat = nont.get('cat')
            tree.set_label(id, cat)
            for child in nont.iterfind('edge'):
                child_id = child.get('idref')
                if not is_punct(graph, child_id):
                    tree.add_child(id, child_id)
        return tree
    else:
        return None


# Is word? Exclude bullet, POS starting with $, and words tagged as
# XY ('Nichtwort, Sonderzeichen').
# pos: string (part of speech)
# word: string
# return: boolean 
def is_word(pos, word):
    return not ( re.search(r'&bullet;', word) or re.search(r'^\$', pos) or \
                 (re.search(r'^XY$', pos) and re.search(r'^[a-z]$', word) ) )


# In graph, is element specified by id punctuation?
# graph: XML element
# id: string
def is_punct(graph, id):
    term = graph.find('terminals/t[@id="%s"]' % id)
    if term is not None:
        word = term.get('word')
        pos = term.get('pos')
        return not is_word(pos, word)
    else:
        return False
