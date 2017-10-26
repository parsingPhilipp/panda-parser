# Parsing of the Tiger corpus and capture of hybrid trees or dee syntax graphs
from __future__ import print_function
import re
from os.path import expanduser

try:
    import xml.etree.cElementTree as cET
except ImportError:
    import xml.etree.ElementTree as cET

from hybridtree.constituent_tree import ConstituentTree
from graphs.dog import DirectedOrderedGraph, DeepSyntaxGraph
from util.enumerator import Enumerator
from hybridtree.monadic_tokens import ConstituentTerminal

# Location of Tiger corpus.
tiger_dir = 'res/tiger'

# Uncomment depending on whether complete corpus is used or subsets.
# For testing purposes, smaller portions where manually extracted from the
# complete XML file, which takes a long time to parse.
# Results were put in tiger_8000.xml and test.xml.
tiger = tiger_dir + '/tiger_release_aug07.corrected.16012013.xml'
tiger_test = tiger_dir + '/tiger_8000.xml'


# To hold parsed XML file. Cached for efficiency.
xml_file = None


def clear():
    global xml_file
    xml_file = None


# Determine XML file holding data, given file name.
# file_name: string
def initialize(file_name):
    global xml_file
    if xml_file is None:
        xml_file = cET.parse(file_name)


# Sentence number to name.
# file_name: int
# return: string
def num_to_name(num):
    return 's' + str(num)


# Return trees for names.
# names: list of string
# file_name: string
# hold: boolean
# return: list of hybrid trees obtained
def sentence_names_to_hybridtrees(names, file_name, hold=True, disconnect_punctuation=True):
    trees = []
    for name in names:
        tree = sentence_name_to_hybridtree(name, file_name, disconnect_punctuation)
        if tree is not None:
            trees += [tree]
        else:
            print('missing', name)

    if not hold:
        clear()
    return trees


# Return tree for name. Return None if none.
# name: string 
# file_name: string
# return: ConstituentTree
def sentence_name_to_hybridtree(name, file_name, disconnect_punctuation=True):
    initialize(expanduser(file_name))
    sent = xml_file.find('.//body/s[@id="%s"]' % name)
    if sent is not None:
        tree = ConstituentTree(name)
        graph = sent.find('graph')
        root = graph.get('root')
        tree.add_to_root(root)
        for term in graph.iterfind('terminals/t'):
            ident = term.get('id')
            word = term.get('word')
            pos = term.get('pos')
            case = term.get('case')
            number = term.get('number')
            gender = term.get('gender')
            person = term.get('person')
            degree = term.get('degree')
            tense = term.get('tense')
            mood = term.get('mood')
            morph_feats = [("case", case), ("number", number), ("gender", gender), ("person", person), ("tense", tense),
                           ("degree", degree), ("mood", mood)]
            if is_word(pos, word) or not disconnect_punctuation:
                tree.add_leaf(ident, pos, word, morph=morph_feats)
            else:
                tree.add_punct(ident, pos, word)
        for nont in graph.iterfind('nonterminals/nt'):
            ident = nont.get('id')
            cat = nont.get('cat')
            tree.set_label(ident, cat)
            for child in nont.iterfind('edge'):
                child_id = child.get('idref')
                if not is_punct(graph, child_id) or not disconnect_punctuation:
                    tree.add_child(ident, child_id)
        for nont in graph.iterfind('nonterminals/nt'):
            for child in nont.iterfind('edge'):
                child_id = child.get('idref')
                edge_label = child.get('label')
                if (not is_punct(graph, child_id) or not disconnect_punctuation) and edge_label is not None:
                    tree.node_token(child_id).set_edge_label(edge_label)
        return tree
    else:
        return None


def sentence_names_to_deep_syntax_graphs(names, file_name, hold=True, reorder_children=False):
        dsgs = []
        for name in names:
            dsg = sentence_name_to_deep_syntax_graph(name, file_name, reorder_children)
            if dsg is not None:
                dsgs += [dsg]
            else:
                print('missing', name)

        if not hold:
            clear()
        return dsgs


# Return tree for name. Return None if none.
# name: string
# file_name: string
# return: ConstituentTree
def sentence_name_to_deep_syntax_graph(name, file_name, reorder_children=False):
    initialize(expanduser(file_name))
    sent = xml_file.find('.//body/s[@id="%s"]' % name)
    if sent is not None:
        dog = DirectedOrderedGraph()
        sync = []
        sentence = []

        deep_syntax_graph = DeepSyntaxGraph(sentence, dog, sync, label=name)

        node_enum = Enumerator()

        inner_nodes = {}

        graph = sent.find('graph')

        for term in graph.iterfind('terminals/t'):
            ident = term.get('id')
            word = term.get('word')
            pos = term.get('pos')
            case = term.get('case')
            number = term.get('number')
            gender = term.get('gender')
            person = term.get('person')
            degree = term.get('degree')
            tense = term.get('tense')
            mood = term.get('mood')
            morph_feats = [("case", case), ("number", number), ("gender", gender), ("person", person),
                           ("tense", tense),
                           ("degree", degree), ("mood", mood)]
            if is_word(pos, word):
                output_idx = node_enum.object_index(ident)
                dog.add_node(output_idx)
                terminal = ConstituentTerminal(word,  # .encode('utf_8'),
                                               pos, morph=morph_feats)
                dog.add_terminal_edge([], ConstituentTerminal(word, pos, morph=morph_feats), output_idx)
                sentence.append(terminal)
                sync.append([output_idx])
                # tree.add_leaf(id, pos, word.encode('utf_8'), morph=morph_feats)
                for parent in term.iterfind('secedge'):
                    parent_id = parent.get('idref')
                    edge_label = parent.get('label')
                    parent_idx = node_enum.object_index(parent_id)
                    if parent_idx not in inner_nodes:
                        inner_nodes[parent_idx] = ('_', [(output_idx, 's', edge_label)])
                    else:
                        inner_nodes[parent_idx][1].append((output_idx, 's', edge_label))
            else:
                # todo: handle punctuation
                pass

        for nont in graph.iterfind('nonterminals/nt'):
            ident = nont.get('id')
            cat = nont.get('cat')
            idx = node_enum.object_index(ident)
            dog.add_node(idx)

            if idx not in inner_nodes:
                inner_nodes[idx] = (cat, [])
            else:
                inner_nodes[idx] = (cat, inner_nodes[idx][1])

            for child in nont.iterfind('edge'):
                child_id = child.get('idref')
                child_idx = node_enum.object_index(child_id)
                edge_label = child.get('label')
                if not is_punct(graph, child_id):
                    inner_nodes[idx][1].append((child_idx, 'p', edge_label))

            for parent in nont.iterfind('secedge'):
                parent_id = parent.get('idref')
                edge_label = parent.get('label')
                parent_idx = node_enum.object_index(parent_id)
                if parent_idx not in inner_nodes:
                    inner_nodes[parent_idx] = ('_', [(idx, 's', edge_label)])
                else:
                    inner_nodes[parent_idx][1].append((idx, 's', edge_label))

        for idx in inner_nodes:
            if reorder_children:
                reordered = sorted(inner_nodes[idx][1], key=lambda x: (x[2], inner_nodes[idx][1].index(x)))
            else:
                reordered = inner_nodes[idx][1]
            edge = dog.add_terminal_edge(reordered, inner_nodes[idx][0], idx)
            for i, tentacle in enumerate(reordered):
                edge.set_function(i, reordered[i][2])

        root = graph.get('root')
        root_idx = node_enum.object_index(root)
        dog.add_to_outputs(root_idx)

        return deep_syntax_graph
    else:
        return None


# Is word? Exclude bullet, POS starting with $, and words tagged as
# XY ('Nichtwort, Sonderzeichen').
# pos: string (part of speech)
# word: string
# return: boolean 
def is_word(pos, word):
    return not (re.search(r'&bullet;', word) or re.search(r'^\$', pos) or
                (re.search(r'^XY$', pos) and re.search(r'^[a-z]$', word)))


# In graph, is element specified by id punctuation?
# graph: XML element
# id: string
def is_punct(graph, ident):
    term = graph.find('terminals/t[@id="%s"]' % ident)
    if term is not None:
        word = term.get('word')
        pos = term.get('pos')
        return not is_word(pos, word)
    else:
        return False
