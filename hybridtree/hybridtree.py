# Hybrid tree.
# Is like ordinary tree, but with separate ordering on leaves

from collections import defaultdict
import Tkinter as tk
import tkFont

from decomposition import *


# Tree uniquely identified by a sentence label.
class HybridTree:
    # Constructor.
    # sent_label: string
    def __init__(self, sent_label=None):
        # label of sentence
        self.__sent_label = sent_label
        # root is id of node
        self.__root = None
        # list of leaves, each of which is triple (id, pos, word)
        self.__leaves = []
        # number of leaves excluding punctuation
        self.__n_real_leaves = 0
        # maps id to leaf
        self.__id_to_leaf = {}
        # maps id of leaf to its index (counting non-punctuation left to right)
        self.__id_to_leaf_index = {}
        # maps index to leaf
        self.__index_to_leaf = {}
        # maps node id to label
        self.__id_to_label = {}
        # maps node id to list of ids of children
        self.__id_to_child_ids = {}

    # Get label of sentence.
    # return: string
    def sent_label(self):
        return self.__sent_label

    # Set root to node given by id.
    # id: string
    def set_root(self, id):
        self.__root = id

    # Has root been determined?
    # return: bool
    def rooted(self):
        return self.__root is not None

    # Get id of root.
    # return: string
    def root(self):
        return self.__root

    # Add next leaf. Order of adding is significant.
    # id: string
    # pos: string (part of speech)
    # word: string
    def add_leaf(self, id, pos, word):
        leaf_descr = (id, pos, word)
        self.__id_to_leaf[id] = leaf_descr
        self.__leaves += [leaf_descr]
        self.__id_to_leaf_index[id] = self.__n_real_leaves
        self.__n_real_leaves += 1
        self.__index_to_leaf[self.__n_real_leaves] = leaf_descr

    # Add punctuation (has no parent).
    # id: string
    # pos: string (part of speech)
    # word: string
    def add_punct(self, id, pos, word):
        leaf_descr = (id, pos, word)
        self.__id_to_leaf[id] = leaf_descr
        self.__leaves += [leaf_descr]

    # Add label of non-leaf. If it has no children, give it empty list of
    # children.
    # id: string
    # label: string
    def set_label(self, id, label):
        self.__id_to_label[id] = label
        if id not in self.__id_to_child_ids:
            self.__id_to_child_ids[id] = []

    # Add child of non-leaf.
    # children.
    # parent: string 
    # child: string 
    def add_child(self, parent, child):
        if parent not in self.__id_to_child_ids:
            self.__id_to_child_ids[parent] = []
        self.__id_to_child_ids[parent] += [child]

    # All leaves of tree.
    # return: list of triples.
    def leaves(self):
        return self.__leaves

    # Is leaf? This is if there are no children (not even empty list of
    # children).
    # id: string
    # return: bool
    def is_leaf(self, id):
        return id not in self.__id_to_child_ids

    # Get leaf for index.
    # index: int
    # return: triple
    def index_leaf(self, index):
        return self.__index_to_leaf[index]

    # Get index for id of leaf.
    # id: string
    # return: int
    def leaf_index(self, id):
        return self.__id_to_leaf_index[id]

    # Get part of speech of node.
    # id: string
    # return: string
    def leaf_pos(self, id):
        return self.__id_to_leaf[id][1]

    # Get word of node.
    # id: string
    # return: string
    def leaf_word(self, id):
        return self.__id_to_leaf[id][2]

    # Get yield as list of POS, omitting punctuation.
    # return: list of string
    def pos_yield(self):
        return [self.leaf_pos(id) for (id, _, _) in self.__leaves
                if id in self.__id_to_leaf_index]

    # Get yield as list of words, omitting punctuation.
    # return: list of string
    def word_yield(self):
        return [self.leaf_word(id) for (id, _, _) in self.__leaves
                if id in self.__id_to_leaf_index]

    # Get label of (non-leaf) node.
    # id: string
    # return: string
    def label(self, id):
        return self.__id_to_label[id]

    # Get child ids of node.
    # id: string
    # return: list of string
    def children(self, id):
        if id in self.__id_to_child_ids:
            return self.__id_to_child_ids[id]
        else:
            return []

    # Get parent (id) of node. Or None.
    # id: string
    # return: string
    def parent(self, id):
        return self.__parent_recur(id, self.root())

    def __parent_recur(self, child, id):
        if child in self.children(id):
            return id
        else:
            for next in self.children(id):
                parent = self.__parent_recur(child, next)
                if parent is not None:
                    return parent
        return None

    # Is there node that is child of two nodes?
    # return: bool
    def reentrant(self):
        parent = defaultdict(list)
        for id in self.__id_to_child_ids:
            for child in self.children(id):
                parent[child] += [id]
        for id in parent:
            if len(parent[id]) > 1:
                return True
        return False

    # Get ids of all nodes.
    # return: list of string
    def ids(self):
        return self.__id_to_child_ids.keys()

    # Reorder children according to first terminal in yield.
    def reorder(self):
        self.__reorder(self.root())

    # id: string
    def __reorder(self, id):
        if not self.is_leaf(id):
            for child in self.children(id):
                self.__reorder(child)
            self.__id_to_child_ids[id] = sorted(self.children(id), \
                                                key=lambda i: self.__first_term_index(i))

    # Return index of left-most terminal in yield.
    # id: string
    # return: int
    def __first_term_index(self, id):
        if self.is_leaf(id):
            if id in self.__id_to_leaf_index:
                return self.leaf_index(id)
            else:
                return -1
        elif len(self.children(id)) > 0:
            return self.__first_term_index(self.children(id)[0])
        else:
            return 0

    # Number of nodes in total tree.
    # return: int
    def n_nodes(self):
        return self.__n_nodes_below(self.root()) + 1

    # TODO (Kilian): root node seems to be counted twice

    # Number of nodes below node.
    # id: string
    # return: int
    def __n_nodes_below(self, id):
        if self.is_leaf(id):
            return 1
        else:
            n = 0
            for child in self.children(id):
                n += self.__n_nodes_below(child)
            return n + 1

    # Total number of gaps in any node.
    # return: int
    def n_gaps(self):
        return self.__n_gaps_below(self.root())

    # id: string
    # return: int
    def __n_gaps_below(self, id):
        n_gaps = self.n_spans(id) - 1
        if not self.is_leaf(id):
            for child in self.children(id):
                n_gaps += self.__n_gaps_below(child)
        return n_gaps

    # Yield below node with id, i.e. indices of leaves. Is unordered.
    # id: string
    # return: list of int
    def fringe(self, id):
        if self.is_leaf(id):
            if id in self.__id_to_leaf_index:
                return [self.leaf_index(id)]
            else:
                return []
        else:
            y = []
            for child in self.children(id):
                y += self.fringe(child)
            return y

    # Is there any internal node with empty yield?
    # Includes the case the root has no children.
    # return: bool
    def empty_fringe(self):
        for id in self.ids():
            if len(self.children(id)) == 0:
                return True
        return self.rooted() and len(self.fringe(self.root())) == 0

    # Does yield cover whole string?
    # return: bool
    def complete(self):
        return self.rooted() and \
               len(self.fringe(self.root())) == self.__n_real_leaves

    # Number of contiguous spans of node.
    # id: string
    # return: int
    def n_spans(self, id):
        if self.is_leaf(id):
            return 1
        else:
            return len(join_spans(self.fringe(id)))

    # Maximum number of spans of any node.
    # return: int
    def max_n_spans(self):
        nums = [self.n_spans(id) for id in self.ids()]
        if len(nums) > 0:
            return max(nums)
        else:
            return 1

    # Create unlabelled structure, only in terms of breakup of yield.
    # return: pair consisting of head
    def unlabelled_structure(self):
        return self.unlabelled_structure_recur(self.root())

    def unlabelled_structure_recur(self, id):
        head = set(self.fringe(id))
        tail = [self.unlabelled_structure_recur(child) for \
                child in self.children(id)]
        # remove useless step
        if len(tail) == 1 and head == tail[0][0]:
            return tail[0]
        else:
            return head, tail

    # Labelled spans.
    # return: list of spans (each of which is string plus an even
    # number of (integer) positions)
    # TODO (kilian) some spans occur multiple times
    def labelled_spans(self):
        spans = []
        for id in self.ids():
            span = [self.label(id)]
            for (low, high) in join_spans(self.fringe(id)):
                span += [low, high]
                spans += [span]
        return sorted(spans, \
                      cmp=lambda x, y: cmp([x[1]] + [-x[2]] + x[3:] + [x[0]], \
                                           [y[1]] + [-y[2]] + y[3:] + [y[0]]))

    # Produce canvas representing hybrid tree.
    # return: canvas
    def canvas(self):
        return canvas(self.__graph())

    # Produce hybrid tree in form of graph, with nice layout.
    def __graph(self):
        return format_tree(self, self.sent_label())

    # String representation by recursive descent.
    # return: string
    def __str__(self):
        return self.__str_recur(self.__root, '')

    # id: string
    # indent: int
    # return: string
    def __str_recur(self, id, indent):
        s = ''
        if self.is_leaf(id):
            for (other_id, pos, word) in self.leaves():
                if other_id == id:
                    s = indent + id + ' = ' + pos + ' ' + word + '\n'
        else:
            s = indent + id + ' = ' + self.label(id) + ' ->\n'
            for child in self.children(id):
                s += self.__str_recur(child, indent + '  ')
        return unicode(s)


# ###########################

# Abstract representation of drawing of tree.
class __Graph:
    # Constructor.
    # label: string
    def __init__(self, label=None):
        self.master = tk.Tk()
        if label:
            self.master.title(label)
        self.font_fam = 'Helvetica'
        self.font_size = 8
        self.font = tkFont.Font(family=self.font_fam, \
                                size=self.font_size)
        self.width = 0
        self.x_max = 0
        self.height = 0
        self.x_unit = 40
        self.y_unit = 30
        self.margin = 10
        # map id to x and y of position
        self.id_to_x = {}
        self.id_to_y = {}
        # maps y to list of spans of horizontal lines
        self.hor_lines = defaultdict(list)
        # list of lines
        self.lines = []
        # list of nodes
        self.nodes = []

    def text_width(self, text):
        return self.font.measure(text)

    def text_height(self):
        return self.font.metrics('linespace')

    # Find y where line or text can be put.
    def find_free_line(self, x_min, x_max, y):
        while True:
            free = True
            for (left, right) in self.hor_lines[y]:
                if x_min <= right and x_max >= left:
                    free = False
            if free:
                return y
            else:
                y += self.y_unit

    def draw_hor_line(self, x_min, x_max, y):
        self.hor_lines[y] += [(x_min, x_max)]
        if x_min < x_max:
            self.lines += [(x_min, y, x_max, y)]

    def draw_label(self, label, x_min, x_max, y):
        self.hor_lines[y] += [(x_min, x_max)]
        self.nodes += [(label, x_min, y)]

    def draw_vert_line(self, x, y_min, y_max):
        self.lines += [(x, y_min, x, y_max)]

    def __str__(self):
        s = ''
        s += 'nodes:\n'
        for (label, x, y) in self.nodes:
            s += label + ' ' + str(x) + ' ' + str(y) + '\n'
        s += 'lines:\n'
        for (x1, y1, x2, y2) in self.lines:
            s += str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n'
        return s


def format_tree(tree, sent_label):
    graph = __Graph(sent_label)
    graph.x_unit = widest_nont(tree, graph)
    graph.y_unit *= 0.8
    format_leaves(tree, graph)
    format_recur(tree, tree.root, graph)
    return graph


def widest_nont(tree, graph):
    w = 5  # arbitrary low value
    for id in tree.ids():
        w = max(w, graph.text_width(tree.label(id) + ' '))
    for (id, pos, word) in tree.leaves():
        w = max(w, graph.text_width(pos + ' '))
    return w


# Place leaves from left to right, moving them up if they could overlap.
def format_leaves(tree, graph):
    for (id, pos, word) in tree.leaves():
        pos_len = graph.text_width(pos + ' ')
        word_len = graph.text_width(word + ' ')
        y_word = graph.find_free_line(graph.width, graph.width + word_len, 0)
        graph.draw_label(word, graph.width, graph.width + word_len, y_word)
        y_pos = graph.find_free_line(graph.width, graph.width + pos_len, y_word + graph.y_unit)
        graph.draw_label(pos, graph.width, graph.width + pos_len, y_pos)
        graph.draw_vert_line(graph.width, y_pos, y_word)
        graph.id_to_x[id] = graph.width
        graph.id_to_y[id] = y_pos
        graph.x_max = max(graph.x_max, graph.width + max(pos_len, word_len))
        graph.width += graph.x_unit


def format_recur(tree, id, graph):
    if not tree.is_leaf(id):
        nont = tree.label(id)
        nont_len = graph.text_width(nont + ' ')
        for child in tree.children(id):
            format_recur(tree, child, graph)
        x_min = graph.width
        x_max = 0
        y = 0
        for child in tree.children(id):
            x_child = graph.id_to_x[child]
            y_child = graph.id_to_y[child]
            x_min = min(x_min, x_child)
            x_max = max(x_max, x_child)
            y = max(y, y_child)
        y_line = graph.find_free_line(x_min, x_max, y + graph.y_unit)
        graph.draw_hor_line(x_min, x_max, y_line)
        for child in tree.children(id):
            x_child = graph.id_to_x[child]
            y_child = graph.id_to_y[child]
            graph.draw_vert_line(x_child, y_line, y_child)
        y_nont = graph.find_free_line(x_min, x_min + nont_len, y_line + graph.y_unit)
        graph.draw_label(nont, x_min, x_min + nont_len, y_nont)
        graph.draw_vert_line(x_min, y_nont, y_line)
        graph.id_to_x[id] = x_min
        graph.id_to_y[id] = y_nont
        graph.height = max(graph.height, y_nont)


###################################################

# Produce canvas from graph. Open window on screen.
# graph: __Graph
def canvas(graph):
    m = graph.master
    w = graph.x_max + 2 * graph.margin
    h = graph.height + 2 * graph.margin
    min_right_space = 40
    min_bottom_space = 100
    window_width = min(m.winfo_screenwidth() - min_right_space, w)
    window_height = min(m.winfo_screenheight() - min_bottom_space, h)
    f = tk.Frame(m, width=w, height=h)
    f.grid(row=0, column=0)
    c = tk.Canvas(f, width=window_width, height=window_height,
                  scrollregion=(0, 0, w, h))
    hbar = tk.Scrollbar(f, orient=tk.HORIZONTAL, command=c.xview)
    hbar.pack(side=tk.BOTTOM, fill=tk.X)
    vbar = tk.Scrollbar(f, orient=tk.VERTICAL, command=c.yview)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    c.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    c.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    c.create_rectangle(0, 0, w, h, fill='white', outline='white')
    for (x1, y1, x2, y2) in graph.lines:
        x1_real = x1 + graph.margin
        y1_real = graph.height - y1 + graph.margin
        x2_real = x2 + graph.margin
        y2_real = graph.height - y2 + graph.margin
        c.create_line(x1_real, y1_real, x2_real, y2_real)
    for (label, x, y) in graph.nodes:
        x_real = x + graph.margin
        y_real = graph.height - y + graph.margin
        c.create_rectangle(x_real, y_real - graph.text_height() / 2, \
                           x_real + graph.text_width(label + ' '), \
                           y_real + graph.text_height() / 2, \
                           fill='white', outline='white')
        c.create_text(x_real, y_real, \
                      text=label, anchor='w', font=graph.font)
    m.mainloop()


#
def test():
    tree = HybridTree("s1")
    tree.add_leaf("f1", "VP", "hat")
    tree.add_leaf("f2", "ADV", "schnell")
    tree.add_leaf("f3", "VP", "gearbeitet")
    tree.add_punct("f4", "PUNC", ".")

    tree.add_child("V", "f1")
    tree.add_child("V", "f3")
    tree.add_child("ADV", "f2")

    tree.add_child("VP", "V")
    tree.add_child("VP", "ADV")

    print "rooted", tree.rooted()
    tree.set_root("VP")
    print "rooted", tree.rooted()
    tree.set_label("V", "V")
    tree.set_label("VP", "VP")
    tree.set_label("ADV", "ADV")

    print "sent label", tree.sent_label()

    print "leaves", tree.leaves()

    print "is leaf (leaves)", [(x, tree.is_leaf(x)) for (x, _, _) in tree.leaves()]
    print "is leaf (internal)", [(x, tree.is_leaf(x)) for x in tree.ids()]
    print "leaf index", [(x, tree.leaf_index(x)) for x in ["f1", "f2", "f3"]]

    print "pos yield", tree.pos_yield()
    print "word yield", tree.word_yield()

    # reentrant
    # parent

    print "ids", tree.ids()

    # reorder
    print "n nodes", tree.n_nodes()
    print "n gaps", tree.n_gaps()

    print "fringe VP", tree.fringe("VP")
    print "fringe V", tree.fringe("V")

    print "empty fringe", tree.empty_fringe()

    print "complete?", tree.complete()

    print "max n spans", tree.max_n_spans()

    print "unlabelled structure", tree.unlabelled_structure()

    print "labelled spans", tree.labelled_spans()


test()