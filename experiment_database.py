__author__ = 'kilian'

import sqlite3
import time
from general_hybrid_tree import GeneralHybridTree
import conll_parse

dbfile = 'examples/example.db'
test_file = 'examples/Dependency_Corpus.conll'
test_file_modified = 'examples/Dependency_Corpus_modified.conll'


def create_experiment_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS experiments (e_id integer primary key autoincrement, term_label text, nont_label text, rec_par text, ignore_punctuation boolean, corpus text, started time, cpu_time time)''')
    connection.commit()


def create_tree_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS trees (t_id integer primary key autoincrement, corpus text, name text, length integer, gaps integer, unique(corpus, name))''')
    # cursor.execute('''CREATE UNIQUE INDEX IF NOT EXISTS tree_idx ON trees(corpus, name)''')
    connection.commit()

def create_result_tree_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS result_trees (rt_id integer primary key autoincrement, t_id integer, exp_id integer, k_best integer, score double, UNIQUE(t_id, exp_id, k_best))''')
    # cursor.execute('''CREATE UNIQUE INDEX IF NOT EXISTS tree_node_idx ON tree_nodes(t_id, sent_position)''')
    connection.commit()

def create_tree_node_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS tree_nodes (t_id integer, sent_position INTEGER, label text, pos text, deprel text, head integer, UNIQUE(t_id, sent_position))''')
    # cursor.execute('''CREATE UNIQUE INDEX IF NOT EXISTS tree_node_idx ON tree_nodes(t_id, sent_position)''')
    connection.commit()

def create_result_tree_node_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS result_tree_nodes (rt_id INTEGER, sent_position INTEGER, deprel text, head integer, unique(rt_id, sent_position))''')
    connection.commit()


def create_grammar_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS grammar (g_id int primary key, experiment integer, nonterminals int, terminals int, rules int, size int, UNIQUE(experiment))''')
    connection.commit()

def create_fanouts_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS fanouts (g_id int primary key, experiment integer, nonterminals int, terminals int, rules int, size int, UNIQUE(experiment))''')
    connection.commit()

def add_grammar(LCFRS):
    pass

def add_experiment(connection, term_label, nont_label, rec_par, ignore_punctuation, corpus, started, cpu_time):
    cursor = connection.cursor()
    cursor.execute('''INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (None, term_label, nont_label, rec_par, ignore_punctuation, corpus, started, cpu_time))
    experiment = cursor.lastrowid
    connection.commit()
    return experiment

def add_tree(connection, tree, corpus):
    """
    :param connection:
    :type tree: GeneralHybridTree
    :type corpus: str
    :return:
    """


    cursor = connection.cursor()
    for row in cursor.execute('''SELECT EXISTS (SELECT * FROM trees WHERE corpus = ? AND name = ?)''',(corpus, tree.sent_label())):
        if row[0]:
            return
    cursor.execute('''INSERT OR IGNORE INTO trees VALUES (?, ?, ?, ?, ?)''', (None
                                                                                  , corpus
                                                                                  , tree.sent_label()
                                                                                  , len(tree.full_yield())
                                                                                  , tree.n_gaps()
                                                                                 ))

    # unique tree key
    tree_id = cursor.lastrowid
    for id in tree.full_yield():
        if tree.root() == id:
            head = 0
        else:
            head = tree.node_index_full(tree.parent(id)) + 1
        cursor.execute('''INSERT INTO tree_nodes VALUES (?, ?, ?, ?, ?, ?)''', (tree_id
                                                                                               , tree.node_index_full(id) + 1
                                                                                               , tree.node_label(id)
                                                                                               , tree.node_pos(id)
                                                                                               , tree.node_dep_label(id)
                                                                                               , head))

    connection.commit()


def add_result_tree(connection, tree, corpus, experiment, k_best, score):
    """
    :param connection:
    :type tree: GeneralHybridTree
    :type corpus: str
    :return:
    """

    cursor = connection.cursor()
    tree_id = None
    for row in cursor.execute('''SELECT t_id FROM trees WHERE corpus = ? AND name = ?''', ( corpus, tree.sent_label())):
        tree_id = row[0]
    if tree_id == None:
        assert("tree not found")

    # unique tree key
    cursor.execute('''INSERT INTO result_trees VALUES (?, ?, ?, ?, ?)''', ( None
                                                                        , tree_id
                                                                        , experiment
                                                                        , k_best
                                                                        , score))
    result_tree_id = cursor.lastrowid

    for id in tree.full_yield():
        # set root head
        if tree.root() == id:
            head = 0
            deprel = "ROOT"
        # connect disconnected nodes to root
        elif tree.disconnected(id):
            head = tree.node_index_full(tree.root()) + 1
            deprel = "PUNC"
        else:
            head = tree.node_index_full(tree.parent(id)) + 1
            deprel = tree.node_dep_label(id)
        cursor.execute('''INSERT INTO result_tree_nodes VALUES (?, ?, ?, ?)''', (result_tree_id
                                                                                       , tree.node_index_full(id) + 1
                                                                                       , deprel
                                                                                       , head))
    connection.commit()

def openDatabase(file):
    connection = sqlite3.connect(file)
    return connection

def dbtest():
    connection = openDatabase(dbfile)
    connection.text_factory = str

    create_experiment_table(connection)

    corpus = test_file
    experiment = add_experiment(connection, 'term_pos', 'child_pos', 'direct_extraction', False, corpus, time.time(), None)

    c = connection.cursor()
    for row in c.execute('SELECT * FROM experiments'):
        print row

    create_tree_table(connection)
    create_tree_node_table(connection)

    for tree in conll_parse.parse_conll_corpus(test_file, False):
        add_tree(connection, tree, test_file)

    for row in c.execute('SELECT * FROM trees'):
        print row

    for row2 in c.execute('SELECT * FROM tree_nodes'):
        print row2


    print

    create_result_tree_table(connection)
    create_result_tree_node_table(connection)
    for tree in conll_parse.parse_conll_corpus(test_file_modified, False):
        add_result_tree(connection, tree, corpus, experiment, 1, 0.142)

    for row3 in c.execute('SELECT  * FROM result_tree_nodes'):
        print row3, type(row3[0]).__name__

    print

    print experiment, type(experiment).__name__

    for row4 in c.execute('''SELECT * FROM result_trees INNER JOIN result_tree_nodes ON result_trees.rt_id = result_tree_nodes.rt_id WHERE exp_id = ?''', (experiment,)):
        print row4

    connection.close()

dbtest()