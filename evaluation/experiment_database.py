__author__ = 'kilian'

import sqlite3
import re
import time
import sys

from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import CoNLLToken
from grammar.LCFRS.lcfrs import LCFRS
from corpora import conll_parse

dbfile = 'examples/example.db'
test_file = 'examples/Dependency_Corpus.conll'
test_file_modified = 'examples/Dependency_Corpus_modified.conll'
sampledb = '/home/kilian/sampledb.db'


def create_experiment_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS experiments ( e_id INTEGER PRIMARY KEY AUTOINCREMENT
                                                              , term_label TEXT
                                                              , nont_label TEXT
                                                              , rec_par TEXT
                                                              , ignore_punctuation BOOLEAN
                                                              , training_corpus TEXT
                                                              , test_corpus TEXT
                                                              , started time
                                                              , cpu_time time)''')
    connection.commit()


def create_tree_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS trees (t_id INTEGER PRIMARY KEY AUTOINCREMENT
                                                        , corpus TEXT
                                                        , name TEXT
                                                        , length INTEGER
                                                        , gaps INTEGER
                                                        , UNIQUE(corpus, name))''')
    # cursor.execute('''CREATE UNIQUE INDEX IF NOT EXISTS tree_idx ON trees(corpus, name)''')
    connection.commit()


def create_result_tree_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS result_trees (rt_id INTEGER PRIMARY KEY AUTOINCREMENT
                                                              , t_id INTEGER
                                                              , exp_id INTEGER
                                                              , k_best INTEGER
                                                              , score DOUBLE
                                                              , parse_time time
                                                              , status TEXT
                                                              , UNIQUE(t_id, exp_id, k_best))''')
    # cursor.execute('''CREATE UNIQUE INDEX IF NOT EXISTS tree_node_idx ON tree_nodes(t_id, sent_position)''')
    connection.commit()


def create_tree_node_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS tree_nodes (t_id INTEGER
                                                            , sent_position INTEGER
                                                            , label TEXT
                                                            , pos TEXT
                                                            , deprel TEXT
                                                            , head INTEGER
                                                            , UNIQUE(t_id, sent_position))''')
    # cursor.execute('''CREATE UNIQUE INDEX IF NOT EXISTS tree_node_idx ON tree_nodes(t_id, sent_position)''')
    connection.commit()


def create_result_tree_node_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS result_tree_nodes (rt_id INTEGER
                                                                   , sent_position INTEGER
                                                                   , deprel TEXT
                                                                   , head INTEGER
                                                                   , UNIQUE(rt_id, sent_position))''')
    connection.commit()


def create_grammar_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS grammar (g_id INTEGER PRIMARY KEY AUTOINCREMENT
                                                          , experiment INTEGER
                                                          , nonterminals INTEGER
                                                          , rules INTEGER
                                                          , size INTEGER
                                                          , UNIQUE(experiment))''')
    connection.commit()


def create_fanouts_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS fanouts (g_id INTEGER
                                                         , fanout INTEGER
                                                         , nonterminals INTEGER
                                                         , UNIQUE(g_id, fanout))''')
    connection.commit()


def add_grammar(connection, grammar, experiment):
    """
    :type grammar: LCFRS
    :type experiment: int
    """
    nont = len(grammar.nonts())
    rules = len(grammar.rules())
    size = grammar.size()
    cursor = connection.cursor()
    # print experiment, nont, rules, size
    cursor.execute('''INSERT INTO grammar VALUES (?, ?, ?, ?, ?)''', (None, experiment, nont, rules, size))
    g_id = cursor.lastrowid

    fanout_nonterminals = {}
    for nont in grammar.nonts():
        fanout = grammar.fanout(nont)
        fanout_nonterminals[fanout] = fanout_nonterminals.get(fanout, 0) + 1
    connection.commit()

    for fanout in fanout_nonterminals.keys():
        nont = fanout_nonterminals[fanout]
        # print g_id, fanout, nont
        cursor.execute('''INSERT INTO fanouts VALUES (?, ?, ?)''', (g_id, fanout, nont))
    connection.commit()


def add_experiment(connection, term_label, nont_label, rec_par, ignore_punctuation, training_corpus, test_corpus,
                   started, cpu_time):
    """
    :type connection: sqlite3.Connection
    :param term_label:
    :param nont_label:
    :param rec_par:
    :param ignore_punctuation: ignore punctuation in the grammar
    :type ignore_punctuation: bool
    :param training_corpus: training corpus path
    :param test_corpus: test corpus path
    :param started: start time
    :param cpu_time: total cpu time for parsing
    :return: experiment id
    :rtype: int
    """
    cursor = connection.cursor()
    cursor.execute('''INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
        None, term_label, nont_label, rec_par, ignore_punctuation, training_corpus, test_corpus, started, cpu_time))
    experiment = cursor.lastrowid
    connection.commit()
    return experiment


def set_experiment_test_corpus(connection, exp_id, test_corpus):
    cursor = connection.cursor()
    cursor.execute('UPDATE experiments SET test_corpus =? WHERE e_id = ?', (test_corpus, exp_id,))
    connection.commit()


def add_tree(connection, tree, corpus):
    """
    :param connection:
    :type tree: HybridTree
    :type corpus: str
    :return: tree_id
    :rtype: int
    """

    cursor = connection.cursor()
    for row in cursor.execute('''SELECT EXISTS (SELECT * FROM trees WHERE corpus = ? AND name = ?)''',
                              (corpus, tree.sent_label())):
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
        if id in tree.root:
            head = 0
        else:
            head = tree.node_index_full(tree.parent(id)) + 1
        token = tree.node_token(id)
        cursor.execute('''INSERT INTO tree_nodes VALUES (?, ?, ?, ?, ?, ?)''', (tree_id
                                                                                , tree.node_index_full(id) + 1
                                                                                , token.form()
                                                                                , token.pos()
                                                                                , token.deprel()
                                                                                , head))

    connection.commit()
    return tree_id


def add_result_tree(connection, tree, corpus, experiment, k_best, score, parse_time, status, root_default=None,
                    disconnect_default=None):
    """

    :type connection: Connection
    :type disconnect_default: str
    :type root_default: str
    :type status: str
    :type parse_time: float
    :type score: float
    :type k_best: int
    :type experiment: int
    :type tree: HybridTree
    :type corpus: str
    """

    cursor = connection.cursor()
    tree_id = None
    for row in cursor.execute('''SELECT t_id FROM trees WHERE corpus = ? AND name = ?''', (corpus, tree.sent_label())):
        tree_id = row[0]
    if tree_id is None:
        assert "tree not found"

    # unique tree key
    cursor.execute('INSERT INTO result_trees VALUES (?, ?, ?, ?, ?, ?, ?)', (None
                                                                             , tree_id
                                                                             , experiment
                                                                             , k_best
                                                                             , score
                                                                             , parse_time
                                                                             , status))
    result_tree_id = cursor.lastrowid

    for id in tree.full_yield():
        # set root head
        if id in tree.root:
            head = 0
            if root_default:
                deprel = root_default
            else:
                deprel = tree.node_token(id).deprel()
        # connect disconnected nodes to root
        elif tree.disconnected(id):
            head = tree.node_index_full(tree.root[0]) + 1
            if disconnect_default:
                deprel = disconnect_default
            else:
                deprel = tree.node_token(id).form()
        else:
            head = tree.node_index_full(tree.parent(id)) + 1
            deprel = tree.node_token(id).deprel()
        cursor.execute('''INSERT INTO result_tree_nodes VALUES (?, ?, ?, ?)''', (result_tree_id
                                                                                 , tree.node_index_full(id) + 1
                                                                                 , deprel
                                                                                 , head))
    connection.commit()


def list_experiments(connection):
    cursor = connection.cursor()
    rows = cursor.execute('''SELECT * FROM experiments''', ()).fetchall()
    return rows


def query_tree_id(connection, corpus, name):
    cursor = connection.cursor()

    rows = cursor.execute('''SELECT t_id FROM trees WHERE corpus = ? AND name = ?''', (corpus, name)).fetchall()

    # There should be at most one entry for every name and corpus.
    assert (len(rows) <= 1)

    if rows:
        return rows[0][0]
    else:
        return None


def query_result_tree(connection, exp, tree_id):
    """
    :param connection:
    :param exp:
    :param tree_id:
    :rtype: str, HybridTree
    :return:
    """
    cursor = connection.cursor()
    result_tree_ids = cursor.execute('''SELECT rt_id, status FROM result_trees WHERE exp_id = ? AND t_id = ?''',
                                     (exp, tree_id)).fetchall()

    # parse:
    if result_tree_ids:
        assert (len(result_tree_ids) == 1)
        result_tree_id, status = result_tree_ids[0]
        if status in ["parse", "fallback"]:
            name = cursor.execute('''SELECT name FROM trees WHERE t_id = ?''', (tree_id,)).fetchall()[0][0]
            tree_nodes = cursor.execute(
                (
                    ' SELECT tree_nodes.sent_position, label, pos, result_tree_nodes.head, result_tree_nodes.deprel FROM result_tree_nodes\n'
                    '                JOIN result_trees\n'
                    '                  ON result_tree_nodes.rt_id = result_trees.rt_id\n'
                    '                JOIN tree_nodes\n'
                    '                  ON result_trees.t_id = tree_nodes.t_id\n'
                    '                  AND result_tree_nodes.sent_position = tree_nodes.sent_position\n'
                    '                WHERE result_tree_nodes.rt_id = ?'
                ), (result_tree_id,))
            tree = HybridTree(name)
            for i, label, pos, head, deprel in tree_nodes:
                if deprel is None:
                    deprel = 'UNKNOWN'
                token = CoNLLToken(label, '_', pos, pos, '_', deprel)
                tree.add_node(str(i), token, True, True)
                if head == 0:
                    tree.add_to_root(str(i))
                else:
                    tree.add_child(str(head), str(i))
            assert tree.root is not []
            return status, tree
    # legacy: no entry found
    else:
        status = "simple_fallback"

    # Create a left branching tree without labels as default strategy
    tree_nodes = cursor.execute(
        ''' SELECT tree_nodes.sent_position, label, pos FROM tree_nodes
        WHERE tree_nodes.t_id = ?''', (tree_id,)).fetchall()

    left_branch = lambda x: x - 1
    right_branch = lambda x: x + 1
    strategy = left_branch

    length = len(tree_nodes)
    tree = HybridTree()
    for i, label, pos in tree_nodes:
        token = CoNLLToken(label, '_', pos, pos, '_', '_')
        tree.add_node(str(i), token, True, True)
        parent = strategy(i)
        if (parent == 0 and strategy == left_branch) or (parent == length + 1 and strategy == right_branch):
            tree.add_to_root(str(i))
        else:
            tree.add_child(str(parent), str(i))
    assert tree.root is not []
    return status, tree


def openDatabase(file):
    connection = sqlite3.connect(file)
    return connection


def dbtest():
    connection = openDatabase(dbfile)
    connection.text_factory = str

    create_experiment_table(connection)

    training_corpus = test_file
    test_corpus = test_file
    experiment = add_experiment(connection, 'term_pos', 'child_pos', 'direct_extraction', False, training_corpus,
                                test_corpus, time.time(), None)

    c = connection.cursor()
    for row in c.execute('SELECT * FROM experiments'):
        print row

    create_tree_table(connection)
    create_tree_node_table(connection)

    for tree in conll_parse.parse_conll_corpus(test_corpus, False):
        add_tree(connection, tree, test_corpus)

    for row in c.execute('SELECT * FROM trees'):
        print row

    for row2 in c.execute('SELECT * FROM tree_nodes'):
        print row2

    print

    create_result_tree_table(connection)
    create_result_tree_node_table(connection)
    time_stamp = time.clock()
    for tree in conll_parse.parse_conll_corpus(test_file_modified, False):
        add_result_tree(connection, tree, test_corpus, experiment, 1, 0.142, time.clock() - time_stamp, "parse")
        time_stamp = time.clock()

    for row3 in c.execute('SELECT  * FROM result_tree_nodes'):
        print row3, type(row3[0]).__name__

    print

    print experiment, type(experiment).__name__

    for row4 in c.execute(
            '''SELECT * FROM result_trees INNER JOIN result_tree_nodes ON result_trees.rt_id = result_tree_nodes.rt_id WHERE exp_id = ?''',
            (experiment,)):
        print row4

    connection.close()


# dbtest()

def initialize_database(dbfile):
    """
    Opens existing or creates new experiment database and returns Connection object to it.
    :param dbfile:
    :type dbfile: str
    :return: connection to database
    :rtype: Connection
    """
    connection = openDatabase(dbfile)
    connection.text_factory = str
    # Faster concurrent read/write access
    connection.cursor().execute('PRAGMA journal_mode=WAL')

    create_experiment_table(connection)
    create_tree_table(connection)
    create_tree_node_table(connection)
    create_result_tree_table(connection)
    create_result_tree_node_table(connection)
    create_grammar_table(connection)
    create_fanouts_table(connection)

    return connection


def scores_and_parse_time(connection, ids, exp):
    UAS_a, LAS_a, UAS_t, LAS_t, LEN = 0, 0, 0, 0, 0
    time = 0
    for id in ids:
        time = time + parsetime(connection, id, exp)

        c, l, uas_a = uas(connection, id, exp)
        UAS_a += uas_a
        LEN = LEN + l
        UAS_t = UAS_t + c

        cl, _, las_a = las(connection, id, exp)
        LAS_a += las_a
        LAS_t = LAS_t + cl
    UAS_a /= len(ids)
    LAS_a /= len(ids)
    UAS_t = 1.0 * UAS_t / LEN
    LAS_t = 1.0 * LAS_t / LEN
    return UAS_a, LAS_a, UAS_t, LAS_t, time


def fanout(fanouts, f):
    for fi, ni in fanouts:
        if f == fi:
            return "{:,}".format(ni)
    return 0


def punct(p):
    if p == 1:
        return 'ignore'
    elif p == 0:
        return 'consider'
    else:
        assert ()


def common_recognised_sentences(connection, experiments, max_length=sys.maxint, ignore_punctuation = False):
    statement = '''
      SELECT trees.t_id
      FROM trees INNER JOIN result_trees
      ON trees.t_id = result_trees.t_id
      WHERE result_trees.exp_id IN ({0})
      AND status = "parse"
      GROUP BY trees.t_id
      HAVING count (DISTINCT result_trees.exp_id) = ?
    '''.format(', '.join('?' * len(experiments)))
    # statement = "SELECT * FROM tab WHERE obj IN ({0})".format(', '.join(['?' * len(list_of_vars)]))
    # print statement
    cursor = connection.cursor()
    ids = cursor.execute(statement, experiments + [len(experiments)]).fetchall()
    # print ids
    return filter_by_length(connection, map(lambda x: x[0], ids), max_length, ignore_punctuation)


def filter_by_length(connection, ids, max_length, ignore_punctuation):
    cursor = connection.cursor()
    ids2 = []
    for id in ids:
        statement = '''
        SELECT tree_nodes.label
        FROM tree_nodes
        WHERE tree_nodes.t_id = ?
        '''
        # print id, statement
        nodes = cursor.execute(statement, [id]).fetchall()
        if (not ignore_punctuation and len(nodes) <= max_length) \
                or (ignore_punctuation and len(
                    filter(lambda x: not conll_parse.is_punctuation(x[0]), nodes)) <= max_length):
            ids2.append(id)
    return ids2


def test_sentences_length_lesseq_than(connection, test_corpus, length, ignore_punctuation):
    cursor = connection.cursor()
    ids = cursor.execute('SELECT t_id FROM trees WHERE corpus = ?',
                       (test_corpus,)).fetchall()
    return len(filter_by_length(connection, map(lambda x: x[0], ids), length, ignore_punctuation))


def all_recognised_sentences_lesseq_than(connection, exp_id, length, test_corpus, ignore_punctuation):
    cursor = connection.cursor()
    ids = cursor.execute('''
    SELECT trees.t_id
    FROM trees JOIN result_trees
      ON trees.t_id = result_trees.t_id
    WHERE trees.corpus = ?
      AND status = "parse"
      AND result_trees.exp_id = ?''', (test_corpus, exp_id,)).fetchall()
    return len(filter_by_length(connection, map(lambda x: x[0], ids), length, ignore_punctuation))


def recognised_sentences_lesseq_than(connection, exp_id, length, test_corpus, ignore_punctuation):
    cursor = connection.cursor()
    ids = cursor.execute('''
    SELECT trees.t_id
    FROM trees JOIN result_trees
      ON trees.t_id = result_trees.t_id
    WHERE trees.corpus = ?
      AND status = "parse"
      AND result_trees.exp_id = ?''', (test_corpus, exp_id,)).fetchall()

    return filter_by_length(connection, map(lambda x: x[0], ids), length, ignore_punctuation)


def parse_time_trees_lesseq_than(connection, exp_id, length, test_corpus, ignore_punctuation):
    cursor = connection.cursor()
    total_time = 0

    ids = cursor.execute('''
    SELECT trees.t_id
    FROM trees JOIN result_trees
      ON trees.t_id = result_trees.t_id
    WHERE trees.corpus = ?
      AND result_trees.exp_id = ?''', (test_corpus, exp_id,)).fetchall()

    ids = filter_by_length(connection, map(lambda x: x[0], ids), length, ignore_punctuation)

    for tree_id in ids:
        time = cursor.execute('''
        SELECT result_trees.parse_time
        FROM trees JOIN result_trees
          ON trees.t_id = result_trees.t_id
        WHERE trees.corpus = ?
          AND result_trees.exp_id = ?
          AND trees.t_id = ?''', (test_corpus, exp_id, tree_id, )).fetchone()[0]
        total_time += time
    return total_time


def percentify(value, precicion):
    p = value * 100
    return ("{:." + str(precicion) + "f}").format(p)


def nontlabelling_strategies(nont_labelling):
    if nont_labelling == 'strict_pos':
        return 'strict + POS'
    elif nont_labelling == 'child_pos':
        return 'child + POS'
    elif nont_labelling == 'child_pos_dep':
        return 'child + POS, leaf: dep'
    elif nont_labelling == 'strict_pos_dep':
        return 'strict + POS, leaf: dep'
    elif nont_labelling == 'strict_dep':
        return 'strict + DEPREL'
    elif nont_labelling == 'strict_pos_dep_overall':
        return 'strict + POS + DEPREL'
    elif nont_labelling == 'child_pos_dep_overall':
        return 'child + POS + DEPREL'
    elif nont_labelling == 'child_dep':
        return 'child + DEPREL'
    else:
        return nont_labelling.replace('_', '-')


def recpac_stategies(rec_par):
    if rec_par == 'direct_extraction':
        return 'direct'
    match = re.search(r'^fanout_(\d+)$', rec_par)
    if match:
        return '$k = ' + str(match.group(1)) + '$'
    else:
        return rec_par.replace('_', '-').replace('branching', 'branch').replace('left', 'l').replace('right', 'r')


def uas(connection, tree_id, e_id):
    cursor = connection.cursor()
    try:
        correct, length = cursor.execute('''
        SELECT count(tree_nodes.head), trees.length
        FROM tree_nodes JOIN trees JOIN result_trees JOIN result_tree_nodes
        ON tree_nodes.t_id =  trees.t_id
            AND trees.t_id = result_trees.t_id
            AND result_trees.rt_id = result_tree_nodes.rt_id
            AND result_tree_nodes.sent_position = tree_nodes.sent_position
        WHERE result_trees.exp_id = ? AND trees.t_id = ?
            AND tree_nodes.head = result_tree_nodes.head
            --and tree_nodes.deprel = result_tree_nodes.deprel
        ''', (e_id, tree_id)).fetchone()
        return correct, length, 1.0 * correct / length
    except TypeError:
        try:
            incorrect, length = cursor.execute('''
            SELECT count(tree_nodes.head), trees.length
            FROM tree_nodes JOIN trees JOIN result_trees JOIN result_tree_nodes
            ON tree_nodes.t_id =  trees.t_id
                AND trees.t_id = result_trees.t_id
                AND result_trees.rt_id = result_tree_nodes.rt_id
                AND result_tree_nodes.sent_position = tree_nodes.sent_position
            WHERE result_trees.exp_id = ? AND trees.t_id = ?
                AND tree_nodes.head != result_tree_nodes.head
                --and tree_nodes.deprel = result_tree_nodes.deprel
            ''', (e_id, tree_id)).fetchone()
            if incorrect == length:
                return 0, length, 0
            else:
                # print "Error:", tree_id, e_id, incorrect, length
                assert ()
        except TypeError:
            print tree_id, e_id
            assert ()


def las(connection, tree_id, e_id):
    cursor = connection.cursor()
    try:
        correct, length = cursor.execute('''
        SELECT count(DISTINCT tree_nodes.sent_position), trees.length
        FROM tree_nodes JOIN trees JOIN result_trees JOIN result_tree_nodes
        ON tree_nodes.t_id =  trees.t_id
            AND trees.t_id = result_trees.t_id
            AND result_trees.rt_id = result_tree_nodes.rt_id
            AND result_tree_nodes.sent_position = tree_nodes.sent_position
        WHERE result_trees.exp_id = ? AND trees.t_id = ?
            AND tree_nodes.head = result_tree_nodes.head
            AND tree_nodes.deprel = result_tree_nodes.deprel
        ''', (e_id, tree_id)).fetchone()
        return correct, length, 1.0 * correct / length
    except TypeError:
        try:
            incorrect, length = cursor.execute('''
            SELECT count(tree_nodes.head), trees.length
            FROM tree_nodes JOIN trees JOIN result_trees JOIN result_tree_nodes
            ON tree_nodes.t_id =  trees.t_id
                AND trees.t_id = result_trees.t_id
                AND result_trees.rt_id = result_tree_nodes.rt_id
                AND result_tree_nodes.sent_position = tree_nodes.sent_position
            WHERE result_trees.exp_id = ? AND trees.t_id = ?
                AND (tree_nodes.head != result_tree_nodes.head
                OR tree_nodes.deprel != result_tree_nodes.deprel)
            ''', (e_id, tree_id)).fetchone()
            if incorrect == length:
                return 0, length, 0
            else:
                assert ()
        except TypeError:
            print tree_id, e_id
            assert ()


def parsetime(connection, tree_id, e_id):
    cursor = connection.cursor()
    t = cursor.execute('''
    SELECT parse_time FROM result_trees WHERE t_id = ? AND exp_id = ?
    ''', (tree_id, e_id)).fetchone()[0]
    return t


def finalize_database(connection):
    """
    :param connection:
    :type connection: sqlite3.Connection
    :return:
    """
    connection.close()


# result_table()
def no_parse_result(connection, tree_name, corpus, experiment, parse_time, message):
    cursor = connection.cursor()

    tree_id = None
    for row in cursor.execute('''SELECT t_id FROM trees WHERE corpus = ? AND name = ?''', (corpus, tree_name)):
        tree_id = row[0]
    if tree_id is None:
        assert "tree not found"

    # unique tree key
    cursor.execute('INSERT INTO result_trees VALUES (?, ?, ?, ?, ?, ?, ?)', (None
                                                                             , tree_id
                                                                             , experiment
                                                                             , None
                                                                             , None
                                                                             , parse_time
                                                                             , message))
    connection.commit()
