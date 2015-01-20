__author__ = 'kilian'

import sqlite3
import time
from general_hybrid_tree import GeneralHybridTree
from lcfrs import LCFRS
import conll_parse
import sys

dbfile = 'examples/example.db'
test_file = 'examples/Dependency_Corpus.conll'
test_file_modified = 'examples/Dependency_Corpus_modified.conll'
sampledb = '/home/kilian/sampledb.db'

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
    cursor.execute('''CREATE TABLE IF NOT EXISTS result_trees (rt_id integer primary key autoincrement, t_id integer, exp_id integer, k_best integer, score double, parse_time time, UNIQUE(t_id, exp_id, k_best))''')
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
    cursor.execute('''CREATE TABLE IF NOT EXISTS grammar (g_id integer primary key autoincrement, experiment integer, nonterminals integer, rules integer, size integer , UNIQUE(experiment))''')
    connection.commit()

def create_fanouts_table(connection):
    # Create Table
    cursor = connection.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS fanouts (g_id integer, fanout integer, nonterminals integer, UNIQUE(g_id, fanout))''')
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


def add_experiment(connection, term_label, nont_label, rec_par, ignore_punctuation, corpus, started, cpu_time):
    """
    :type connection: Connection
    :param term_label:
    :param nont_label:
    :param rec_par:
    :param ignore_punctuation: ignore punctuation in the grammar
    :type ignore_punctuation: bool
    :param corpus: corpus path
    :param started: start time
    :param cpu_time: total cpu time for parsing
    :return: experiment id
    :rtype: int
    """
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


def add_result_tree(connection, tree, corpus, experiment, k_best, score, parse_time):
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
    cursor.execute('''INSERT INTO result_trees VALUES (?, ?, ?, ?, ?, ?)''', ( None
                                                                        , tree_id
                                                                        , experiment
                                                                        , k_best
                                                                        , score
                                                                        , parse_time))
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


def list_experiments(connection):
    cursor = connection.cursor()
    rows = cursor.execute('''SELECT * FROM experiments''',()).fetchall()
    return rows



def query_tree_id(connection, corpus, name):
    cursor = connection.cursor()

    rows = cursor.execute('''SELECT t_id FROM trees WHERE corpus = ? AND name = ?''', ( corpus, name)).fetchall()

    # There should be at most one entry for every name and corpus.
    assert(len(rows) <= 1)

    if rows:
        return rows[0][0]
    else:
        return None

def query_result_tree(connection, exp, tree_id):
    cursor = connection.cursor()
    result_tree_ids = cursor.execute('''select rt_id from result_trees where exp_id = ? and t_id = ?''', (exp, tree_id)).fetchall()

    # parse:
    if result_tree_ids:
        assert(len(result_tree_ids) == 1)
        result_tree_id = result_tree_ids[0][0]
        tree_nodes = cursor.execute(
        ''' select tree_nodes.sent_position, label, pos, result_tree_nodes.head, result_tree_nodes.deprel from result_tree_nodes
	        join result_trees
		      on result_tree_nodes.rt_id = result_trees.rt_id
	        join tree_nodes
		      on result_trees.t_id = tree_nodes.t_id
		      and result_tree_nodes.sent_position = tree_nodes.sent_position
            where result_tree_nodes.rt_id = ?''', (result_tree_id,))
        tree = GeneralHybridTree()
        for i, label, pos, head, deprel in tree_nodes:
            tree.add_node(str(i), label, pos, True, True)
            tree.set_dep_label(str(i), deprel)
            if head == 0:
                tree.set_root(str(i))
            else:
                tree.add_child(str(head), str(i))
        assert(tree.rooted())
        return ('parse', tree)

    # fallback
    else:
        tree_nodes = cursor.execute(
        ''' select tree_nodes.sent_position, label, pos from tree_nodes
            where tree_nodes.t_id = ?''', (tree_id,)).fetchall()

        left_branch = lambda x: x-1
        right_branch = lambda x: x+1
        strategy = right_branch

        length = len(tree_nodes)
        tree = GeneralHybridTree()
        for i, label, pos in tree_nodes:
            tree.add_node(str(i), label, pos, True, True)
            tree.set_dep_label(str(i), '_')
            parent = strategy(i)
            if (parent == 0 and strategy == left_branch) or (parent == length + 1 and strategy == right_branch):
                tree.set_root(str(i))
            else:
                tree.add_child(str(parent), str(i))
        assert(tree.rooted())
        return ('fallback', tree)

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
    time_stamp = time.clock()
    for tree in conll_parse.parse_conll_corpus(test_file_modified, False):
        add_result_tree(connection, tree, corpus, experiment, 1, 0.142, time.clock() - time_stamp)
        time_stamp = time.clock()

    for row3 in c.execute('SELECT  * FROM result_tree_nodes'):
        print row3, type(row3[0]).__name__

    print

    print experiment, type(experiment).__name__

    for row4 in c.execute('''SELECT * FROM result_trees INNER JOIN result_tree_nodes ON result_trees.rt_id = result_tree_nodes.rt_id WHERE exp_id = ?''', (experiment,)):
        print row4

    connection.close()

# dbtest()

def initalize_database(dbfile):
    """
    Opens existing or creates new experiment database and returns Connection object to it.
    :param dbfile:
    :type dbfile: str
    :return: connection to database
    :rtype: Connection
    """
    connection = openDatabase(dbfile)
    connection.text_factory = str

    create_experiment_table(connection)
    create_tree_table(connection)
    create_tree_node_table(connection)
    create_result_tree_table(connection)
    create_result_tree_node_table(connection)
    create_grammar_table(connection)
    create_fanouts_table(connection)

    return connection

def create_latex_table_from_database(connection, experiments, pipe = sys.stdout):
    columns_style = {}
    table_columns = ['nont_labelling', 'rec_par', 'training_corpus', 'n_nonterminals', 'n_rules', 'fanout'
        , 'f1', 'f2', 'f3', 'f4', 'f5', 'test_total', 'UAS^c_avg', 'LAS^c_avg', 'LAS^c_t', 'UAS^c_t'
        , 'fail', 'UAS_avg', 'LAS_avg', 'UAS_t', 'LAS_t', 'n_gaps_test', 'n_gaps_gold', 'parse_time', 'punc']
    selected_columns = ['punc', 'nont_labelling', 'rec_par', 'f1', 'f2', 'f3', 'f4', 'f5'
        #, 'fail'
        , 'UAS_avg', 'LAS_avg', 'UAS_t', 'LAS_t', 'parse_time', 'fail', 'UAS^c_avg', 'LAS^c_avg', 'UAS^c_t', 'LAS^c_t'
        #, 'n_gaps_test', 'parse_time'
        ]
    header = {}
    header['nont_labelling'] = 'nont.~lab.'
    columns_style['nont_labelling'] = 'l'
    header['rec_par'] = 'extraction'
    columns_style['rec_par'] ='l'
    header['training_corpus'] = 'training sent.'
    columns_style['training_corpus'] = 'r'
    header['punc'] = 'punct.'
    columns_style['punc'] = 'l'
    header['n_nonterminals'] = 'nont.'
    columns_style['n_nonterminals'] = 'r'
    header['n_rules'] = 'rules'
    columns_style['n_rules'] ='r'
    header['fanout'] = 'fanout'
    columns_style['fanout'] = 'r'
    for i in range(1,6,1):
        header['f'+str(i)] = 'f '+str(i)
        columns_style['f'+str(i)] = 'r'
    header['test_total'] = 'test sent.'
    columns_style['test_total'] = 'r'
    header['test_succ'] = 'succ'
    columns_style['test_succ'] = 'r'
    header['fail'] = 'fail'
    columns_style['fail'] = 'r'
    header['UAS_avg'] = '$UAS_a$'
    columns_style['UAS_avg'] = 'r'
    header['LAS_avg'] = '$LAS_a$'
    columns_style['LAS_avg'] = 'r'
    header['UAS_t'] = '$UAS_t$'
    columns_style['UAS_t'] = 'r'
    header['LAS_t'] = '$LAS_t$'
    columns_style['LAS_t'] = 'r'
    header['UAS^c_avg'] = '$UAS^c_a$'
    columns_style['UAS^c_avg'] = 'r'
    header['LAS^c_avg'] = '$LAS^c_a$'
    columns_style['LAS^c_avg'] = 'r'
    header['UAS^c_t'] = '$UAS^c_t$'
    columns_style['UAS^c_t'] = 'r'
    header['LAS^c_t'] = '$LAS^c_t$'
    columns_style['LAS^c_t'] = 'r'
    header['n_gaps_test'] = '\\# gaps (test)'
    columns_style['n_gaps_test'] = 'r'
    header['n_gaps_gold'] = '\\# gaps (gold)'
    columns_style['n_gaps_gold'] = 'r'
    header['parse_time']  = 'time (s)'
    columns_style['parse_time'] = 'r'

    common_results = common_recognised_sentences(connection, experiments)

    pipe.write('''
    \\documentclass[a4paper,10pt, fullpage]{scrartcl}
    \\usepackage[utf8]{inputenc}
    \\usepackage{booktabs}
    \\usepackage[landscape, left = 1em, right = 1cm, top = 1cm, bottom = 1cm]{geometry}
    %opening
    \\author{Kilian Gebhardt}

    \\begin{document}
    \\centering
    \\thispagestyle{empty}
    \\begin{table}
    \\centering
    \n''')
    pipe.write('\\begin{tabular}{' + ''.join(columns_style[id] for id in selected_columns) + '}\n')
    pipe.write('\t \multicolumn{8}{l}{Intersection of recognised sentences of length $\leq$ 20: ' + str(len(common_results)) + ' / ' + str(test_sentences_length_lesseq_than(connection,20))+ '}\\\\\n')
    pipe.write('\t\\toprule\n')
    pipe.write('\t' + ' & '.join([header[id] for id in selected_columns]) + '\\\\\n')
    for exp in experiments:
        line = compute_line(connection, common_results, exp)
        pipe.write('\t' + ' & '.join([str(line[id]) for id in selected_columns]) + '\\\\\n')
    pipe.write('\t\\bottomrule\n')
    pipe.write('\\end{tabular}\n')
    pipe.write('''
    \\end{table}

\\end{document}
    \n''')

def compute_line(connection, ids, exp):
    line = {}

    cursor = connection.cursor()
    experiment = cursor.execute('select nont_label, rec_par, corpus, ignore_punctuation from experiments where e_id = ?', (exp, )).fetchone()
    g_id, nont, rules    = cursor.execute('select g_id, nonterminals, rules from grammar where experiment = ?', (exp,)).fetchone()
    fanouts = cursor.execute('select fanout, nonterminals from fanouts where g_id = ?', (g_id, )).fetchall()

    line['nont_labelling'] = nontlabelling_strategies(experiment[0])
    line['rec_par'] = recpac_stategies(experiment[1])
    line['training_corpus'] = experiment[2]
    line['punc'] = punct(experiment[3])
    line['n_nonterminals'] = nont
    line['n_rules'] = rules
    # line['fanout'] = 'fanout'
    line['f1'] = fanout(fanouts, 1)
    line['f2'] = fanout(fanouts, 2)
    line['f3'] = fanout(fanouts, 3)
    line['f4'] = fanout(fanouts, 4)
    line['f5'] = fanout(fanouts, 5)

    UAS_a, LAS_a, UAS_t, LAS_t, LEN = 0, 0, 0, 0, 0
    time = 0
    for id in ids:
        time = time + parsetime(connection, id, exp)

        c, l , uas_a = uas(connection, id, exp)
        UAS_a = UAS_a + uas_a
        LEN = LEN + l
        UAS_t = UAS_t + c

        cl, _, las_a = las(connection, id, exp)
        LAS_a = LAS_a + las_a
        LAS_t = LAS_t + cl
    UAS_a = UAS_a / len(ids)
    LAS_a = LAS_a / len(ids)
    UAS_t = 1.0 * UAS_t / LEN
    LAS_t = 1.0 * LAS_t / LEN

    recogn_ids = recognised_sentences_lesseq_than(connection, exp, 20)
    UAS_c_a, LAS_c_a, UAS_c_t, LAS_c_t, LEN_c = 0, 0, 0, 0, 0
    for id in recogn_ids:
        c, l , uas_a = uas(connection, id, exp)
        UAS_c_a = UAS_c_a + uas_a
        LEN_c = LEN_c + l
        UAS_c_t = UAS_c_t + c

        cl, _, las_a = las(connection, id, exp)
        LAS_c_a += las_a
        LAS_c_t += cl
    UAS_c_a = UAS_c_a / len(recogn_ids)
    LAS_c_a = LAS_c_a / len(recogn_ids)
    UAS_c_t = 1.0 * UAS_c_t / LEN_c
    LAS_c_t = 1.0 * LAS_c_t / LEN_c

    # line['test_total'] = 'test sent.'
    # line['test_succ'] = 'succ'
    line['fail'] = test_sentences_length_lesseq_than(connection, 20) - all_recognised_sentences_lesseq_than(connection, exp, 20)
    precicion = 2
    line['UAS_avg'] = percentify(UAS_a, precicion)
    line['LAS_avg'] = percentify(LAS_a, precicion)
    line['UAS_t'] = percentify(UAS_t, precicion)
    line['LAS_t'] = percentify(LAS_t, precicion)
    line['UAS^c_avg'] = percentify(UAS_c_a, precicion)
    line['LAS^c_avg'] = percentify(LAS_c_a, precicion)
    line['UAS^c_t'] = percentify(UAS_c_t, precicion)
    line['LAS^c_t'] = percentify(LAS_c_t, precicion)
    # line['n_gaps_test'] = '\\# gaps (test)'
    # line['n_gaps_gold'] = '\\# gaps (gold)'
    line['parse_time']  = "{:.0f}".format(time)
    return line

def fanout(fanouts, f):
    for fi, ni in fanouts:
        if f == fi:
            return ni
    return 0

def punct(p):
    if p == 1:
        return 'ignore'
    elif p == 0:
        return 'consider'
    else:
        assert()

def common_recognised_sentences(connection, experiments):
    statement = '''
      select trees.t_id
      from trees inner join result_trees
      on trees.t_id = result_trees.t_id
      where result_trees.exp_id in ({0})
      group by trees.t_id
      having count (distinct result_trees.exp_id) = ?
    '''.format(', '.join('?' * len(experiments)))
    # statement = "SELECT * FROM tab WHERE obj IN ({0})".format(', '.join(['?' * len(list_of_vars)]))
    # print statement
    cursor = connection.cursor()
    ids = cursor.execute(statement, experiments + [len(experiments)]).fetchall()
    ids = map(lambda x: x[0], ids)
    # print ids
    return ids

def test_sentences_length_lesseq_than(connection, length):
    cursor = connection.cursor()
    number = cursor.execute('select count(t_id) from trees where corpus like "%test.conll" and length <= ?', (length, )).fetchone()[0]
    return number

def all_recognised_sentences_lesseq_than(connection, exp_id, length):
    cursor = connection.cursor()
    number = cursor.execute('''
    select count(trees.t_id)
    from trees join result_trees
      on trees.t_id = result_trees.t_id
    where trees.corpus like "%test.conll"
      and trees.length <= ?
      and result_trees.exp_id = ?''', (length, exp_id, )).fetchone()[0]
    return number

def recognised_sentences_lesseq_than(connection, exp_id, length):
    cursor = connection.cursor()
    ids = cursor.execute('''
    select trees.t_id
    from trees join result_trees
      on trees.t_id = result_trees.t_id
    where trees.corpus like "%test.conll"
      and trees.length <= ?
      and result_trees.exp_id = ?''', (length, exp_id, )).fetchall()
    ids = map(lambda x: x[0], ids)
    return ids

def percentify(value, precicion):
    p = value * 100
    return ("{:." + str(precicion) + "f}").format(p)

def nontlabelling_strategies(nont_labelling):
    if nont_labelling == 'strict_pos':
        return 'strict'
    elif nont_labelling == 'child_pos':
        return'child'
    elif nont_labelling == 'child_pos_dep':
        return 'child + dep'
    elif nont_labelling == 'strict_pos_dep':
        return 'strict + dep'

def recpac_stategies(rec_par):
    if rec_par == 'direct_extraction':
        return 'direct'
    else:
        return rec_par.replace('_', ' ').replace('branching','branch.')

def uas(connection, tree_id, e_id):
    cursor = connection.cursor()
    try:
        correct, length = cursor.execute('''
        select count(tree_nodes.head), trees.length
        from tree_nodes join trees join result_trees join result_tree_nodes
        on tree_nodes.t_id =  trees.t_id
            and trees.t_id = result_trees.t_id
            and result_trees.rt_id = result_tree_nodes.rt_id
            and result_tree_nodes.sent_position = tree_nodes.sent_position
        where result_trees.exp_id = ? and trees.t_id = ?
            and tree_nodes.head = result_tree_nodes.head
            --and tree_nodes.deprel = result_tree_nodes.deprel
        ''', (e_id, tree_id)).fetchone()
        return correct, length, 1.0 * correct/ length
    except TypeError:
        try:
            incorrect, length = cursor.execute('''
            select count(tree_nodes.head), trees.length
            from tree_nodes join trees join result_trees join result_tree_nodes
            on tree_nodes.t_id =  trees.t_id
                and trees.t_id = result_trees.t_id
                and result_trees.rt_id = result_tree_nodes.rt_id
                and result_tree_nodes.sent_position = tree_nodes.sent_position
            where result_trees.exp_id = ? and trees.t_id = ?
                and tree_nodes.head != result_tree_nodes.head
                --and tree_nodes.deprel = result_tree_nodes.deprel
            ''', (e_id, tree_id)).fetchone()
            if incorrect == length:
                return 0, length, 0
            else:
                assert()
        except TypeError:
            print tree_id, e_id
            assert()


def las(connection, tree_id, e_id):
    cursor = connection.cursor()
    try:
        correct, length = cursor.execute('''
        select count(distinct tree_nodes.sent_position), trees.length
        from tree_nodes join trees join result_trees join result_tree_nodes
        on tree_nodes.t_id =  trees.t_id
            and trees.t_id = result_trees.t_id
            and result_trees.rt_id = result_tree_nodes.rt_id
            and result_tree_nodes.sent_position = tree_nodes.sent_position
        where result_trees.exp_id = ? and trees.t_id = ?
            and tree_nodes.head = result_tree_nodes.head
            and tree_nodes.deprel = result_tree_nodes.deprel
        ''', (e_id, tree_id)).fetchone()
        return correct, length, 1.0 * correct/ length
    except TypeError:
        try:
            incorrect, length = cursor.execute('''
            select count(tree_nodes.head), trees.length
            from tree_nodes join trees join result_trees join result_tree_nodes
            on tree_nodes.t_id =  trees.t_id
                and trees.t_id = result_trees.t_id
                and result_trees.rt_id = result_tree_nodes.rt_id
                and result_tree_nodes.sent_position = tree_nodes.sent_position
            where result_trees.exp_id = ? and trees.t_id = ?
                and (tree_nodes.head != result_tree_nodes.head
                or tree_nodes.deprel != result_tree_nodes.deprel)
            ''', (e_id, tree_id)).fetchone()
            if incorrect == length:
                return 0, length, 0
            else:
                assert()
        except TypeError:
            print tree_id, e_id
            assert()

def parsetime(connection, tree_id, e_id):
    cursor = connection.cursor()
    t = cursor.execute('''
    select parse_time from result_trees where t_id = ? and exp_id = ?
    ''', (tree_id, e_id)).fetchone()[0]
    return t


def finalize_database(connection):
    """
    :param connection:
    :type connection: Connection
    :return:
    """
    connection.close()


def result_table():
	connection = openDatabase(sampledb)
    # create_latex_table_from_database(connection, range(1,41,1))
	create_latex_table_from_database(connection, [4,5,6,7,8,9,10,15,19,20,27,28,29,30,39,40])
	finalize_database(connection)

# result_table()
