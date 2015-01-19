__author__ = 'kilian'

import experiment_database
import conll_parse
import dependency_experiments_db
import os
import subprocess
import re

hypothesis_prefix = 'examples/sys-output'
eval_pl = 'util/eval.pl'


def eval_pl_scores(db_file, corpus, experiment):
    """
    :param db_file: path to the sqlite database file
    :type db_file: str
    :param corpus:  path to the gold standard corpus (CoNLL)
    :type corpus: str
    :param experiment: id of the experiment in the database
    :type experiment: int
    :return: labeled attachment score, unlabeled attachment score, label accuracy
    :rtype: float, float, float
    """
    test_file_path = hypothesis_test_path(hypothesis_prefix, experiment)
    connection = experiment_database.initalize_database(db_file)

    trees = dependency_experiments_db.parse_conll_corpus(corpus, False)

    # Remove file if exists
    try:
        os.remove(test_file_path)
    except OSError:
        pass

    CoNLL_strings = []
    recognised = 0

    for tree in trees:
        tree_name = tree.sent_label()
        CoNLL_strings.append(CoNLL_string_for_tree(connection, corpus, tree_name, experiment))

    CoNLL_strings.append('')
    test_file = open(test_file_path, 'a+')
    test_file.write('\n\n'.join(CoNLL_strings))
    test_file.close()

    eval_pl_call_strings = ["-g {!s}".format(corpus), "-s {!s}".format(test_file_path), "-q"]
    p = subprocess.Popen(['perl', eval_pl] + eval_pl_call_strings, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    lines = out.split('\n')
    # print lines
    uas = 0.0
    las = 0.0
    la = 0.0
    for line in lines:
        m = re.search(r'^\s*Labeled\s*attachment\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
        if m:
            las = float(m.group(1))
        m = re.search(r'^\s*Unlabeled\s*attachment\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
        if m:
            uas = float(m.group(1))
        m = re.search(r'^\s*Label\s*accuracy\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
        if m:
            la = float(m.group(1))
    return las, uas, la


def hypothesis_test_path(prefix, experiment):
    """
    :param prefix: common prefix for system output
    :param experiment: experiment id in database
    :return: path of system output file
    """
    return '{:s}-{:d}.conll'.format(prefix, experiment)


def CoNLL_string_for_tree(connection, corpus, tree_name, experiment):
    """
    :param connection: database connection
    :param corpus: path to corpus
    :param tree_name: name of tree (in database)
    :param experiment: experiment id (in database)
    :return: (multiline) string with system output for tree in CoNLL format
    Retrieves the system output for a test tree in some experiment in the database.
    If none exists, a fallback strategy is used (hidden in the database module).
    """
    tree_id_in_db = experiment_database.query_tree_id(connection, corpus, tree_name)
    assert(tree_id_in_db)

    flag, hypothesis_tree = experiment_database.query_result_tree(connection, experiment, tree_id_in_db)

    return conll_parse.tree_to_conll_str(hypothesis_tree)

# Sample evaluation
if __name__ == '__main__':
    conll_test = '../dependency_conll/german/tiger/test/german_tiger_test.conll'
    conll_train = '../dependency_conll/german/tiger/train/german_tiger_train.conll'
    sample_db = 'examples/sampledb.db'
    db_file = sample_db
    corpus = conll_test
    experiment = 40
    las, uas, la = eval_pl_scores(db_file, corpus, experiment)
    print "Labeled attachment score {:.2f}".format(las)
    print "Unlabeled attachment score {:.2f}".format(uas)
    print "Label accuracy {:.2f}".format(la)
