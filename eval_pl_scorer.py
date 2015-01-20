__author__ = 'kilian'

import experiment_database
import conll_parse
import dependency_experiments_db
import os
import subprocess
import re

hypothesis_prefix = 'examples/sys-output'
gold_prefix = 'examples/gold-output'
eval_pl = 'util/eval.pl'


def eval_pl_scores(connection, corpus, experiment, filter = []):
    """
    :param connection: database connection
    :param corpus:  path to the gold standard corpus (CoNLL)
    :type corpus: str
    :param experiment: id of the experiment in the database
    :type experiment: int
    :return: labeled attachment score, unlabeled attachment score, label accuracy
    :rtype: float, float, float
    """
    test_file_path = hypothesis_test_path(hypothesis_prefix, experiment)
    if not filter:
        gold_file_path = corpus
    else:
        gold_file_path = hypothesis_test_path(gold_prefix, experiment)

    trees = dependency_experiments_db.parse_conll_corpus(corpus, False)

    # Remove file if exists
    try:
        os.remove(test_file_path)
    except OSError:
        pass

    if filter:
        gold_CoNLL_strings = []

        try:
            os.remove(gold_file_path)
        except OSError:
            pass

    CoNLL_strings = []
    recognised = 0

    for tree in trees:
        tree_name = tree.sent_label()
        tree_id = experiment_database.query_tree_id(connection, corpus, tree_name)
        assert(tree_id)
        if not filter or tree_id in filter:
            CoNLL_strings.append(CoNLL_string_for_tree(connection, tree_id, experiment))
            if filter:
                gold_CoNLL_strings.append(conll_parse.tree_to_conll_str(tree))

    CoNLL_strings.append('')
    test_file = open(test_file_path, 'a+')
    test_file.write('\n\n'.join(CoNLL_strings))
    test_file.close()

    if filter:
        gold_CoNLL_strings.append('')
        gold_file = open(gold_file_path, 'a+')
        gold_file.write('\n\n'.join(gold_CoNLL_strings))
        gold_file.close()

    eval_pl_call_strings = ["-g {!s}".format(gold_file_path), "-s {!s}".format(test_file_path), "-q"]
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
            las = float(m.group(1)) / 100
        m = re.search(r'^\s*Unlabeled\s*attachment\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
        if m:
            uas = float(m.group(1)) / 100
        m = re.search(r'^\s*Label\s*accuracy\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%$', line)
        if m:
            la = float(m.group(1)) / 100
    return las, uas, la


def hypothesis_test_path(prefix, experiment):
    """
    :param prefix: common prefix for system output
    :param experiment: experiment id in database
    :return: path of system output file
    """
    return '{:s}-{:d}.conll'.format(prefix, experiment)


def CoNLL_string_for_tree(connection, tree_id_in_db, experiment):
    """
    :param connection: database connection
    :return: (multiline) string with system output for tree in CoNLL format
    Retrieves the system output for a test tree in some experiment in the database.
    If none exists, a fallback strategy is used (hidden in the database module).
    """
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
    connection = experiment_database.initalize_database(db_file)
    common = experiment_database.common_recognised_sentences(connection, [experiment])
    las, uas, la = eval_pl_scores(connection, corpus, experiment, common)
    experiment_database.finalize_database(connection)
    print "Labeled attachment score {:.2f}".format(las)
    print "Unlabeled attachment score {:.2f}".format(uas)
    print "Label accuracy {:.2f}".format(la)
