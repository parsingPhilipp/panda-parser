__author__ = 'kilian'

conll_test = '../dependency_conll/german/tiger/test/german_tiger_test.conll'
conll_train = '../dependency_conll/german/tiger/train/german_tiger_train.conll'
sample_db = 'examples/sampledb.db'

import time
import sys
import re
import os
import gc

from hybridtree.general_hybrid_tree import GeneralHybridTree
import dependency.induction as d_i
import dependency.labeling as label
from parser.active.parsing import Parser as ActiveParser
from parser.naive.parsing import LCFRS_parser as NaiveParser
from corpora.conll_parse import parse_conll_corpus, score_cmp_dep_trees
import experiment_database
import resource_limits


def add_trees_to_db(path, connection, trees):
    """
    :param path: name (path) of the corpus in database
    :type path: str
    :param connection: connection to experiment database
    :type connection: Connection
    :param trees: a corpus of trees
    :type trees: __generator[GeneralHybridTree]
    :return: a corpus of trees
    :rtype: __generator[GeneralHybridTree]
    insert corpus of hybrid trees lazily into experiment database
    """
    for tree in trees:
        experiment_database.add_tree(connection, tree, path)
        yield tree


def disconnect_punctuation(trees):
    """
    :param trees: corpus of hybrid trees
    :type trees: __generator[GeneralHybridTree]
    :return: corpus of hybrid trees
    :rtype: __generator[GeneralHybridTree]
    lazily disconnect punctuation from each hybrid tree in a corpus of hybrid trees
    """
    for tree in trees:
        tree2 = GeneralHybridTree(tree.sent_label())
        tree2.set_root(tree.root())
        for id in tree.full_yield():
            label = tree.node_label(id)
            pos = tree.node_pos(id)
            deprel = tree.node_dep_label(id)
            if not re.search(r'^\$.*$', pos):
                parent = tree.parent(id)
                tree2.add_node(id, label, pos, True, True)
                tree2.add_child(parent, id)
            else:
                tree2.add_node(id, label, pos, True, False)

            tree2.set_dep_label(id, deprel)

        if tree2:
            # basic sanity checks
            if not tree2.rooted():
                continue
            elif tree2.n_nodes() != len(tree2.id_yield()) or len(tree2.nodes()) != len(tree2.full_yield()):
                continue
            yield tree2


def induce_grammar_from_file(path
                             , connection
                             , nont_labelling
                             , term_labelling
                             , recursive_partitioning
                             , limit=sys.maxint
                             , quiet=False
                             , start='START'
                             , ignore_punctuation=True
):
    """
    :param path: path to dependency corpus in CoNLL format
    :type path: str
    :param connection: database connection
    :type connection: Connection
    :param nont_labelling: GeneralHybridTree, Top_max, Bottom_max, Fanout -> str
    :type nont_labelling: AbstractLabeling
    :param term_labelling: GeneralHybridTree, NodeId -> str
    :type term_labelling: GeneralHybridTree, str -> str
    :param recursive_partitioning: GeneralHybridTree -> RecursivePartitioning
    :type recursive_partitioning: GeneralHybridTree -> [str], unknown
    :param limit: use only the first _limit_ trees for grammar induction
    :type limit: int
    :param quiet: status output
    :type quiet: bool
    :param start: set start nonterminal for grammar
    :type start: str
    :param ignore_punctuation: include punctuation into grammar
    :type ignore_punctuation: bool
    :rtype: LCFRS, int
    Extract an LCFRS/sDCP-Hybrid Grammar from a dependency corpus in CoNLL format.
    """

    experiment = experiment_database.add_experiment(connection
                                                    , term_labelling.func_name
                                                    , str(nont_labelling)
                                                    , recursive_partitioning.func_name
                                                    , ignore_punctuation
                                                    , path
                                                    , ''
                                                    , time.time()
                                                    , None)

    if not quiet:
        print 'Inducing grammar'
        print 'file: ' + path
        print 'Nonterminal labelling strategy: ', nont_labelling.__str__()
        print 'Terminal labelling strategy:    ', term_labelling.func_name
        print 'Recursive partitioning strategy:', recursive_partitioning.func_name
        print 'limit:                          ', str(limit)
        print 'Ignoring punctuation            ', ignore_punctuation
    start_clock = time.clock()

    trees = parse_conll_corpus(path, False, limit)
    trees = add_trees_to_db(path, connection, trees)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar) = d_i.induce_grammar(trees, nont_labelling, term_labelling, recursive_partitioning, start)

    end_clock = time.clock()
    if not quiet:
        print 'Number of trees:                ', str(n_trees)
        print 'Number of nonterminals:         ', len(grammar.nonts())
        print 'Number of rules:                ', len(grammar.rules())
        print 'Total size:                     ', grammar.size()
        print 'Fanout:                         ', max(map(grammar.fanout, grammar.nonts()))
        print 'Induction time:                 ', end_clock - start_clock, 'seconds'

    print experiment
    experiment_database.add_grammar(connection, grammar, experiment)
    assert grammar.ordered()
    return grammar, experiment


def parse_sentences_from_file(grammar
                              , experiment
                              , connection
                              , path
                              , tree_yield
                              , max_length=sys.maxint
                              , limit=sys.maxint
                              , quiet=False
                              , ignore_punctuation=True
                              , root_default_deprel=None
                              , disconnected_default_deprel=None
                              , max_parse_time=sys.maxint
                              , max_parse_memory=resource_limits.unlimited_memory):
    """
    :rtype: None
    :type grammar: LCFRS
    :param path: file path for test corpus (dependency grammar in CoNLL format)
    :type path: str
    :param tree_yield: parse on words or POS
    :type tree_yield: GeneralHybridTree -> List[str]
    :param max_length: don't parse sentences with yield > max_length
    :type max_length: int
    :param limit:      only parse the limit first sentences of the corpus
    :type limit: int
    :param quiet:      output status information
    :type quiet: bool
    :param ignore_punctuation: exclude punctuation from parsing
    :type ignore_punctuation: bool
    Parse sentences from corpus and compare derived dependency structure with gold standard information.
    """

    experiment_database.set_experiment_test_corpus(connection, experiment, path)

    if not quiet:
        if max_length != sys.maxint:
            s = ', ignoring sentences with length > ' + str(max_length)
        else:
            s = ''
        print 'Start parsing sentences' + s

    trees = parse_conll_corpus(path, False, limit)
    trees = add_trees_to_db(path, connection, trees)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)

    (UAS, LAS, UEM, LEM) = (0, 0, 0, 0)
    parse = 0
    no_parse = 0
    n_gaps_gold = 0
    n_gaps_test = 0
    skipped = 0
    start_at = time.clock()
    for tree in trees:
        # if tree.sent_label() != 'tree53':
        # continue
        if len(tree.id_yield()) > max_length:
            skipped += 1
            continue
        time_stamp = time.clock()

        parser = resource_limits.run(NaiveParser, max_parse_time, max_parse_memory, grammar, tree_yield(tree))
        time_stamp = time.clock() - time_stamp

        if isinstance(parser, resource_limits.TimeoutError):
            experiment_database.no_parse_result(connection, tree.sent_label(), path, experiment, time_stamp, "timeout")
            no_parse += 1
        elif isinstance(parser, resource_limits.MemoryoutError):
            experiment_database.no_parse_result(connection, tree.sent_label(), path, experiment, time_stamp,
                                                "memoryout=" + str(max_parse_memory))
            no_parse += 1
            gc.collect()
        else:
            # print tree.sent_label(),
            h_tree = GeneralHybridTree(tree.sent_label())
            h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, tree.full_pos_yield(), tree.full_labelled_yield(),
                                                            ignore_punctuation)

            if h_tree:
                experiment_database.add_result_tree(connection, h_tree, path, experiment, 1, parser.best(), time_stamp,
                                                    'parse', root_default_deprel, disconnected_default_deprel)
                n_gaps_gold += tree.n_gaps()
                n_gaps_test += h_tree.n_gaps()
                parse += 1
                (dUAS, dLAS, dUEM, dLEM) = score_cmp_dep_trees(tree, h_tree)
                UAS += dUAS
                LAS += dLAS
                UEM += dUEM
                LEM += dLEM
            else:
                experiment_database.no_parse_result(connection, tree.sent_label(), path, experiment, time_stamp,
                                                    "no_parse")
                no_parse += 1

    end_at = time.clock()
    total = parse + no_parse
    if not quiet:
        print 'Parsed ' + str(parse) + ' out of ' + str(total) + ' (skipped ' + str(skipped) + ')'
        print 'fail: ', no_parse
        if parse > 0:
            print 'UAS: ', UAS / parse
            print 'LAS: ', LAS / parse
            print 'UEM: ', UEM / parse
            print 'LEM: ', LEM / parse
            print 'n gaps (gold): ', n_gaps_gold * 1.0 / parse
            print 'n gaps (test): ', n_gaps_test * 1.0 / parse
        print 'parse time: ', end_at - start_at, 's'
        print


def test_conll_grammar_induction():
    db_connection = experiment_database.initialize_database(sample_db)

    # if 'ignore_punctuation' in sys.argv:
    # ignore_punctuation = True
    # else:
    # ignore_punctuation = False
    # if 'strict' in sys.argv:
    # nont_labelling = d_i.strict_pos
    # else:
    #     nont_labelling = d_i.child_pos
    # for ignore_punctuation in [True, False]:
    #     for nont_labelling in [d_i.strict_pos, d_i.child_pos]:
    # for rec_par in [d_i.direct_extraction, d_i.fanout_1, d_i.fanout_2, d_i.fanout_3, d_i.fanout_4
    #                , d_i.left_branching, d_i.right_branching]:
    # for nont_labelling, rec_par, ignore_punctuation in [ (d_i.strict_pos_dep, d_i.direct_extraction, True)
    #                                                     , (d_i.strict_pos_dep, d_i.left_branching, True)
    #                                                     , (d_i.child_pos_dep, d_i.direct_extraction, True)
    #                                                     , (d_i.child_pos_dep, d_i.left_branching, True)]:

    root_default_deprel = 'ROOT'
    disconnected_default_deprel = 'PUNC'

    for ignore_punctuation in [True, False]:
        for nont_labelling in [label.StrictPOSLabeling(), label.ChildPOSLabeling(), label.StrictPOSdepAtLeafLabeling(),
                               label.ChildPOSdepAtLeafLabeling()]:
            for rec_par in [d_i.direct_extraction, d_i.left_branching, d_i.right_branching, d_i.fanout_1, d_i.fanout_2]:
                grammar, experiment = induce_grammar_from_file(conll_train, db_connection, nont_labelling, d_i.term_pos,
                                                               rec_par, sys.maxint
                                                               , False, 'START', ignore_punctuation)
                print
                parse_sentences_from_file(grammar, experiment, db_connection, conll_test, d_i.pos_yield, 20, sys.maxint,
                                          False, ignore_punctuation, root_default_deprel, disconnected_default_deprel)

    experiment_database.finalize_database(db_connection)


def run_experiment(db_file, training_corpus, test_corpus, ignore_punctuation, length_limit, labeling, partitioning,
                   root_default_deprel, disconnected_default_deprel, max_training, max_test, max_parse_time,
                   max_parse_memory):
    if labeling == 'strict-pos-leaf:dep':
        nont_labelling = label.StrictPOSdepAtLeafLabeling()
    elif labeling == 'strict-pos':
        nont_labelling = label.StrictPOSLabeling()
    elif labeling == 'strict-pos-dep':
        nont_labelling = label.StrictPOSdepLabeling()
    elif labeling == 'strict-dep':
        nont_labelling = label.StrictDepLabeling()
    elif labeling == 'child-pos-leaf:dep':
        nont_labelling = label.ChildPOSdepAtLeafLabeling()
    elif labeling == 'child-pos':
        nont_labelling = label.ChildPOSLabeling()
    elif labeling == 'child-pos-dep':
        nont_labelling = label.ChildPOSdepLabeling()
    else:
        print("Error: Invalid labeling strategy: " + labeling)
        exit(1)

    if partitioning == 'left-branching':
        rec_par = d_i.left_branching
    elif partitioning == 'right-branching':
        rec_par = d_i.right_branching
    elif partitioning == 'direct-extraction':
        rec_par = d_i.direct_extraction
    elif partitioning == 'fanout-1':
        rec_par = d_i.fanout_1
    elif partitioning == 'fanout-2':
        rec_par = d_i.fanout_2
    elif partitioning == 'fanout-3':
        rec_par = d_i.fanout_3
    elif partitioning == 'fanout-4':
        rec_par = d_i.fanout_4
    elif partitioning == 'fanout-5':
        rec_par = d_i.fanout_5
    elif partitioning == 'fanout-6':
        rec_par = d_i.fanout_6
    elif partitioning == 'fanout-7':
        rec_par = d_i.fanout_7
    elif partitioning == 'fanout-8':
        rec_par = d_i.fanout_8
    else:
        print("Error: Invalid recursive partitioning strategy: " + partitioning)
        exit(1)

    connection = experiment_database.initialize_database(db_file)
    grammar, experiment = induce_grammar_from_file(training_corpus, connection, nont_labelling, d_i.term_pos, rec_par,
                                                   max_training, False, 'START', ignore_punctuation)
    parse_sentences_from_file(grammar, experiment, connection, test_corpus, d_i.pos_yield, length_limit, max_test,
                              False,
                              ignore_punctuation, root_default_deprel, disconnected_default_deprel, max_parse_time,
                              max_parse_memory)
    experiment_database.finalize_database(connection)


def single_experiment_from_config_file(config_path):
    if not os.path.isfile(config_path):
        print "Error: File not found: " + config_path
        exit(1)

    db_file = ''
    training_corpus = ''
    test_corpus = ''
    labeling = ''
    partitioning = ''
    ignore_punctuation = False
    root_default_deprel = None
    disconnected_default_deprel = None
    max_train = sys.maxint
    max_test = sys.maxint
    max_length = sys.maxint
    max_parse_time = sys.maxint
    max_parse_memory = resource_limits.unlimited_memory
    line_nr = 0
    config = open(config_path, "r")
    for line in config.readlines():
        line_nr += 1
        # remove comments
        line = line.split('#')[0]
        if re.search(r'^\s*$', line):
            continue

        match = match_string_argument("Database", line)
        if match:
            db_file = match
            continue

        match = match_string_argument("Training Corpus", line)
        if match:
            training_corpus = match
            continue

        match = match_string_argument("Test Corpus", line)
        if match:
            test_corpus = match
            continue

        match = match_string_argument("Nonterminal Labeling", line)
        if match:
            labeling = match
            continue

        match = match_string_argument("Recursive Partitioning", line)
        if match:
            partitioning = match
            continue

        match = match_string_argument("Nonterminal Labeling", line)
        if match:
            labeling = match
            continue

        match = match_string_argument("Default Root DEPREL", line)
        if match:
            root_default_deprel = match
            continue

        match = match_string_argument("Default Disconnected DEPREL", line)
        if match:
            disconnected_default_deprel = match
            continue

        match = match_string_argument("Ignore Punctuation", line)
        if match:
            if match == "YES":
                ignore_punctuation = True
            elif match == "NO":
                ignore_punctuation = False
            continue

        match = match_integer_argument("Training Limit", line)
        if match is not None:
            max_train = match
            continue

        match = match_integer_argument("Test Limit", line)
        if match is not None:
            max_test = match
            continue

        match = match_integer_argument("Test Length Limit", line)
        if match is not None:
            max_length = match
            continue

        match = match_integer_argument("Parse Memory Limit", line)
        if match is not None:
            max_parse_memory = match
            continue

        match = match_integer_argument("Parse Time Limit", line)
        if match is not None:
            max_parse_time = match
            continue

        print "Error: could not parse line " + str(line_nr) + ": " + line
        exit(1)

    if not db_file:
        print "Error: no database file specified."
        exit(1)
    if not test_corpus:
        print "Error: no test corpus specified."
        exit(1)
    if not training_corpus:
        print "Error: no training corpus specified."
        exit(1)
    if not labeling:
        print "Error: no nonterminal labeling strategy specified."
        exit(1)
    if not partitioning:
        print "Error: no recursive partitioning strategy specified."
        exit(1)

    run_experiment(db_file, training_corpus, test_corpus, ignore_punctuation, max_length, labeling, partitioning,
                   root_default_deprel, disconnected_default_deprel, max_train, max_test, max_parse_time,
                   max_parse_memory)


def match_string_argument(keyword, line):
    match = re.search(r'^' + keyword + ':\s*(\S+)\s*$', line)
    if match:
        return match.group(1)
    else:
        return None


def match_integer_argument(keyword, line):
    match = re.search(r'^' + keyword + ':\s*(\d+)\s*$', line)
    if match:
        return int(match.group(1))
    else:
        return None


if __name__ == '__main__':
    print sys.argv
    if len(sys.argv) > 2 and sys.argv[1] == "run-experiment":
        config = sys.argv[2]
        if os.path.isfile(config):
            single_experiment_from_config_file(config)
        else:
            print "File not found: " + config
