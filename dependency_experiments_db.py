__author__ = 'kilian'

conll_test = '../dependency_conll/german/tiger/test/german_tiger_test.conll'
conll_train = '../dependency_conll/german/tiger/train/german_tiger_train.conll'
sample_db = 'examples/sampledb.db'

from general_hybrid_tree import GeneralHybridTree
import dependency_induction as d_i
from parsing import LCFRS_parser
from conll_parse import parse_conll_corpus, score_cmp_dep_trees
import time
import sys
import experiment_database
import re

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


def induce_grammar_from_file(  path
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
    :type nont_labelling: GeneralHybridTree, List[List[str]], List[List[str]], int -> str
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
                                       , nont_labelling.func_name
                                       , recursive_partitioning.func_name
                                       , ignore_punctuation
                                       , path
                                       , time.time()
                                       , None)

    if not quiet:
        print 'Inducing grammar'
        print 'file: ' + path
        print 'Nonterminal labelling strategy: ', nont_labelling.func_name
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
        print 'Number of nonterimals:          ', len(grammar.nonts())
        print 'Number of rules:                ', len(grammar.rules())
        print 'Total size:                     ', grammar.size()
        print 'Fanout:                         ', max(map(grammar.fanout, grammar.nonts()))
        print 'Induction time:                 ', end_clock - start_clock, 'seconds'

    print experiment
    experiment_database.add_grammar(connection, grammar, experiment)
    return grammar, experiment


def parse_sentences_from_file( grammar
                             , experiment
                             , connection
                             , path
                             , tree_yield
                             , max_length = sys.maxint
                             , limit=sys.maxint
                             , quiet=False
                             , ignore_punctuation=True
                             ):
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
        if len(tree.id_yield()) > max_length:
            skipped += 1
            continue
        time_stamp = time.clock()
        parser = LCFRS_parser(grammar, tree_yield(tree))
        h_tree = GeneralHybridTree(tree.sent_label())
        h_tree = parser.new_DCP_Hybrid_Tree(h_tree, tree.full_pos_yield(), tree.full_labelled_yield(), ignore_punctuation)
        time_stamp = time.clock() - time_stamp
        if h_tree:
            experiment_database.add_result_tree(connection, h_tree, path, experiment, 1, parser.best(), time_stamp)
            n_gaps_gold += tree.n_gaps()
            n_gaps_test += h_tree.n_gaps()
            parse += 1
            (dUAS, dLAS, dUEM, dLEM) = score_cmp_dep_trees(tree, h_tree)
            UAS += dUAS
            LAS += dLAS
            UEM += dUEM
            LEM += dLEM
        else:
            no_parse += 1

    end_at = time.clock()
    total = parse + no_parse
    if not quiet:
        print 'Parsed ' + str(parse) + ' out of ' + str(total) + ' (skipped ' + str(skipped) +')'
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

    db_connection = experiment_database.initalize_database(sample_db)

    # if 'ignore_punctuation' in sys.argv:
    #     ignore_punctuation = True
    # else:
    #     ignore_punctuation = False
    # if 'strict' in sys.argv:
    #     nont_labelling = d_i.strict_pos
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
    for ignore_punctuation in [True, False]:
        for nont_labelling in [d_i.strict_pos, d_i.child_pos, d_i.strict_pos_dep, d_i.child_pos_dep]:
            for rec_par in [d_i.direct_extraction, d_i.left_branching, d_i.right_branching, d_i.fanout_1, d_i.fanout_2]:
                grammar, experiment = induce_grammar_from_file(conll_train, db_connection, nont_labelling, d_i.term_pos, rec_par, sys.maxint
                                                   , False, 'START', ignore_punctuation)
                print
                parse_sentences_from_file(grammar, experiment, db_connection, conll_test, d_i.pos_yield, 20, sys.maxint, False, ignore_punctuation)

    experiment_database.finalize_database(db_connection)

if __name__ == '__main__':
    test_conll_grammar_induction()

