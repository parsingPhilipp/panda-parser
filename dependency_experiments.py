__author__ = 'kilian'

conll_test = '../dependency_conll/german/tiger/test/german_tiger_test.conll'
conll_train = '../dependency_conll/german/tiger/train/german_tiger_train.conll'

import time
import sys
import copy

from hybridtree.general_hybrid_tree import GeneralHybridTree
import dependency.induction as d_i
import dependency.labeling as label
from hybridtree.biranked_tokens import construct_conll_token
from parser.naive.parsing import LCFRS_parser
from corpora.conll_parse import parse_conll_corpus, score_cmp_dep_trees


def induce_grammar_from_file(path
                             , nont_labelling
                             , term_labelling
                             , recursive_partitioning
                             , limit=sys.maxint
                             , quiet=False
                             , start='START'
                             , ignore_punctuation=True
                             ):
    """
    Extract an LCFRS/sDCP-Hybrid Grammar from a dependency corpus in CoNLL format.
    :param path: str, (path to dependency corpus in CoNLL format)
    :param nont_labelling: GeneralHybridTree, Top_max, Bottom_max, Fanout -> str
    :param term_labelling: GeneralHybridTree, NodeId -> str
    :param recursive_partitioning: GeneralHybridTree -> RecursivePartitioning
    :param limit: int (use only the first _limit_ trees for grammar induction)
    :param quiet: bool (status output)
    :param start: str (set start nonterminal for grammar)
    :param ignore_punctuation: (include punctuation into grammar)
    :return: LCFRS
    """
    if not quiet:
        print 'Inducing grammar'
        print 'file: ' + path
        print 'Nonterminal labelling strategy: ', nont_labelling.func_name
        print 'Terminal labelling strategy:    ', term_labelling.func_name
        print 'Recursive partitioning strategy:', recursive_partitioning.func_name
        print 'limit:                          ', str(limit)
        print 'Ignoring punctuation            ', ignore_punctuation
    start_at = time.time()

    trees = parse_conll_corpus(path, ignore_punctuation, limit)
    (n_trees, grammar) = d_i.induce_grammar(trees, nont_labelling, term_labelling, recursive_partitioning, start)

    end_at = time.time()
    if not quiet:
        print 'Number of trees:                ', str(n_trees)
        print 'Number of nonterminals:          ', len(grammar.nonts())
        print 'Number of rules:                ', len(grammar.rules())
        print 'Total size:                     ', grammar.size()
        print 'Fanout:                         ', max(map(grammar.fanout, grammar.nonts()))
        print 'Induction time:                 ', end_at - start_at, 'seconds'

    return grammar


def parse_sentences_from_file(grammar
                              , path
                              , tree_yield
                              , max_length=sys.maxint
                              , limit=sys.maxint
                              , quiet=False
                              , ignore_punctuation=True
                              ):
    """
    Parse sentences from corpus and compare derived dependency structure with gold standard information.
    :rtype: None
    :param grammar: LCFRS
    :param path:    file path for test corpus (dependency grammar in CoNLL format)
    :param tree_yield: GeneralHybridTree -> [string] (parse on words or POS)
    :param max_length: don't parse sentences with yield > max_length
    :param limit:      only parse the limit first sentences of the corpus
    :param quiet:      output status information
    :param ignore_punctuation: exclude punctuation from parsing
    """
    if not quiet:
        if max_length != sys.maxint:
            s = ', ignoring sentences with length > ' + str(max_length)
        else:
            s = ''
        print 'Start parsing sentences' + s

    trees = parse_conll_corpus(path, ignore_punctuation, limit)

    (UAS, LAS, UEM, LEM) = (0, 0, 0, 0)
    parse = 0
    no_parse = 0
    n_gaps_gold = 0
    n_gaps_test = 0
    skipped = 0
    start_at = time.time()
    for tree in trees:
        if len(tree.id_yield()) > max_length:
            skipped += 1
            continue
        parser = LCFRS_parser(grammar, tree_yield(tree))
        h_tree = GeneralHybridTree()
        cleaned_tokens = copy.deepcopy(tree.full_token_yield())
        for token in cleaned_tokens:
            token.set_deprel('_')
        h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, cleaned_tokens,
                                                        ignore_punctuation, construct_conll_token)
        if h_tree:
            n_gaps_gold += tree.n_gaps()
            n_gaps_test += h_tree.n_gaps()
            parse += 1
            dUAS, dLAS, dUEM, dLEM = score_cmp_dep_trees(tree, h_tree)
            UAS += dUAS
            LAS += dLAS
            UEM += dUEM
            LEM += dLEM
        else:
            no_parse += 1

    end_at = time.time()
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
    nt_labeling = label.the_labeling_factory().create_simple_labeling_strategy('strict', 'pos+deprel')
    nt_labeling2 = label.the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel')
    t_labeling = d_i.the_terminal_labeling_factory().get_strategy('pos')
    for nont_labelling, rec_par, ignore_punctuation in [
        (nt_labeling, d_i.direct_extraction, True)
        , (nt_labeling, d_i.left_branching, True)
        , (nt_labeling2, d_i.direct_extraction, True)
        , (nt_labeling2, d_i.left_branching, True)]:
        grammar = induce_grammar_from_file(conll_train, nont_labelling, t_labeling.token_label, rec_par, 200
                                           , False, 'START', ignore_punctuation)
        print
        parse_sentences_from_file(grammar, conll_test, t_labeling.prepare_parser_input, 20, sys.maxint, False,
                                  ignore_punctuation)


if __name__ == '__main__':
    test_conll_grammar_induction()
