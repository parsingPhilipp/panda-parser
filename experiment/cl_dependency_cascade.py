import grammar.induction.recursive_partitioning
import grammar.induction.terminal_labeling
from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from hybridtree.dependency_tree import disconnect_punctuation, fall_back_left_branching
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
import dependency.induction as d_i
import dependency.labeling as d_l
import time
import parser.parser_factory
import copy
import subprocess

TEST = 'res/negra-dep/negra-lower-punct-test.conll'
TRAIN = 'res/negra-dep/negra-lower-punct-train.conll'
RESULT = '.tmp/cascade-parse-results.conll'
START = 'START'
TERMINAL_LABELLING = grammar.induction.terminal_labeling.the_terminal_labeling_factory().get_strategy('pos')
RECURSIVE_PARTITIONING = grammar.induction.recursive_partitioning.the_recursive_partitioning_factory(). \
    get_partitioning('fanout-1')
PRIMARY_LABELLING = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel')
SECONDARY_LABELLING = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'pos')
TERNARY_LABELLING = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'deprel')

PARSER_TYPE = parser.parser_factory.the_parser_factory().getParser("fanout-1")
TREE_YIELD = TERMINAL_LABELLING.prepare_parser_input


def main(limit=100000, ignore_punctuation=False):
    if PARSER_TYPE.__name__ != 'GFParser':
        print('GFParser not found, using', PARSER_TYPE.__name__, 'instead!')
        print('Please install grammatical framework to reproduce experiments.')

    test_limit = 10000
    trees = parse_conll_corpus(TRAIN, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar_prim) = d_i.induce_grammar(trees, PRIMARY_LABELLING, TERMINAL_LABELLING.token_label,
                                                 RECURSIVE_PARTITIONING, START)
    PARSER_TYPE.preprocess_grammar(grammar_prim)

    trees = parse_conll_corpus(TRAIN, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar_second) = d_i.induce_grammar(trees, SECONDARY_LABELLING, TERMINAL_LABELLING.token_label,
                                                   RECURSIVE_PARTITIONING, START)
    PARSER_TYPE.preprocess_grammar(grammar_second)

    trees = parse_conll_corpus(TRAIN, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar_tern) = d_i.induce_grammar(trees, TERNARY_LABELLING, TERMINAL_LABELLING.token_label,
                                                 RECURSIVE_PARTITIONING, START)
    PARSER_TYPE.preprocess_grammar(grammar_tern)

    trees = parse_conll_corpus(TEST, False, test_limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)

    total_time = 0.0

    with open(RESULT, 'w') as result_file:
        failures = 0
        for tree in trees:
            time_stamp = time.clock()

            the_parser = PARSER_TYPE(grammar_prim, TREE_YIELD(tree.token_yield()))
            if not the_parser.recognized():
                the_parser = PARSER_TYPE(grammar_second, TREE_YIELD(tree.token_yield()))
            if not the_parser.recognized():
                the_parser = PARSER_TYPE(grammar_tern, TREE_YIELD(tree.token_yield()))
            time_stamp = time.clock() - time_stamp
            total_time += time_stamp

            cleaned_tokens = copy.deepcopy(tree.full_token_yield())
            for token in cleaned_tokens:
                token.set_edge_label('_')
            h_tree = HybridTree(tree.sent_label())
            h_tree = the_parser.dcp_hybrid_tree_best_derivation(h_tree, cleaned_tokens, ignore_punctuation,
                                                                construct_conll_token)

            if h_tree:
                result_file.write(tree_to_conll_str(h_tree))
                result_file.write('\n\n')
            else:
                failures += 1
                forms = [token.form() for token in tree.full_token_yield()]
                poss = [token.pos() for token in tree.full_token_yield()]
                result_file.write(tree_to_conll_str(fall_back_left_branching(forms, poss)))
                result_file.write('\n\n')

    print("parse failures", failures)
    print("parse time", total_time)

    print("eval.pl", "no punctuation")
    p = subprocess.Popen(["perl", "util/eval.pl", "-g", TEST, "-s", RESULT, "-q"])
    p.communicate()
    print("eval.pl", "punctuation")
    p = subprocess.Popen(
        ["perl", "util/eval.pl", "-g", TEST, "-s", RESULT, "-q", "-p"])
    p.communicate()


if __name__ == '__main__':
    main()
