from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from hybridtree.dependency_tree import disconnect_punctuation
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
import dependency.induction as d_i
import dependency.labeling as d_l
import time
import parser.parser_factory
import copy
from parser.sDCP_parser.sdcp_parser_wrapper import print_grammar, PysDCPParser, LCFRS_sDCP_Parser, SDCPDerivation
from playground_rparse.process_rparse_grammar import fall_back_left_branching
import subprocess

test = '../res/negra-dep/negra-lower-punct-test.conll'
train ='../res/negra-dep/negra-lower-punct-train.conll'
result = 'recursive-partitoning-results.conll'
start = 'START'
term_labelling = d_i.the_terminal_labeling_factory().get_strategy('pos')
recursive_partitioning = d_i.the_recursive_partitioning_factory().getPartitioning('fanout-1')
primary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel')

# parser_type = parser.parser_factory.GFParser  # slower, can be used for arbitrary fanout
# parser_type = parser.parser_factory.CFGParser  # potentially faster, only for fanout 1
parser_type = LCFRS_sDCP_Parser  # tree parser to count derivations per hybrid tree


tree_yield = term_labelling.prepare_parser_input

train_limit = 2000
test_limit = 2000


def main(ignore_punctuation=False):
    trees = parse_conll_corpus(train, False, train_limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar) = d_i.induce_grammar(trees, primary_labelling, term_labelling.token_label, recursive_partitioning, start)

    #
    #
    # grammar is the induced hybrid grammar
    #
    #


    total_time = 0.0

    # # The following code works for string parsers for evaluating
    #
    # parser_type.preprocess_grammar(grammar)
    #
    # trees = parse_conll_corpus(test, False, test_limit)
    # if ignore_punctuation:
    #     trees = disconnect_punctuation(trees)
    #
    # with open(result, 'w') as result_file:
    #     failures = 0
    #     for tree in trees:
    #         time_stamp = time.clock()
    #
    #         parser = parser_type(grammar, tree_yield(tree.token_yield()))
    #
    #         time_stamp = time.clock() - time_stamp
    #         total_time += time_stamp
    #
    #
    #         cleaned_tokens = copy.deepcopy(tree.full_token_yield())
    #         for token in cleaned_tokens:
    #             token.set_deprel('_')
    #         h_tree = HybridTree(tree.sent_label())
    #         h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, cleaned_tokens, ignore_punctuation,
    #                                                         construct_conll_token)
    #
    #         if h_tree:
    #             result_file.write(tree_to_conll_str(h_tree))
    #             result_file.write('\n\n')
    #         else:
    #             failures += 1
    #             forms = [token.form() for token in tree.full_token_yield()]
    #             poss = [token.pos() for token in tree.full_token_yield()]
    #             result_file.write(tree_to_conll_str(fall_back_left_branching(forms, poss)))
    #             result_file.write('\n\n')
    #
    # print "parse failures", failures
    # print "parse time", total_time
    #
    # print "eval.pl", "no punctuation"
    # p = subprocess.Popen(["perl", "../util/eval.pl", "-g", test, "-s", result, "-q"])
    # p.communicate()
    #
    # print "eval.pl", "punctation"
    # p = subprocess.Popen(
    #     ["perl", "../util/eval.pl", "-g", test, "-s", result, "-q", "-p"])
    # p.communicate()


    # The following code is to count the number of derivations for a hypergraph (tree parser required)
    parser_type.preprocess_grammar(grammar)

    trees = parse_conll_corpus(train , False, train_limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)

    derCount = 0
    for tree in trees:
        parser = parser_type(grammar, tree)  # if tree parser is used
        derCount += parser.count_derivation_trees()

    print 1.0*derCount/train_limit


if __name__ == '__main__':
    main()
