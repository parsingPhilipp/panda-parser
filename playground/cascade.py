import grammar.induction.recursive_partitioning
import grammar.induction.terminal_labeling
from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from hybridtree.dependency_tree import disconnect_punctuation
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
import dependency.induction as d_i
import dependency.labeling as d_l
import time
import parser.parser_factory
import copy
from playground_rparse.process_rparse_grammar import fall_back_left_branching
import subprocess

test = '../res/negra-dep/negra-lower-punct-test.conll'
train = '../res/negra-dep/negra-lower-punct-train.conll'
result = 'cascade-parse-results.conll'
start = 'START'
term_labelling = grammar.induction.terminal_labeling.the_terminal_labeling_factory().get_strategy('pos')
recursive_partitioning = grammar.induction.recursive_partitioning.the_recursive_partitioning_factory().\
    getPartitioning('fanout-1')
primary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel')
secondary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'pos')
ternary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'deprel')
# parser_type = parser.parser_factory.CFGParser
parser_type = parser.parser_factory.GFParser
tree_yield = term_labelling.prepare_parser_input


def main(limit=100000, ignore_punctuation=False):
    test_limit = 10000
    trees = parse_conll_corpus(train, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar_prim) = d_i.induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)
    parser_type.preprocess_grammar(grammar_prim)

    trees = parse_conll_corpus(train, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar_second) = d_i.induce_grammar(trees, secondary_labelling, term_labelling.token_label,
                                                   recursive_partitioning, start)
    parser_type.preprocess_grammar(grammar_second)

    trees = parse_conll_corpus(train, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar_tern) = d_i.induce_grammar(trees, ternary_labelling, term_labelling.token_label,
                                                 recursive_partitioning, start)
    parser_type.preprocess_grammar(grammar_tern)

    trees = parse_conll_corpus(test, False, test_limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)

    total_time = 0.0

    with open(result, 'w') as result_file:
        failures = 0
        for tree in trees:
            time_stamp = time.clock()

            parser = parser_type(grammar_prim, tree_yield(tree.token_yield()))
            if not parser.recognized():
                parser = parser_type(grammar_second, tree_yield(tree.token_yield()))
            if not parser.recognized():
                parser = parser_type(grammar_tern, tree_yield(tree.token_yield()))
            time_stamp = time.clock() - time_stamp
            total_time += time_stamp

            cleaned_tokens = copy.deepcopy(tree.full_token_yield())
            for token in cleaned_tokens:
                token.set_edge_label('_')
            h_tree = HybridTree(tree.sent_label())
            h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, cleaned_tokens, ignore_punctuation,
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
    p = subprocess.Popen(["perl", "../util/eval.pl", "-g", test, "-s", result, "-q"])
    p.communicate()
    print("eval.pl", "punctation")
    p = subprocess.Popen(
        ["perl", "../util/eval.pl", "-g", test, "-s", result, "-q", "-p"])
    p.communicate()


if __name__ == '__main__':
    main()
