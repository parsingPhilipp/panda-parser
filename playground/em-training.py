from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from hybridtree.dependency_tree import disconnect_punctuation
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
import dependency.induction as d_i
import dependency.labeling as d_l
import time
import parser.parser_factory
import parser.gf_parser.gf_interface
from parser.sDCPevaluation.evaluator import dcp_to_hybridtree, The_DCP_evaluator
import copy
from playground_rparse.process_rparse_grammar import fall_back_left_branching
import subprocess
from parser.sDCP_parser.sdcp_parser_wrapper import em_training, split_merge_training, compute_reducts, load_reducts
from math import exp

test = '../res/negra-dep/negra-lower-punct-test.conll'
train ='../res/negra-dep/negra-lower-punct-train.conll'
result = 'cascade-parse-results.conll'
start = 'START'
term_labelling = d_i.the_terminal_labeling_factory().get_strategy('pos')
recursive_partitioning = d_i.the_recursive_partitioning_factory().getPartitioning('fanout-1')
primary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel')
secondary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('strict', 'deprel')
ternary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'deprel')
#parser_type = parser.parser_factory.CFGParser
# parser_type = parser.parser_factory.GFParser
parser_type = parser.gf_parser.gf_interface.GFParser_k_best
tree_yield = term_labelling.prepare_parser_input


def do_parsing(grammar_prim, limit, ignore_punctuation):
    trees = parse_conll_corpus(test, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)

    total_time = 0.0

    with open(result, 'w') as result_file:
        failures = 0
        for tree in trees:
            if len(tree.id_yield()) > limit:
                continue
            time_stamp = time.clock()

            parser = parser_type(grammar_prim, tree_yield(tree.token_yield()))
            # if not parser.recognized():
            #     parser = parser_type(grammar_second, tree_yield(tree.token_yield()))
            # if not parser.recognized():
            #     parser = parser_type(grammar_tern, tree_yield(tree.token_yield()))
            time_stamp = time.clock() - time_stamp
            total_time += time_stamp

            cleaned_tokens = copy.deepcopy(tree.full_token_yield())
            for token in cleaned_tokens:
                token.set_deprel('_')

            h_tree = HybridTree(tree.sent_label())

            if parser_type == parser.gf_parser.gf_interface.GFParser_k_best and parser.recognized():
                der_to_tree = lambda der: dcp_to_hybridtree(HybridTree(), The_DCP_evaluator(der).getEvaluation(),
                                                        copy.deepcopy(tree.full_token_yield()), False,
                                                        construct_conll_token)
                h_tree = parser.best_trees(der_to_tree)[0][0]
            elif parser_type == parser.parser_factory.CFGParser:
                h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, cleaned_tokens, ignore_punctuation,
                                                              construct_conll_token)
            else:
                h_tree = None

            if h_tree:
                result_file.write(tree_to_conll_str(h_tree))
                result_file.write('\n\n')
            else:
                failures += 1
                forms = [token.form() for token in tree.full_token_yield()]
                poss = [token.pos() for token in tree.full_token_yield()]
                result_file.write(tree_to_conll_str(fall_back_left_branching(forms, poss)))
                result_file.write('\n\n')

    print "parse failures", failures
    print "parse time", total_time

    print "eval.pl", "no punctuation"
    p = subprocess.Popen(["perl", "../util/eval.pl", "-g", test, "-s", result, "-q"])
    p.communicate()
    print "eval.pl", "punctation"
    p = subprocess.Popen(
        ["perl", "../util/eval.pl", "-g", test, "-s", result, "-q", "-p"])
    p.communicate()

def main(limit=500, ignore_punctuation=False):
    trees = parse_conll_corpus(train, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar_prim) = d_i.induce_grammar(trees, ternary_labelling, term_labelling.token_label, recursive_partitioning, start)
    test_limit = 10000
    print "Rules: ", len(grammar_prim.rules())

    parser_type.preprocess_grammar(grammar_prim)
    do_parsing(grammar_prim, test_limit, ignore_punctuation)

    trees = parse_conll_corpus(train, False, limit)
    trace = compute_reducts(grammar_prim, trees)
    trace.em_training(grammar_prim, n_epochs=50, init="rfe", tie_breaking=True)

    # em_training(grammar_prim, trees, 50, tie_breaking=True)
    parser_type.preprocess_grammar(grammar_prim)

    do_parsing(grammar_prim, test_limit, ignore_punctuation)

    grammar_sm = {}

    for cycles in range(1, 4):
        grammar_sm[cycles] = trace.split_merge_training(grammar_prim, cycles, em_epochs=10, init="rfe", tie_breaking=True, merge_threshold=0.1, rule_pruning=exp(-200))
        print "Rules: ", len(grammar_sm[cycles].rules())

        parser_type.preprocess_grammar(grammar_sm[cycles])
        do_parsing(grammar_sm[cycles], test_limit, ignore_punctuation)







if __name__ == '__main__':
    main()
