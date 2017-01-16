from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from hybridtree.dependency_tree import disconnect_punctuation
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
import dependency.induction as d_i
import dependency.labeling as d_l
import time
from parser.parser_factory import GFParser, GFParser_k_best, CFGParser
import parser.gf_parser.gf_interface
from parser.sDCPevaluation.evaluator import dcp_to_hybridtree, The_DCP_evaluator
import copy
from playground_rparse.process_rparse_grammar import fall_back_left_branching
import subprocess
from parser.sDCP_parser.sdcp_parser_wrapper import em_training, split_merge_training, compute_reducts, load_reducts
from math import exp
import pickle, os

test = '../res/negra-dep/negra-lower-punct-test.conll'
train ='../res/negra-dep/negra-lower-punct-train.conll'
result = 'experiment_parse_results.conll'
start = 'START'
dir = 'exp4/'
baseline_path = dir + 'baseline_grammar.pkl'
reduct_path = dir + 'reduct.pkl'
sm_info_path = dir + 'sm_info.pkl'
def em_trained_path(n_epochs, init, tie_breaking):
    return dir + 'em_trained_grammar_' + str(n_epochs) + '_' + init + ('_tie_breaking_' if tie_breaking else '')  + '.pkl'
def sm_path(cycles):
    return dir + 'sm_' + str(cycles) + '_grammar.pkl'


term_labelling = d_i.the_terminal_labeling_factory().get_strategy('pos')
recursive_partitioning = d_i.the_recursive_partitioning_factory().getPartitioning('fanout-1')
primary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel')
secondary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('strict', 'deprel')
ternary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'deprel')
child_top_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('childtop', 'deprel')
empty_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('empty', 'pos');
#parser_type = parser.parser_factory.CFGParser
parser_type = GFParser
# parser_type = GFParser_k_best
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

            if parser_type == GFParser_k_best and parser.recognized():
                der_to_tree = lambda der: dcp_to_hybridtree(HybridTree(), The_DCP_evaluator(der).getEvaluation(),
                                                        copy.deepcopy(tree.full_token_yield()), False,
                                                        construct_conll_token)
                h_tree = parser.best_trees(der_to_tree)[0][0]
            elif parser_type == CFGParser or parser_type == GFParser:
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


def main(limit=500, ignore_punctuation=False, baseline_path=baseline_path, recompileGrammar=False, retrain=False, parsing=True):
    trees = parse_conll_corpus(train, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)

    if recompileGrammar or not os.path.isfile(baseline_path):
        (n_trees, baseline_grammar) = d_i.induce_grammar(trees, empty_labelling, term_labelling.token_label, recursive_partitioning, start)
        pickle.dump(baseline_grammar, open(baseline_path, 'wb'))
    else:
        baseline_grammar = pickle.load(open(baseline_path))

    test_limit = 10000
    print "Rules: ", len(baseline_grammar.rules())

    if parsing:
        parser_type.preprocess_grammar(baseline_grammar)
        do_parsing(baseline_grammar, test_limit, ignore_punctuation)

    em_trained = pickle.load(open(baseline_path))
    if recompileGrammar or not os.path.isfile(reduct_path):
        trees = parse_conll_corpus(train, False, limit)
        trace = compute_reducts(em_trained, trees)

        reducts = trace.serialize_trace()
        pickle.dump(reducts, open(reduct_path, 'wb'))
    else:
        reducts = pickle.load(open(reduct_path, "rb"))
        trace = load_reducts(em_trained, reducts)

    n_epochs = 50
    init = "rfe"
    tie_breaking = True
    em_trained_path_ = em_trained_path(n_epochs, init, tie_breaking)

    if recompileGrammar or retrain or not os.path.isfile(em_trained_path_):
        trace.em_training(em_trained, n_epochs=n_epochs, init=init, tie_breaking=tie_breaking)
        pickle.dump(em_trained, open(em_trained_path_, 'wb'))
    else:
        em_trained = pickle.load(open(em_trained_path_, 'rb'))

    if parsing:
        parser_type.preprocess_grammar(em_trained)
        do_parsing(em_trained, test_limit, ignore_punctuation)

    trace = load_reducts(baseline_grammar, reducts)
    if not retrain and os.path.isfile(sm_info_path):
        nont_split_list, rule_weight_list = pickle.load(open(sm_info_path, 'rb'))
        trace.deserialize_la_state(nont_split_list, rule_weight_list)
        print "Loading splits and weights of LA rules"

    grammar_sm = {}
    max_cycles = 4
    for cycles_, grammar in enumerate(trace.split_merge_training(baseline_grammar, max_cycles, em_epochs=20, init="rfe", tie_breaking=True, merge_threshold=0.1, rule_pruning=exp(-50))):
        cycles = cycles_ + 1

        sm_path_ = sm_path(cycles)
        if recompileGrammar or retrain or not os.path.isfile(sm_path_):
            grammar_sm[cycles] = grammar
            # saving grammar
            pickle.dump(grammar_sm[cycles], open(sm_path_, 'wb'))
            # saving S/M state
            pickle.dump(trace.serialize_la_state(), open(sm_info_path, 'wb'))
        else:
            grammar_sm[cycles] = pickle.load(open(sm_path_, 'rb'))
        print "Rules: ", len(grammar_sm[cycles].rules())

        if parsing:
            parser_type.preprocess_grammar(grammar_sm[cycles])
            do_parsing(grammar_sm[cycles], test_limit, ignore_punctuation)




if __name__ == '__main__':
    main()
