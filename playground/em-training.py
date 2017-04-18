from __future__ import print_function

import copy
import os
import pickle
import subprocess
import time

import dependency.induction as d_i
import dependency.labeling as d_l
from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from hybridtree.dependency_tree import disconnect_punctuation
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
from parser.LCFRS.LCFRS_trace_manager import compute_LCFRS_reducts, PyLCFRSTraceManager
from parser.parser_factory import GFParser, GFParser_k_best, CFGParser, LeftBranchingFSTParser, RightBranchingFSTParser
from parser.sDCPevaluation.evaluator import dcp_to_hybridtree, The_DCP_evaluator
from parser.trace_manager.sm_trainer import PyEMTrainer, PyGrammarInfo, PyStorageManager, PySplitMergeTrainerBuilder, build_PyLatentAnnotation_initial, build_PyLatentAnnotation
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
from playground_rparse.process_rparse_grammar import fall_back_left_branching

test = '../res/negra-dep/negra-lower-punct-test.conll'
train ='../res/negra-dep/negra-lower-punct-train.conll'
result = 'experiment_parse_results.conll'
start = 'START'
dir = 'exp12/'
baseline_path = dir + 'baseline_grammar.pkl'
reduct_path = dir + 'reduct.pkl'
reduct_path_discr = dir + 'reduct_discr.pkl'
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
parser_type = CFGParser
# parser_type = GFParser
# parser_type = GFParser_k_best
tree_yield = term_labelling.prepare_parser_input


def do_parsing(grammar_prim, limit, ignore_punctuation, recompile=True, preprocess_path=None):
    trees = parse_conll_corpus(test, False, limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)

    total_time = 0.0

    load_preprocess = preprocess_path
    if recompile or (not os.path.isfile(parser_type.resolve_path(preprocess_path))):
        load_preprocess=None

    parser = parser_type(grammar_prim, save_preprocess=preprocess_path, load_preprocess=load_preprocess)

    with open(result, 'w') as result_file:
        failures = 0
        for tree in trees:
            if len(tree.id_yield()) > limit:
                continue
            time_stamp = time.clock()

            parser.set_input(tree_yield(tree.token_yield()))
            parser.parse()
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
            elif parser_type == CFGParser \
                     or parser_type == GFParser \
                     or parser_type == LeftBranchingFSTParser \
                     or parser_type == RightBranchingFSTParser:
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

            parser.clear()

    print("parse failures", failures)
    print("parse time", total_time)

    print("eval.pl", "no punctuation")
    p = subprocess.Popen(["perl", "../util/eval.pl", "-g", test, "-s", result, "-q"])
    p.communicate()
    print("eval.pl", "punctation")
    p = subprocess.Popen(
        ["perl", "../util/eval.pl", "-g", test, "-s", result, "-q", "-p"])
    p.communicate()

def length_limit(trees, max_length=50):
    for tree in trees:
        if len(tree.full_token_yield()) <= max_length:
            yield tree

def main(limit=300, ignore_punctuation=False, baseline_path=baseline_path, recompileGrammar=True, retrain=True, parsing=True, seed=1337):
    max_length = 20
    trees = length_limit(parse_conll_corpus(train, False, limit), max_length)

    if recompileGrammar or not os.path.isfile(baseline_path):
        (n_trees, baseline_grammar) = d_i.induce_grammar(trees, empty_labelling, term_labelling.token_label, recursive_partitioning, start)
        pickle.dump(baseline_grammar, open(baseline_path, 'wb'))
    else:
        baseline_grammar = pickle.load(open(baseline_path))

    test_limit = 10000
    print("Rules: ", len(baseline_grammar.rules()))

    if parsing:
        do_parsing(baseline_grammar, test_limit, ignore_punctuation, recompileGrammar, [dir, "baseline_gf_grammar"])

    em_trained = pickle.load(open(baseline_path))
    if recompileGrammar or not os.path.isfile(reduct_path):
        trees = length_limit(parse_conll_corpus(train, False, limit), max_length)
        trace = compute_reducts(em_trained, trees, term_labelling)
        trace.serialize(reduct_path)
    else:
        print("loading trace")
        trace = PySDCPTraceManager(em_trained, term_labelling)
        trace.load_traces_from_file(reduct_path)

    discr = False
    if discr:
        if recompileGrammar or not os.path.isfile(reduct_path_discr):
            trees = length_limit(parse_conll_corpus(train, False, limit), max_length)
            trace_discr = compute_LCFRS_reducts(em_trained, trees, term_labelling, nonterminal_map=trace.get_nonterminal_map())
            trace_discr.serialize(reduct_path_discr)
        else:
            print("loading trace discriminative")
            trace_discr = PyLCFRSTraceManager(em_trained, trace.get_nonterminal_map())
            trace_discr.load_traces_from_file(reduct_path_discr)

    n_epochs = 20
    init = "rfe"
    tie_breaking = True
    em_trained_path_ = em_trained_path(n_epochs, init, tie_breaking)

    if recompileGrammar or retrain or not os.path.isfile(em_trained_path_):
        emTrainer = PyEMTrainer(trace)
        emTrainer.em_training(em_trained, n_epochs=n_epochs, init=init, tie_breaking=tie_breaking, seed=seed)
        pickle.dump(em_trained, open(em_trained_path_, 'wb'))
    else:
        em_trained = pickle.load(open(em_trained_path_, 'rb'))

    if parsing:
        do_parsing(em_trained, test_limit, ignore_punctuation, recompileGrammar or retrain, [dir, "em_trained_gf_grammar"])

    grammarInfo = PyGrammarInfo(baseline_grammar, trace.get_nonterminal_map())
    storageManager = PyStorageManager()

    builder = PySplitMergeTrainerBuilder(trace, grammarInfo)
    builder.set_em_epochs(n_epochs)
    builder.set_split_randomization(1.0, seed + 1)
    if discr:
        builder.set_discriminative_expector(trace_discr, maxScale=10, threads=1)
    else:
        builder.set_simple_expector(threads=1)
    splitMergeTrainer = builder.set_percent_merger(65.0).build()


    if (not recompileGrammar) and (not retrain) and os.path.isfile(sm_info_path):
        print("Loading splits and weights of LA rules")
        latentAnnotation = map(lambda t: build_PyLatentAnnotation(t[0], t[1], t[2], grammarInfo, storageManager)
                               , pickle.load(open(sm_info_path, 'rb')))
    else:
        latentAnnotation = [build_PyLatentAnnotation_initial(em_trained, grammarInfo, storageManager)]

    max_cycles = 4
    reparse = False
    # parsing = False
    for i in range(max_cycles + 1):
        if i < len(latentAnnotation):
            if reparse:
                smGrammar = latentAnnotation[i].build_sm_grammar(baseline_grammar
                                                                 , grammarInfo
                                                                 , rule_pruning=0.0001
                                                                 , rule_smoothing=0.01)
                print("Cycle: ", i, "Rules: ", len(smGrammar.rules()))
                do_parsing(smGrammar, test_limit, ignore_punctuation, recompileGrammar or retrain, [dir, "sm_cycles" + str(i) + "_gf_grammar"])
        else:
            # setting the seed to achieve reproducibility in case of continued training
            splitMergeTrainer.reset_random_seed(seed + i + 1)
            latentAnnotation.append(splitMergeTrainer.split_merge_cycle(latentAnnotation[-1]))
            pickle.dump(map(lambda la: la.serialize(), latentAnnotation), open(sm_info_path, 'wb'))
            smGrammar = latentAnnotation[i].build_sm_grammar(baseline_grammar
                                                             , grammarInfo
                                                             , rule_pruning=0.0001
                                                             , rule_smoothing=0.1)
            print("Cycle: ", i, "Rules: ", len(smGrammar.rules()))
            if parsing:
                do_parsing(smGrammar, test_limit, ignore_punctuation, recompileGrammar or retrain, [dir, "sm_cycles" + str(i) + "_gf_grammar"])

if __name__ == '__main__':
    main()
