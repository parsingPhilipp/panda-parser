from __future__ import print_function
from sys import stderr
from parser.sDCP_parser.sdcp_trace_manager import PySDCPTraceManager
from parser.trace_manager.sm_trainer_util import PyGrammarInfo, PyStorageManager
from parser.trace_manager.sm_trainer import PyEMTrainer, build_PyLatentAnnotation_initial, PySplitMergeTrainerBuilder
from math import exp
from parser.lcfrs_la import build_sm_grammar


def split_merge_training(grammar, term_labelling, corpus, cycles, em_epochs, init="rfe", tie_breaking=False, sigma=0.005, seed=0, merge_threshold=0.5, debug=False, rule_pruning=exp(-100)):
    print("creating trace", file=stderr)
    trace = PySDCPTraceManager(grammar, term_labelling, debug=debug)
    print("computing reducts", file=stderr)
    trace.compute_reducts(corpus)
    print("pre em-training", file=stderr)
    emTrainer = PyEMTrainer(trace)
    emTrainer.em_training(grammar, em_epochs, init, tie_breaking, sigma, seed)
    print("starting actual split/merge training", file=stderr)
    grammarInfo = PyGrammarInfo(grammar, trace.get_nonterminal_map())
    storageManager = PyStorageManager()
    las = [build_PyLatentAnnotation_initial(grammar, grammarInfo, storageManager)]

    trainer = PySplitMergeTrainerBuilder(trace, grammarInfo).set_em_epochs(em_epochs).set_threshold_merger(merge_threshold).build()

    for i in range(cycles):
        las.append(trainer.split_merge_cycle(las[i]))
        smGrammar = build_sm_grammar(las[-1], grammar, grammarInfo, rule_pruning)
        yield smGrammar