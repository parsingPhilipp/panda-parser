from sys import stderr
from parser.sDCP_parser.sdcp_trace_manager import PySDCPTraceManager
from parser.trace_manager.sm_trainer_util import PyGrammarInfo, PyStorageManager
from parser.trace_manager.sm_trainer import PyEMTrainer, build_PyLatentAnnotation_initial, PySplitMergeTrainerBuilder
from math import exp

def split_merge_training(grammar, corpus, cycles, em_epochs, init="rfe", tie_breaking=False, sigma=0.005, seed=0, merge_threshold=0.5, debug=False, rule_pruning=exp(-100)):
    print >>stderr, "creating trace"
    trace = PySDCPTraceManager(grammar, debug=debug)
    print >>stderr, "computing reducts"
    trace.compute_reducts(corpus)
    print >>stderr, "pre em-training"
    emTrainer = PyEMTrainer(trace)
    emTrainer.em_training(grammar, em_epochs, init, tie_breaking, sigma, seed)
    print >>stderr, "starting actual split/merge training"
    grammarInfo = PyGrammarInfo(grammar, trace.get_nonterminal_map())
    storageManager = PyStorageManager()
    las = [build_PyLatentAnnotation_initial(grammar, grammarInfo, storageManager)]

    trainer = PySplitMergeTrainerBuilder(trace, grammarInfo).set_em_epochs(em_epochs).set_threshold_merger(merge_threshold).build()

    for i in range(cycles):
        las.append(trainer.split_merge_cycle(las[i]))
        smGrammar = las[-1].build_sm_grammar(grammar, grammarInfo, rule_pruning)
        yield smGrammar