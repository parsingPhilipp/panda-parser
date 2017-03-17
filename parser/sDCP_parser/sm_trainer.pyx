from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
from cython.operator cimport dereference as deref
from parser.sDCP_parser.trace_manager cimport PyTraceManager, TraceManagerPtr, build_trace_manager_ptr
import time
import random
import grammar.lcfrs as gl
import itertools
from math import exp

DEF ENCODE_NONTERMINALS = True
DEF ENCODE_TERMINALS = True

from parser.sDCP_parser.sdcp_parser_wrapper cimport SDCPParser, PySDCPParser, unsigned_int, NONTERMINAL,\
    TERMINAL, string, output_helper, Enumerator, SDCP, grammar_to_SDCP

cdef extern from "Trainer/TrainingCommon.h":
    pass

cdef extern from "Trainer/GrammarInfo.h" namespace "Trainer":
    cdef cppclass GrammarInfo2:
        vector[vector[size_t]] rule_to_nonterminals

cdef extern from "Trainer/StorageManager.h" namespace "Trainer":
    cdef cppclass StorageManager:
         StorageManager(bint)

cdef extern from "Trainer/EMTrainer.h" namespace "Trainer":
    cdef cppclass EMTrainer[Nonterminal, TraceID]:
        vector[double] do_em_training[SemiRing](vector[double], vector[vector[unsigned_int]], unsigned)

cdef extern from "Trainer/LatentAnnotation.h" namespace "Trainer":
    cdef cppclass LatentAnnotation:
        LatentAnnotation(vector[size_t], vector[double], vector[vector[double]], GrammarInfo2, StorageManager)
        LatentAnnotation(vector[double], GrammarInfo2, StorageManager)
        vector[size_t] nonterminalSplits
        double get_weight(size_t, vector[size_t])
        vector[vector[double]] get_rule_weights()
        vector[double] get_root_weights()

cdef extern from "Trainer/SplitMergeTrainer.h" namespace "Trainer":
    cdef cppclass SplitMergeTrainer[Nonterminal, TraceID]:
        LatentAnnotation split_merge_cycle(LatentAnnotation)

cdef extern from "Trainer/EMTrainerLA.h":
    pass

cdef extern from "Trainer/MergePreparator.h":
    pass

cdef extern from "Trainer/TrainerBuilder.h" namespace "Trainer":
    cdef cppclass EMTrainerBuilder:
         EMTrainer[Nonterminal, TraceID] build_em_trainer[Nonterminal, TraceID](TraceManagerPtr[Nonterminal, TraceID])

    cdef cppclass SplitMergeTrainerBuilder[Nonterminal, TraceID]:
        SplitMergeTrainerBuilder(TraceManagerPtr[Nonterminal, TraceID], shared_ptr[GrammarInfo2])
        SplitMergeTrainerBuilder& set_simple_expector()
        SplitMergeTrainerBuilder& set_simple_expector(unsigned_int)
        SplitMergeTrainerBuilder& set_discriminative_expector(
                TraceManagerPtr[Nonterminal, TraceID] discriminativeTraceManager
                , double maxScale)
        SplitMergeTrainerBuilder& set_discriminative_expector(
                TraceManagerPtr[Nonterminal, TraceID] discriminativeTraceManager
                , double maxScale
                , unsigned_int threads)
        SplitMergeTrainerBuilder& set_simple_maximizer()
        SplitMergeTrainerBuilder& set_simple_maximizer(unsigned_int)
        SplitMergeTrainerBuilder& set_em_epochs(unsigned_int)
        SplitMergeTrainerBuilder& set_percent_merger(double)
        SplitMergeTrainerBuilder& set_percent_merger(double, unsigned_int)
        SplitMergeTrainerBuilder& set_threshold_merger(double)
        SplitMergeTrainerBuilder& set_threshold_merger(double, unsigned_int)
        SplitMergeTrainerBuilder& set_split_randomizer(double)
        SplitMergeTrainerBuilder& set_threads(unsigned_int)
        SplitMergeTrainer[Nonterminal, TraceID] build()

cdef extern from "DCP/util.h" namespace "DCP":
    cdef void add_trace_to_manager[Nonterminal, Terminal, Position, TraceID]\
        (SDCPParser[Nonterminal, Terminal, Position]
         , TraceManagerPtr[Nonterminal, TraceID])

cdef extern from "util.h":
    cdef cppclass Double
    cdef cppclass LogDouble

cdef class PySDCPTraceManager(PyTraceManager):
    cdef PySDCPParser parser

    def __init__(self, grammar, lcfrs_parsing=True, debug=False):
        """
        :param grammar:
        :type grammar: gl.LCFRS
        :param lcfrs_parsing:
        :type lcfrs_parsing:
        :param debug:
        :type debug:
        """
        output_helper("initializing PyTraceManager")

        cdef Enumerator nonterminal_map = Enumerator()
        cdef Enumerator terminal_map = Enumerator()
        nonterminal_encoder = (lambda s: nonterminal_map.object_index(s)) if ENCODE_NONTERMINALS else lambda s: str(s)
        terminal_encoder = (lambda s: terminal_map.object_index(s)) if ENCODE_TERMINALS else lambda s: str(s)

        self.parser = PySDCPParser(grammar, lcfrs_parsing, debug)
        self.parser.set_sdcp(grammar_to_SDCP(grammar, nonterminal_encoder, terminal_encoder, lcfrs_parsing))
        self.parser.set_terminal_map(terminal_map)
        self.parser.set_nonterminal_map(nonterminal_map)

        cdef vector[NONTERMINAL] node_labels = range(0, self.parser.nonterminal_map.counter)
        cdef vector[size_t] edge_labels = range(0, len(grammar.rule_index()))

        self.trace_manager = build_trace_manager_ptr[NONTERMINAL, size_t](
            make_shared[vector[NONTERMINAL]](node_labels)
            , make_shared[vector[size_t]](edge_labels)
            , False)

    def compute_reducts(self, corpus):
        start_time = time.time()
        for i, tree in enumerate(corpus):
            self.parser.clear()
            self.parser.set_input(tree)
            self.parser.do_parse()
            if self.parser.recognized():
                add_trace_to_manager[NONTERMINAL,TERMINAL,int,size_t](self.parser.parser[0],self.trace_manager)
                # self.parser.print_trace()

            if i % 100 == 0:
                output_helper(str(i) + ' ' + str(time.time() - start_time))

    cpdef Enumerator get_nonterminal_map(self):
        return self.parser.nonterminal_map

def compute_reducts(grammar, corpus, debug=False):
    output_helper("creating trace")
    trace = PySDCPTraceManager(grammar, debug=debug)
    output_helper("computing reducts")
    trace.compute_reducts(corpus)
    return trace

# choose representation: prob / log-prob
# ctypedef LogDouble SemiRing
ctypedef Double SemiRing

cdef class PyEMTrainer:
    cdef PyTraceManager traceManager

    def __init__(self, PyTraceManager traceManager):
        self.traceManager = traceManager

    def em_training(self, grammar, n_epochs, init="rfe", tie_breaking=False, sigma=0.005, seed=0):
        random.seed(seed)
        assert isinstance(grammar, gl.LCFRS)
        normalization_groups = []
        rule_to_group = {}
        for nont in grammar.nonts():
            normalization_group = []
            for rule in grammar.lhs_nont_to_rules(nont):
                rule_idx = rule.get_idx()
                normalization_group.append(rule_idx)
                rule_to_group[rule_idx] = self.traceManager.get_nonterminal_map().object_index(nont)
            normalization_groups.append(normalization_group)
        initial_weights = [0.0] * 0
        for i in range(0, len(grammar.rule_index())):
            if init == "rfe":
                prob = grammar.rule_index(i).weight()
            elif init == "equal" or True:
                prob = 1.0 / len(normalization_groups[rule_to_group[i]])

            # this may violates properness
            # but EM makes the grammar proper again
            if tie_breaking:
                prob_new = random.gauss(prob, sigma)
                while prob_new < 0.0:
                    prob_new = random.gauss(prob, sigma)
                prob = prob_new

            initial_weights.append(prob)

        cdef EMTrainerBuilder trainerBuilder
        cdef shared_ptr[EMTrainer[NONTERMINAL, size_t]] emTrainer \
            = make_shared[EMTrainer[NONTERMINAL, size_t]](trainerBuilder.build_em_trainer[NONTERMINAL, size_t](self.traceManager.trace_manager))

        final_weights = deref(emTrainer).do_em_training[SemiRing](initial_weights, normalization_groups, n_epochs)

        for i in range(0, len(grammar.rule_index())):
            grammar.rule_index(i).set_weight(final_weights[i])


cdef class PyGrammarInfo:
    cdef shared_ptr[GrammarInfo2] grammarInfo

    def __init__(self, grammar, Enumerator nont_map):
        """
        :type grammar: gl.LCFRS
        """
        cdef vector[vector[size_t]] rule_to_nonterminals = []
        cdef size_t i
        # for i in range(traceManager.parser.rule_map.first_index, traceManager.parser.rule_map.counter):
        for i in range(0, len(grammar.rule_index())):
            rule = grammar.rule_index(i)
            nonts = [nont_map.object_index(rule.lhs().nont())] + [nont_map.object_index(nont) for nont in rule.rhs()]
            rule_to_nonterminals.push_back(nonts)

        self.grammarInfo = make_shared[GrammarInfo2](rule_to_nonterminals, nont_map.object_index(grammar.start()))

cdef class PyStorageManager:
    cdef shared_ptr[StorageManager] storageManager
    def __init__(self, bint selfMalloc=False):
        self.storageManager = make_shared[StorageManager](selfMalloc)

cdef class PySplitMergeTrainerBuilder:
    cdef shared_ptr[SplitMergeTrainerBuilder[NONTERMINAL, size_t]] splitMergeTrainerBuilder
    def __init__(self, PyTraceManager traceManager, PyGrammarInfo grammarInfo):
        self.splitMergeTrainerBuilder = make_shared[SplitMergeTrainerBuilder[NONTERMINAL, size_t]](traceManager.trace_manager, grammarInfo.grammarInfo)

    cpdef PySplitMergeTrainerBuilder set_threads(self, unsigned_int threads):
        deref(self.splitMergeTrainerBuilder).set_threads(threads)
        return self

    cpdef PySplitMergeTrainerBuilder set_simple_expector(self, unsigned_int threads=0):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder).set_simple_expector(threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_simple_expector()
        return self

    cpdef PySplitMergeTrainerBuilder set_discriminative_expector(
            self
            , PyTraceManager discriminativeTraces
            , double maxScale = float("inf")
            , unsigned_int threads=0
    ):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder)\
                .set_discriminative_expector(discriminativeTraces.trace_manager, maxScale, threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_discriminative_expector(discriminativeTraces.trace_manager, maxScale)
        return self

    cpdef PySplitMergeTrainerBuilder set_simple_maximizer(self, unsigned_int threads=0):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder).set_simple_maximizer(threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_simple_maximizer()
        return self

    cpdef PySplitMergeTrainerBuilder set_em_epochs(self, size_t epochs):
        deref(self.splitMergeTrainerBuilder).set_em_epochs(epochs)
        return self

    cpdef PySplitMergeTrainerBuilder set_percent_merger(self, double percent=50.0, unsigned_int threads=0):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder).set_percent_merger(percent, threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_percent_merger(percent)
        return self

    cpdef PySplitMergeTrainerBuilder set_threshold_merger(self, double threshold, unsigned_int threads=0):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder).set_threshold_merger(threshold, threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_threshold_merger(threshold)
        return self

    cpdef PySplitMergeTrainer build(self):
        trainer = PySplitMergeTrainer()
        trainer.splitMergeTrainer = make_shared[SplitMergeTrainer[NONTERMINAL, size_t]](deref(self.splitMergeTrainerBuilder).build())
        return trainer

cdef class PyLatentAnnotation:
    cdef shared_ptr[LatentAnnotation] latentAnnotation
    cdef set_latent_annotation(self, shared_ptr[LatentAnnotation] la):
        self.latentAnnotation = la


    def build_sm_grammar(self, grammar, PyGrammarInfo grammarInfo, rule_pruning, rule_smoothing=0.0):
        new_grammar = gl.LCFRS(grammar.start() + "[0]")
        for i in range(0, len(grammar.rule_index())):
            rule = grammar.rule_index(i)

            rule_dimensions = [deref(self.latentAnnotation).nonterminalSplits[nont]
                               for nont in deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]]
            rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

            lhs_dims = deref(self.latentAnnotation).nonterminalSplits[
                deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
            ]

            for la in rule_dimensions_product:
                index = list(la)

                if rule_smoothing > 0.0:
                    weight_av = sum([deref(self.latentAnnotation).get_weight(i, [lhs] + list(la)[1:])
                        for lhs in range(lhs_dims)]) / lhs_dims

                    weight = (1 - rule_smoothing) * deref(self.latentAnnotation).get_weight(i, index) \
                             + rule_smoothing * weight_av
                else:
                    weight = deref(self.latentAnnotation).get_weight(i, index)
                if weight > rule_pruning:
                    lhs_la = gl.LCFRS_lhs(rule.lhs().nont() + "[" + str(la[0]) + "]")
                    for arg in rule.lhs().args():
                        lhs_la.add_arg(arg)
                    nonts = [rhs_nont + "[" + str(la[1 + j]) + "]" for j, rhs_nont in enumerate(rule.rhs())]
                    new_grammar.add_rule(lhs_la, nonts, weight, rule.dcp())

        return new_grammar

    cpdef tuple serialize(self):
        cdef vector[size_t] splits = deref(self.latentAnnotation).nonterminalSplits
        cdef vector[double] rootWeights = deref(self.latentAnnotation).get_root_weights()
        cdef vector[vector[double]] ruleWeights = deref(self.latentAnnotation).get_rule_weights()
        return splits, rootWeights, ruleWeights

cpdef PyLatentAnnotation build_PyLatentAnnotation(vector[size_t] nonterminalSplits
                                                  , vector[double] rootWeights
                                                  , vector[vector[double]] ruleWeights
                                                  , PyGrammarInfo grammarInfo
                                                  , PyStorageManager storageManager):
    cdef PyLatentAnnotation latentAnnotation = PyLatentAnnotation()
    latentAnnotation.latentAnnotation = make_shared[LatentAnnotation](nonterminalSplits
                                                                      , rootWeights
                                                                      , ruleWeights
                                                                      , deref(grammarInfo.grammarInfo)
                                                                      , deref(storageManager.storageManager))
    return latentAnnotation

cpdef PyLatentAnnotation build_PyLatentAnnotation_initial(
        grammar
        , PyGrammarInfo grammarInfo
        , PyStorageManager storageManager):
    cdef vector[double] ruleWeights = []
    cdef size_t i
    for i in range(0, len(grammar.rule_index())):
        rule = grammar.rule_index(i)
        assert(isinstance(rule, gl.LCFRS_rule))
        ruleWeights.push_back(rule.weight())
    # output_helper(str(ruleWeights) + "\n")
    cdef PyLatentAnnotation latentAnnotation = PyLatentAnnotation()
    latentAnnotation.latentAnnotation \
        = make_shared[LatentAnnotation](LatentAnnotation(ruleWeights
                                                         , deref(grammarInfo.grammarInfo)
                                                         , deref(storageManager.storageManager)))
    return latentAnnotation

cdef class PySplitMergeTrainer:
    cdef shared_ptr[SplitMergeTrainer[NONTERMINAL, size_t]] splitMergeTrainer

    cpdef PyLatentAnnotation split_merge_cycle (self, PyLatentAnnotation la):
        timeStart = time.time()
        cdef shared_ptr[LatentAnnotation] la_trained \
            = make_shared[LatentAnnotation](deref(self.splitMergeTrainer).split_merge_cycle(deref(la.latentAnnotation)))
        cdef PyLatentAnnotation pyLaTrained = PyLatentAnnotation()
        pyLaTrained.latentAnnotation = la_trained
        output_helper("Completed split/merge cycles in " + str(time.time() - timeStart) + " seconds")
        return pyLaTrained


def split_merge_training(grammar, corpus, cycles, em_epochs, init="rfe", tie_breaking=False, sigma=0.005, seed=0, merge_threshold=0.5, debug=False, rule_pruning=exp(-100)):
    output_helper("creating trace")
    trace = PySDCPTraceManager(grammar, debug=debug)
    output_helper("computing reducts")
    trace.compute_reducts(corpus)
    output_helper("pre em-training")
    emTrainer = PyEMTrainer(trace)
    emTrainer.em_training(grammar, em_epochs, init, tie_breaking, sigma, seed)
    output_helper("starting actual split/merge training")
    grammarInfo = PyGrammarInfo(grammar, trace.get_nonterminal_map())
    storageManager = PyStorageManager()
    las = [build_PyLatentAnnotation_initial(grammar, grammarInfo, storageManager)]

    trainer = PySplitMergeTrainerBuilder(trace, grammarInfo).set_em_epochs(em_epochs).set_threshold_merger(merge_threshold).build()

    for i in range(cycles):
        las.append(trainer.split_merge_cycle(las[i]))
        smGrammar = las[-1].build_sm_grammar(grammar, grammarInfo, rule_pruning)
        yield smGrammar