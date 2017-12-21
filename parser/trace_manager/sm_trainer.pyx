from libcpp.map cimport map
from libcpp.memory cimport make_shared
from cython.operator cimport dereference as deref
from libcpp.functional cimport function
from libcpp cimport bool as c_bool
from parser.commons.commons cimport *
from parser.trace_manager.trace_manager cimport PyTraceManager, TraceManagerPtr
from parser.trace_manager.sm_trainer_util cimport PyGrammarInfo, GrammarInfo2, PyStorageManager
from parser.trace_manager.score_validator cimport PyCandidateScoreValidator, CandidateScoreValidator
import time
import random
import grammar.lcfrs as gl
import itertools
from util.enumerator import Enumerator
from collections import defaultdict

DEF ENCODE_NONTERMINALS = True
DEF ENCODE_TERMINALS = True

DEF IO_PRECISION_DEFAULT = 0.000001
DEF IO_CYCLE_LIMIT_DEFAULT = 200

cdef extern from "Trainer/EMTrainer.h" namespace "Trainer":
    cdef cppclass EMTrainer[Nonterminal, TraceID]:
        vector[double] do_em_training[SemiRing](vector[double], vector[vector[unsigned_int]], unsigned)

cdef extern from "Trainer/SplitMergeTrainer.h" namespace "Trainer":
    cdef cppclass Splitter:
        void reset_random_seed(unsigned_int seed)
    cdef cppclass SplitMergeTrainer[Nonterminal, TraceID]:
        shared_ptr[Splitter] splitter
        LatentAnnotation split_merge_cycle(LatentAnnotation)
        LatentAnnotation merge(LatentAnnotation)
        void em_train(LatentAnnotation)
        vector[vector[vector[size_t]]] get_current_merge_sources()

cdef extern from "Trainer/EMTrainerLA.h" namespace "Trainer":
    ctypedef enum TrainingMode:
        Default,
        Splitting,
        Merging,
        Smoothing

    cdef cppclass EMTrainerLA:
        void setEMepochs(unsigned epochs, TrainingMode mode)
    cdef cppclass EMTrainerLAValidation(EMTrainerLA):
        void setMaxDrops(unsigned maxDrops, TrainingMode mode)
        void setMinEpochs(unsigned minEpochs, TrainingMode mode)

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
        SplitMergeTrainerBuilder& set_no_count_modification()
        SplitMergeTrainerBuilder& set_count_smoothing(vector[size_t] ruleIDs, double smoothValue)
        SplitMergeTrainerBuilder& set_simple_validator(
                TraceManagerPtr[Nonterminal, TraceID] discriminativeTraceManager
                , unsigned_int maxDrops)
        SplitMergeTrainerBuilder& set_simple_validator(
                TraceManagerPtr[Nonterminal, TraceID] discriminativeTraceManager
                , unsigned_int maxDrops
                , unsigned_int threads)
        SplitMergeTrainerBuilder& set_score_validator(shared_ptr[CandidateScoreValidator[Nonterminal, TraceID]], unsigned)
        SplitMergeTrainerBuilder& set_percent_merger(double)
        SplitMergeTrainerBuilder& set_percent_merger(double, unsigned)
        SplitMergeTrainerBuilder& set_threshold_merger(double)
        SplitMergeTrainerBuilder& set_threshold_merger(double, unsigned)
        SplitMergeTrainerBuilder& set_scc_merger(double threshold)
        SplitMergeTrainerBuilder& set_scc_merger(double threshold, unsigned threads)
        SplitMergeTrainerBuilder& set_scc_merger(double threshold, vector[size_t] relevantNonterminals, unsigned threads)
        SplitMergeTrainerBuilder& set_scc_merge_threshold_function(function[double(vector[double])])
        SplitMergeTrainerBuilder& set_split_randomization(double, unsigned)
        SplitMergeTrainerBuilder& set_smoothing_factor(double smoothingFactor, double smoothingFactorUnary)
        SplitMergeTrainerBuilder& set_threads(unsigned_int)
        SplitMergeTrainer[Nonterminal, TraceID] build()
        shared_ptr[EMTrainerLA] getEmTrainer()

    function[double(vector[double])] interpolate3rdQuartileMax(double factor)

cdef extern from "Trainer/AnnotationProjection.h" namespace "Trainer":
    cdef const double IO_PRECISION_DEFAULT
    cdef const double IO_CYCLE_LIMIT_DEFAULT
    cdef LatentAnnotation project_annotation[Nonterminal](const LatentAnnotation & annotation,
                                                          const GrammarInfo2 & grammarInfo,
                                                          const double ioPrecision,
                                                          const size_t ioCycleLimit,
                                                          const c_bool debug)
    cdef LatentAnnotation project_annotation_by_merging[Nonterminal](const LatentAnnotation & annotation,
                                                                     const GrammarInfo2 & grammarInfo,
                                                                     const vector[vector[vector[size_t]]] & merge_sources,
                                                                     double ioPrecision,
                                                                     double ioCycleLimit,
                                                                     const bint debug)

cdef extern from "Trainer/GeneticCrosser.h" namespace "Trainer":
    cdef LatentAnnotation mix_annotations[Nonterminal](const LatentAnnotation& la1
            , const LatentAnnotation& la2
            , const GrammarInfo2& info
            , const vector[c_bool]& keepFromOne
            , const double ioPrecision
            , const unsigned_int ioCycleLimit
    )

cdef extern from "util.h":
    cdef cppclass Double
    cdef cppclass LogDouble
    cdef void output_helper(string)

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
            # but we make the grammar proper again soon
            if tie_breaking:
                prob_new = random.gauss(prob, sigma)
                while prob_new <= 0.0:
                    prob_new = random.gauss(prob, sigma)
                prob = prob_new

            initial_weights.append(prob)

        # restore properness
        if tie_breaking:
            for group in normalization_groups:
                group_sum = 0.0
                for idx in group:
                    group_sum += initial_weights[idx]
                if group_sum > 0:
                    for idx in group:
                        initial_weights[idx] = initial_weights[idx] / group_sum
                else:
                    for idx in group:
                        initial_weights[idx] = 1 / len(group)

        cdef EMTrainerBuilder trainerBuilder
        cdef shared_ptr[EMTrainer[NONTERMINAL, size_t]] emTrainer \
            = make_shared[EMTrainer[NONTERMINAL, size_t]](trainerBuilder.build_em_trainer[NONTERMINAL, size_t](self.traceManager.trace_manager))

        final_weights = deref(emTrainer).do_em_training[SemiRing](initial_weights, normalization_groups, n_epochs)

        # ensure properness
        if tie_breaking:
            for group in normalization_groups:
                group_sum = 0.0
                for idx in group:
                    group_sum += final_weights[idx]
                if group_sum > 0:
                    for idx in group:
                        final_weights[idx] = final_weights[idx] / group_sum
                else:
                    for idx in group:
                        final_weights[idx] = 1 / len(group)

        for i in range(0, len(grammar.rule_index())):
            grammar.rule_index(i).set_weight(final_weights[i])


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

    cpdef PySplitMergeTrainerBuilder set_no_count_modification(self):
        deref(self.splitMergeTrainerBuilder).set_no_count_modification()
        return self

    cpdef PySplitMergeTrainerBuilder set_count_smoothing(self, vector[size_t] ruleIDs, double smoothValue):
        deref(self.splitMergeTrainerBuilder).set_count_smoothing(ruleIDs, smoothValue)
        return self

    cpdef PySplitMergeTrainerBuilder set_simple_validator(
            self
            , PyTraceManager discriminativeTraces
            , unsigned_int maxDrops=6
            , unsigned_int threads=0
    ):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder)\
                .set_simple_validator(discriminativeTraces.trace_manager, maxDrops, threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_simple_validator(discriminativeTraces.trace_manager, maxDrops)
        return self

    cpdef PySplitMergeTrainerBuilder set_score_validator(
            self
            , PyCandidateScoreValidator validator
            , unsigned_int maxDrops=6
    ):
        deref(self.splitMergeTrainerBuilder).set_score_validator(validator.validator, maxDrops)
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

    cpdef PySplitMergeTrainerBuilder set_scc_merger(self, double threshold, unsigned threads=0, vector[size_t] relevantNonterminals=[]):
        if threads > 0:
            if len(relevantNonterminals) > 0:
                deref(self.splitMergeTrainerBuilder).set_scc_merger(threshold, relevantNonterminals, threads)
            else:
                deref(self.splitMergeTrainerBuilder).set_scc_merger(threshold, threads)
        else:
            if len(relevantNonterminals) > 0:
                deref(self.splitMergeTrainerBuilder).set_scc_merger(threshold, relevantNonterminals, 1)
            else:
                deref(self.splitMergeTrainerBuilder).set_scc_merger(threshold)
        return self

    cpdef PySplitMergeTrainerBuilder set_scc_merge_threshold_function(self, double factor):
        deref(self.splitMergeTrainerBuilder).set_scc_merge_threshold_function(interpolate3rdQuartileMax(factor))
        return self

    cpdef PySplitMergeTrainerBuilder set_split_randomization(self, double percent=1.0, unsigned seed=0):
        deref(self.splitMergeTrainerBuilder).set_split_randomization(percent, seed)
        return self

    cpdef PySplitMergeTrainerBuilder set_smoothing_factor(self,
                                                          double smoothingFactor=0.01,
                                                          double smoothingFactorUnary=0.1):
        deref(self.splitMergeTrainerBuilder).set_smoothing_factor(smoothingFactor, smoothingFactorUnary)
        return self

    cpdef PySplitMergeTrainer build(self):
        trainer = PySplitMergeTrainer()
        trainer.splitMergeTrainer = make_shared[SplitMergeTrainer[NONTERMINAL, size_t]](deref(self.splitMergeTrainerBuilder).build())
        trainer.emTrainer = (deref(self.splitMergeTrainerBuilder)).getEmTrainer()
        return trainer

cdef class PyLatentAnnotation:
    cdef set_latent_annotation(self, shared_ptr[LatentAnnotation] la):
        self.latentAnnotation = la

    cpdef void add_random_noise(self, PyGrammarInfo grammarInfo, double randPercent = 1.0, size_t seed=0, double bias=0.01):
        deref(self.latentAnnotation).add_random_noise(grammarInfo.grammarInfo, randPercent, seed, bias)

    def project_weights(self, grammar, PyGrammarInfo grammarInfo, debug=False):
        trivial_split = True
        cdef size_t nont

        for nont in range(len(deref(self.latentAnnotation).nonterminalSplits)):
            if deref(self.latentAnnotation).nonterminalSplits[nont] > 1:
                trivial_split = False
                break

        cdef shared_ptr[LatentAnnotation] la_proj
        if trivial_split:
            la_proj = self.latentAnnotation
        else:
            # guarantee properness:
            for nont in range(len(deref(grammarInfo.grammarInfo).normalizationGroups)):
                group = deref(grammarInfo.grammarInfo).normalizationGroups[nont]
                split_total_probs = [0.0] * (deref(self.latentAnnotation).nonterminalSplits[nont])
                for i in group:
                    rule_dimensions = [deref(self.latentAnnotation).nonterminalSplits[_nont]
                                       for _nont in deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]]
                    rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

                    lhs_dims = deref(self.latentAnnotation).nonterminalSplits[
                        deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
                    ]

                    for la in rule_dimensions_product:
                        index = list(la)
                        weight = deref(self.latentAnnotation).get_weight(i, index)
                        if not weight >= 0.0 and weight <= 1.0001:
                            output_helper("Weight not in range: " + str(weight))
                            assert weight >= 0.0 and weight <= 1.0001
                        split_total_probs[la[0]] += weight
                if not all([ abs(x - 1.0) <= 0.0001 for x in split_total_probs]):
                    output_helper(str(nont) + " " + str(split_total_probs))
                    raise Exception(nont, split_total_probs)

            la_proj = make_shared[LatentAnnotation](project_annotation[NONTERMINAL](deref(self.latentAnnotation),
                                                                                    deref(grammarInfo.grammarInfo),
                                                                                    IO_PRECISION_DEFAULT,
                                                                                    IO_CYCLE_LIMIT_DEFAULT,
                                                                                    debug))

            # guarantee properness:
            for nont in range(len(deref(grammarInfo.grammarInfo).normalizationGroups)):
                group = deref(grammarInfo.grammarInfo).normalizationGroups[nont]
                split_total_probs = [0.0] * (deref(la_proj).nonterminalSplits[nont])
                for i in group:
                    rule_dimensions = [deref(la_proj).nonterminalSplits[_nont]
                                       for _nont in deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]]
                    rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

                    lhs_dims = deref(la_proj).nonterminalSplits[
                        deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
                    ]

                    for la in rule_dimensions_product:
                        index = list(la)
                        weight = deref(la_proj).get_weight(i, index)
                        if not weight >= 0.0 and weight <= 1.0001:
                            output_helper(str(weight))
                            assert weight >= 0.0 and weight <= 1.0001
                        split_total_probs[la[0]] += weight
                if not all([ abs(x - 1.0) <= 0.0001 for x in split_total_probs]):
                    output_helper(str(split_total_probs))
                    if not all([ abs(x - 1.0) <= 0.1 for x in split_total_probs]):
                        output_helper("Error: Grammar is not proper!")
                        for i in group:
                            rule_dimensions = [deref(la_proj).nonterminalSplits[_nont]
                                               for _nont in deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]]
                            rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

                            lhs_dims = deref(la_proj).nonterminalSplits[
                                deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
                            ]

                            for la in rule_dimensions_product:
                                index = list(la)
                                weight = deref(la_proj).get_weight(i, index)
                                output_helper(str(i) + " " + str(index) + " " + str(weight))
                        raise

        for i in range(0, len(grammar.rule_index())):
            rule = grammar.rule_index(i)

            rule_dimensions = [deref(la_proj).nonterminalSplits[nont]
                               for nont in deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]]
            rule_dimensions_product = [la for la in itertools.product(*[range(dim) for dim in rule_dimensions])]

            lhs_dims = deref(la_proj).nonterminalSplits[
                deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
            ]

            assert len(rule_dimensions_product) == 1
            index = list(rule_dimensions_product[0])
            weight = deref(la_proj).get_weight(i, index)
            rule.set_weight(weight)


    def project_annotation_by_merging(self,
                                                           PyGrammarInfo grammarInfo,
                                                           vector[vector[vector[size_t]]] merge_sources,
                                                           c_bool debug=False):
        cdef shared_ptr[LatentAnnotation] la_projected\
            = make_shared[LatentAnnotation](project_annotation_by_merging[NONTERMINAL](deref(self.latentAnnotation), deref(grammarInfo.grammarInfo), merge_sources, IO_PRECISION_DEFAULT, IO_CYCLE_LIMIT_DEFAULT, debug))
        cdef PyLatentAnnotation pyLaProjected = PyLatentAnnotation()
        pyLaProjected.latentAnnotation = la_projected


        # guarantee properness:
        for nont in range(len(deref(grammarInfo.grammarInfo).normalizationGroups)):
            group = deref(grammarInfo.grammarInfo).normalizationGroups[nont]
            split_total_probs = [0.0] * (deref(la_projected).nonterminalSplits[nont])
            for i in group:
                rule_dimensions = [deref(la_projected).nonterminalSplits[_nont]
                                   for _nont in deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]]
                rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

                lhs_dims = deref(la_projected).nonterminalSplits[
                    deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
                ]

                for la in rule_dimensions_product:
                    index = list(la)
                    weight = deref(la_projected).get_weight(i, index)
                    if not weight >= 0.0 and weight <= 1.0001:
                        output_helper(str(weight))
                        assert weight >= 0.0 and weight <= 1.0001
                    split_total_probs[la[0]] += weight
            if not all([ abs(x - 1.0) <= 0.0001 for x in split_total_probs]):
                output_helper(str(split_total_probs))
                if not all([ abs(x - 1.0) <= 0.1 for x in split_total_probs]):
                    for i in group:
                        rule_dimensions = [deref(la_projected).nonterminalSplits[_nont]
                                       for _nont in deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]]
                        rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

                        lhs_dims = deref(la_projected).nonterminalSplits[
                            deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
                        ]

                        for la in rule_dimensions_product:
                            index = list(la)
                            weight = deref(la_projected).get_weight(i, index)
                            output_helper(str(i) + " " + str(index) + " " + str(weight))
                    raise Exception()

        return pyLaProjected


    cpdef genetic_recombination(self, PyLatentAnnotation otherAnnotation
                        , PyGrammarInfo info
                        , vector[c_bool] keepFromOne
                        , double ioPrecision
                        , unsigned_int ioCycleLimit
                        ):
        cdef shared_ptr[LatentAnnotation] la_trained \
                    = make_shared[LatentAnnotation](
                                  mix_annotations[NONTERMINAL](deref(self.latentAnnotation)
                                                              , deref(otherAnnotation.latentAnnotation)
                                                              , deref(info.grammarInfo)
                                                              , keepFromOne
                                                              , IO_PRECISION_DEFAULT
                                                              , IO_CYCLE_LIMIT_DEFAULT
                                                              )
                              )
        cdef PyLatentAnnotation pyLaTrained = PyLatentAnnotation()
        pyLaTrained.latentAnnotation = la_trained

        return pyLaTrained

    cpdef c_bool check_for_validity(self, double delta = 0.0005):
        return deref(self.latentAnnotation).check_for_validity(delta)

    cpdef c_bool is_proper(self, PyGrammarInfo info):
        return deref(self.latentAnnotation).is_proper(info.grammarInfo);


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

    def construct_fine_grammar(self, grammar, PyGrammarInfo grammarInfo, arg_transform, PyLatentAnnotation la_full,
                               smooth_transform=None, smooth_weight=0.01):
        """
        :param grammar:
        :type grammar: gl.LCFRS
        :type grammarInfo: PyGrammarInfo
        :return:
        :rtype:
        """
        nont_translation = defaultdict(lambda: -1)
        nonterminals = Enumerator()

        def rename(nont, split_id, nont_id):
            if deref(self.latentAnnotation).nonterminalSplits[nont_id] == 1:
                idx = nonterminals.object_index(nont)
                nont_translation[idx] = nont_id
                return nont
            else:
                new_nont = str(nont) + "[" + str(split_id) + str("]")
                nonterminals.object_index(new_nont)
                return new_nont

        new_grammar = gl.LCFRS(grammar.start())

        cdef vector[double] root_weights = deref(la_full.latentAnnotation).get_root_weights()
        cdef vector[size_t] smooth_rules = []
        latent_rule_weights = defaultdict(lambda: defaultdict(lambda: 0.0))

        for i in range(0, len(grammar.rule_index())):
            rule = grammar.rule_index(i)
            # nonts = [rule.lhs().nont()] + rule.rhs()
            nont_ids = deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]
            rule_dimensions = []

            for nont_id in nont_ids:
                rule_dimensions.append(deref(self.latentAnnotation).nonterminalSplits[nont_id])

            rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

            # lhs_dims = deref(self.latentAnnotation).nonterminalSplits[
            #     deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
            # ]

            rule_dimensions_full = []
            for nont_id in nont_ids:
                rule_dimensions_full.append(deref(la_full.latentAnnotation).nonterminalSplits[nont_id])

            for la in rule_dimensions_product:
                index = list(la)
                weight = deref(self.latentAnnotation).get_weight(i, index)
                if weight > 0.0:
                    lhs_la = gl.LCFRS_lhs(rename(rule.lhs().nont(), la[0], nont_ids[0]))
                    for arg in rule.lhs().args():
                        lhs_la.add_arg(arg_transform(arg, la))
                    nonts = [rename(rhs_nont, la[1 + j], nont_ids[1 + j]) for j, rhs_nont in enumerate(rule.rhs())]
                    new_rule = new_grammar.add_rule(lhs_la, nonts, weight, rule.dcp())

                    product_range = []
                    mask = []
                    for i2, j2, la_idx in zip(rule_dimensions, rule_dimensions_full, index):
                        if i2 < j2:
                            product_range.append([x for x in range(j2)])
                            mask.append(False)
                        else:
                            product_range.append([la_idx])
                            mask.append(True)

                    for laf in itertools.product(*product_range):
                        laf_masked = tuple([0 if mb else laf[mi] for mi, mb in enumerate(mask)])
                        latent_rule_weights[new_rule.get_idx()][laf_masked] \
                            = deref(la_full.latentAnnotation).get_weight(i, list(laf))

                    if nonts == [] and smooth_transform is not None:
                        # smoothing part
                        lhs_smooth = gl.LCFRS_lhs(rename(rule.lhs().nont(), la[0], nont_ids[0]))
                        for arg in rule.lhs().args():
                            lhs_smooth.add_arg(smooth_transform(arg))
                        new_rule = new_grammar.add_rule(lhs_smooth, [], smooth_weight, rule.dcp())
                        smooth_rules.push_back(new_rule.get_idx())

                        for laf in itertools.product(*product_range):
                            laf_masked = tuple([0 if mb else laf[mi] for mi, mb in enumerate(mask)])
                            latent_rule_weights[new_rule.get_idx()][laf_masked] \
                                = smooth_weight

        cdef PyGrammarInfo new_grammar_info = PyGrammarInfo(new_grammar, nonterminals)
        cdef vector[size_t] nonterminal_splits = []
        for nont_id in range(0, nonterminals.get_counter()):
            old_idx = nont_translation[nont_id]
            if old_idx == -1:
                nonterminal_splits.push_back(1)

            else:
                nonterminal_splits.push_back(deref(la_full.latentAnnotation).nonterminalSplits[old_idx])

        cdef vector[vector[double]] rule_weights = []

        for idx, nonts in enumerate(deref(new_grammar_info.grammarInfo).rule_to_nonterminals):
            weights = []
            splits = [nonterminal_splits[nont] for nont in nonts]
            for la in itertools.product(*[range(s) for s in splits]):
                weights.append(latent_rule_weights[idx][la])
            rule_weights.push_back(weights)

        cdef PyStorageManager storage_manager = PyStorageManager()
        la_new_grammar = build_PyLatentAnnotation(nonterminal_splits, root_weights, rule_weights, new_grammar_info,
                                                  storage_manager)
        return new_grammar, la_new_grammar, new_grammar_info, nonterminals, nont_translation, smooth_rules

    cpdef tuple serialize(self):
        cdef vector[size_t] splits = deref(self.latentAnnotation).nonterminalSplits
        cdef vector[double] rootWeights = deref(self.latentAnnotation).get_root_weights()
        cdef vector[vector[double]] ruleWeights = deref(self.latentAnnotation).get_rule_weights()
        return splits, rootWeights, ruleWeights

    cpdef void make_proper(self, PyGrammarInfo grammarInfo):
        deref(self.latentAnnotation).make_proper(grammarInfo.grammarInfo)



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
        = make_shared[LatentAnnotation](ruleWeights
                                        , deref(grammarInfo.grammarInfo)
                                        , deref(storageManager.storageManager))
    return latentAnnotation

cdef class PySplitMergeTrainer:
    cdef map[string,TrainingMode] modes
    cdef shared_ptr[SplitMergeTrainer[NONTERMINAL, size_t]] splitMergeTrainer
    cdef shared_ptr[EMTrainerLA] emTrainer

    def __init__(self):
        modes_ = { b"default": Default
                 , b"splitting": Splitting
                 , b"merging": Merging
                 , b"smoothing": Smoothing}
        self.modes = modes_

    cpdef PyLatentAnnotation split_merge_cycle (self, PyLatentAnnotation la):
        timeStart = time.time()
        cdef shared_ptr[LatentAnnotation] la_trained \
            = make_shared[LatentAnnotation](deref(self.splitMergeTrainer).split_merge_cycle(deref(la.latentAnnotation)))
        cdef PyLatentAnnotation pyLaTrained = PyLatentAnnotation()
        pyLaTrained.latentAnnotation = la_trained
        output_helper(b"Completed split/merge cycles in " + bytes(str(time.time() - timeStart), encoding="utf-8") + b" seconds")
        return pyLaTrained

    cpdef PyLatentAnnotation merge(self, PyLatentAnnotation la):
        cdef shared_ptr[LatentAnnotation] la_merged \
            = make_shared[LatentAnnotation](deref(self.splitMergeTrainer).merge(deref(la.latentAnnotation)))
        cdef PyLatentAnnotation pyLaMerged = PyLatentAnnotation()
        pyLaMerged.latentAnnotation = la_merged
        return pyLaMerged

    cpdef void em_train(self, PyLatentAnnotation la):
        deref(self.splitMergeTrainer).em_train(deref(la.latentAnnotation))

    cpdef reset_random_seed(self, unsigned seed):
        deref(deref(self.splitMergeTrainer).splitter).reset_random_seed(seed)

    cpdef setEMepochs(self, unsigned epochs, mode="default"):
        deref(self.emTrainer).setEMepochs(epochs, self.modes.at(bytes(mode, encoding="utf-8")))

    cpdef setMaxDrops(self, unsigned maxDrops, mode="default"):
        (<EMTrainerLAValidation &> deref(self.emTrainer)).setMaxDrops(maxDrops, self.modes.at(bytes(mode, encoding="utf-8")))

    cpdef setMinEpochs(self, unsigned epochs, mode="default"):
        (<EMTrainerLAValidation &> deref(self.emTrainer)).setMinEpochs(epochs, self.modes.at(bytes(mode, encoding="utf-8")))

    cpdef get_current_merge_sources(self):
        return deref(self.splitMergeTrainer).get_current_merge_sources()
