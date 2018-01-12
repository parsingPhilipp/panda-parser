from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp cimport bool as c_bool
from parser.commons.commons cimport unsigned_int
from parser.trace_manager.sm_trainer_util cimport PyGrammarInfo, GrammarInfo2

cdef extern from "Trainer/TrainingCommon.h":
    pass

cdef extern from "Trainer/LatentAnnotation.h" namespace "Trainer":
    cdef cppclass LatentAnnotation:
        LatentAnnotation(vector[size_t], vector[double], vector[vector[double]], GrammarInfo2, StorageManager)
        LatentAnnotation(vector[double], GrammarInfo2, StorageManager)
        vector[size_t] nonterminalSplits
        double get_weight(size_t, vector[size_t])
        vector[vector[double]] get_rule_weights()
        vector[double] get_root_weights()
        void add_random_noise(double randPercent, size_t seed, double bias)
        void make_proper()
        c_bool is_proper()
        c_bool check_for_validity(double delta)
        c_bool check_rule_split_alignment()


cdef class PyLatentAnnotation:
    cdef shared_ptr[LatentAnnotation] latentAnnotation
    cpdef void add_random_noise(self, double randPercent=?, size_t seed=?, double bias=?)
    cdef set_latent_annotation(self, shared_ptr[LatentAnnotation] la)
    cpdef genetic_recombination(self, PyLatentAnnotation otherAnnotation
                        , PyGrammarInfo info
                        , vector[c_bool] keepFromOne
                        , double ioPrecision
                        , unsigned_int ioCycleLimit)
    cpdef tuple serialize(self)
    # cpdef PyLatentAnnotation project_annotation_by_merging(self,
    #                                                        PyGrammarInfo grammarInfo,
    #                                                        vector[vector[vector[size_t]]] merge_sources,
    #                                                        c_bool debug)
    cpdef void make_proper(self)
    cpdef c_bool is_proper(self)
    cpdef c_bool check_for_validity(self, double delta = *)
    cpdef c_bool check_rule_split_alignment(self)
