from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

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

cdef class PyLatentAnnotation:
    cdef shared_ptr[LatentAnnotation] latentAnnotation
    cdef set_latent_annotation(self, shared_ptr[LatentAnnotation] la)
    cpdef tuple serialize(self)
