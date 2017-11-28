from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool as c_bool

cdef extern from "Trainer/GrammarInfo.h" namespace "Trainer":
    cdef cppclass GrammarInfo2:
        vector[vector[size_t]] rule_to_nonterminals
        const vector[vector[size_t]] normalizationGroups
        c_bool check_for_consistency()

cdef extern from "Trainer/StorageManager.h" namespace "Trainer":
    cdef cppclass StorageManager:
         StorageManager(bint)

cdef class PyGrammarInfo:
    cdef shared_ptr[GrammarInfo2] grammarInfo
    cpdef c_bool check_for_consistency(self)

cdef class PyStorageManager:
    cdef shared_ptr[StorageManager] storageManager

