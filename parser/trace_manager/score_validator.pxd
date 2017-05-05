from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from parser.trace_manager.sm_trainer_util cimport GrammarInfo2, StorageManager
from parser.trace_manager.trace_manager cimport PyTraceManager, TraceManagerPtr, NONTERMINAL


cdef extern from "Trainer/Validation.h" namespace "Trainer":
    cdef cppclass CandidateScoreValidator[Nonterminal, TraceID]:
        CandidateScoreValidator(shared_ptr[GrammarInfo2] grammarInfo
                                , shared_ptr[StorageManager] storageManager
                                , string quantity
                                , double minimumScore)
        void add_scored_candidates(TraceManagerPtr[Nonterminal, TraceID], vector[double], double)

cdef class PyCandidateScoreValidator:
    cdef shared_ptr[CandidateScoreValidator[NONTERMINAL, size_t]] validator
    cpdef void add_scored_candidates(self, PyTraceManager traces, vector[double] scores, double maxScore)