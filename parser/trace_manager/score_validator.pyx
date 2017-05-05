from libcpp.memory cimport make_shared
from cython.operator cimport dereference as deref
from parser.trace_manager.sm_trainer_util cimport PyGrammarInfo, PyStorageManager
from parser.trace_manager.trace_manager cimport PyTraceManager, TraceManagerPtr, NONTERMINAL

cdef extern from "Trainer/Validation.h" namespace "Trainer":
    cdef cppclass CandidateScoreValidator[Nonterminal, TraceID]:
        CandidateScoreValidator(shared_ptr[GrammarInfo2] grammarInfo
                                , shared_ptr[StorageManager] storageManager
                                , string quantity
                                , double minimumScore)
        void add_scored_candidates(TraceManagerPtr[Nonterminal, TraceID], vector[double], double)

cdef class PyCandidateScoreValidator:
    def __init__(self, PyGrammarInfo gi, PyStorageManager sm, string quantity="score", double minScore= - float("inf")):
        self.validator = make_shared[CandidateScoreValidator[NONTERMINAL, size_t]](
                            gi.grammarInfo
                            , sm.storageManager
                            , quantity
                            , minScore)

    cpdef void add_scored_candidates(self, PyTraceManager traces, vector[double] scores, double maxScore):
        deref(self.validator).add_scored_candidates(traces.trace_manager, scores, maxScore)