from libcpp.memory cimport shared_ptr, make_shared
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from util.enumerator cimport Enumerator
from parser.supervised_trainer.trainer cimport PyDerivationManager, NONTERMINAL
from parser.trace_manager.trace_manager cimport TraceManagerPtr, PyTraceManager
from parser.trace_manager.sm_trainer_util cimport PyStorageManager, StorageManager, PyGrammarInfo, GrammarInfo2
from parser.trace_manager.sm_trainer cimport PyLatentAnnotation, LatentAnnotation


cdef extern from "Trainer/TrainingCommon.h" namespace "Trainer":
    pass

cdef extern from "Trainer/HypergraphRanker.h" namespace "Trainer":
    cdef cppclass HypergraphRanker[Nonterminal, TraceID]:
        HypergraphRanker(TraceManagerPtr[Nonterminal, TraceID] traceManager
                        , shared_ptr[GrammarInfo2] grammarInfo
                        , shared_ptr[StorageManager] storageManager)
        vector[pair[size_t, double]] rank(LatentAnnotation& la)
        void clean_up()

cdef class PyHypergraphRanker:
    cdef shared_ptr[HypergraphRanker[NONTERMINAL, size_t]] ranker
    def __init__(self, PyTraceManager traceManager, PyGrammarInfo grammarInfo, PyStorageManager storageManager):
        # assert isinstance(traceManager, PyTraceManager)
        # assert isinstance(grammarInfo, PyGrammarInfo)
        # assert isinstance(storageManager, PyStorageManager)
        self.ranker = make_shared[HypergraphRanker[NONTERMINAL, size_t]]( #HypergraphRanker[NONTERMINAL, size_t](
            traceManager.trace_manager
            , grammarInfo.grammarInfo
            , storageManager.storageManager)#)

    cpdef vector[pair[size_t, double]] rank(self, PyLatentAnnotation la):
        return deref(self.ranker).rank(deref(la.latentAnnotation))

    cpdef void clean_up(self):
        deref(self.ranker).clean_up()

def build_ranker(derivations, grammar, PyGrammarInfo grammarInfo, Enumerator nonterminal_map):
    manager = PyDerivationManager(grammar, nonterminal_map)
    manager.convert_derivations_to_hypergraphs(derivations)
    ranker = PyHypergraphRanker(manager, grammarInfo, PyStorageManager())
    return ranker

def rank_derivations(derivations, PyLatentAnnotation la, grammar, PyGrammarInfo grammarInfo, Enumerator nonterminal_map):
    ranker = build_ranker(derivations, grammar, grammarInfo, nonterminal_map)
    return ranker.rank(la)
