from libcpp.vector cimport vector
from libcpp.memory cimport make_shared, shared_ptr
from parser.trace_manager.trace_manager cimport PyTraceManager, TraceManagerPtr, build_trace_manager_ptr
from util.enumerator cimport Enumerator

ctypedef size_t NONTERMINAL

cdef class PyDerivationManager(PyTraceManager):
    cdef shared_ptr[vector[NONTERMINAL]] node_labels
    cdef shared_ptr[vector[size_t]] edge_labels
    cdef Enumerator nonterminal_map

    cpdef void convert_hypergraphs(self, corpus)

    cpdef Enumerator get_nonterminal_map(self)