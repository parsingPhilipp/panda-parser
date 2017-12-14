from cython.operator cimport dereference as deref
from libcpp.vector cimport vector

from parser.trace_manager.trace_manager cimport PyTraceManager, TraceManagerPtr, build_trace_manager_ptr, Element, Hypergraph, Node, HyperEdge, Trace, fool_cython_unwrap
from parser.trace_manager.sm_trainer cimport LatentAnnotation, PyLatentAnnotation


cdef extern from "Trainer/AnnotationProjection.h" namespace "Trainer":
    cdef vector[double] edge_weight_projection[Nonterminal](
          const LatentAnnotation &annotation
        , const Trace[Nonterminal, size_t]& trace
        , const bint variational
    )


cpdef vector[double] py_edge_weight_projection(PyLatentAnnotation annotation, PyTraceManager traceManager, size_t traceId=0, bint variational=False):
    return edge_weight_projection(deref(annotation.latentAnnotation), deref(fool_cython_unwrap(traceManager.trace_manager))[traceId], variational)
