from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, make_shared
from libcpp.string cimport string

cdef extern from "Trainer/TraceManager.h" namespace "Trainer":
    cdef cppclass TraceManagerPtr[Nonterminal, TraceID]:
        pass
    cdef TraceManagerPtr[Nonterminal, TraceID] build_trace_manager_ptr[Nonterminal, TraceID](
            shared_ptr[vector[Nonterminal]]
            , shared_ptr[vector[size_t]]
            , bint)
    cdef void serialize_trace[Nonterminal, TraceID](TraceManagerPtr[Nonterminal, TraceID] traceManager, string path)
    cdef TraceManagerPtr[Nonterminal, TraceID] load_trace_manager[Nonterminal, TraceID](string path)

ctypedef size_t NONTERMINAL

cdef class PyTraceManager:
    cdef TraceManagerPtr[NONTERMINAL, size_t] trace_manager
    cpdef serialize(self, string path)
    cpdef void load_traces_from_file(self, string path)