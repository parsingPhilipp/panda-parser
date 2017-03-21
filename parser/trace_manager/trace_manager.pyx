cdef class PyTraceManager:
    cpdef serialize(self, string path):
        serialize_trace(self.trace_manager, path)

    cpdef void load_traces_from_file(self, string path):
        cdef TraceManagerPtr[NONTERMINAL, size_t] tm = load_trace_manager[NONTERMINAL, size_t](path)
        self.trace_manager = tm

    cpdef Enumerator get_nonterminal_map(self):
        pass