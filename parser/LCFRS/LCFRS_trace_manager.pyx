from parser.commons.commons cimport *
from LCFRS_Parser_Wrapper cimport LCFRS_Parser, PyLCFRSFactory, PyLCFRSParser, Enumerator
from libcpp.memory cimport make_shared
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
import time
from parser.trace_manager.trace_manager cimport PyTraceManager, build_trace_manager_ptr, TraceManagerPtr


cdef extern from "LCFR/manager_util.h":
    cdef void add_trace_to_manager[Nonterminal, Terminal, TraceID](
            LCFRS_Parser[Nonterminal, Terminal] parser
            , TraceManagerPtr[Nonterminal, TraceID] traceManager)


cdef class PyLCFRSTraceManager(PyTraceManager):
    cdef PyLCFRSParser parser
    cdef Enumerator nonterminal_map

    def __init__(self, grammar, Enumerator nonterminal_map):
        """
        :param grammar:
        :type grammar: PyLCFRS
        """
        factory = PyLCFRSFactory(grammar.start(), nonterminal_map)
        factory.import_grammar(grammar)
        self.parser = factory.build_parser()

        cdef vector[NONTERMINAL] node_labels = range(0, nonterminal_map.counter)
        cdef vector[size_t] edge_labels = range(0, len(grammar.rule_index()))

        self.trace_manager = build_trace_manager_ptr[NONTERMINAL, size_t](
            make_shared[vector[NONTERMINAL]](node_labels)
            , make_shared[vector[size_t]](edge_labels)
            , False)

        self.nonterminal_map = nonterminal_map

    cpdef void compute_reducts(self, corpus, terminal_labelling):
        start_time = time.time()
        for i, tree in enumerate(corpus):
            word = [terminal_labelling.token_label(token) for token in tree.token_yield()]
            self.parser.do_parse(word)
            self.parser.prune_trace()
            add_trace_to_manager[NONTERMINAL, TERMINAL, size_t](deref(self.parser.parser)
                                 , self.trace_manager)

            # deref(self.parser.parser).print_trace()

            if i % 100 == 0:
                print(i, time.time() - start_time) #output_helper(str(i) + ' ' + str(time.time() - start_time))

    cpdef Enumerator get_nonterminal_map(self):
        return self.nonterminal_map


def compute_LCFRS_reducts(grammar, corpus, terminal_labelling, nonterminal_map=Enumerator()):
    #output_helper("creating trace")
    print("creating trace")
    trace = PyLCFRSTraceManager(grammar, nonterminal_map)
    # output_helper("computing reducts")
    print("computing reducts")
    trace.compute_reducts(corpus, terminal_labelling)
    return trace