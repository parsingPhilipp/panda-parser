from libcpp.vector cimport vector
from libcpp.memory cimport make_shared
from util.enumerator cimport Enumerator
from parser.commons.commons cimport *
from parser.trace_manager.trace_manager cimport PyTraceManager, build_trace_manager_ptr, TraceManagerPtr
from sdcp_parser_wrapper cimport PySDCPParser, grammar_to_SDCP, SDCPParser
import time


# this needs to be consistent with parser.commons.commons
DEF ENCODE_NONTERMINALS = True
DEF ENCODE_TERMINALS = True


cdef extern from "DCP/util.h" namespace "DCP":
    cdef void add_trace_to_manager[Nonterminal, Terminal, Position, TraceID]\
        (SDCPParser[Nonterminal, Terminal, Position]
         , TraceManagerPtr[Nonterminal, TraceID])

cdef extern from "util.h":
    cdef void output_helper(string)

cdef class PySDCPTraceManager(PyTraceManager):
    cdef PySDCPParser parser

    def __init__(self, grammar, term_labelling, lcfrs_parsing=True, debug=False):
        """
        :param grammar:
        :type grammar: gl.LCFRS
        :param lcfrs_parsing:
        :type lcfrs_parsing:
        :param debug:
        :type debug:
        """
        output_helper("initializing PyTraceManager")

        cdef Enumerator nonterminal_map = Enumerator()
        cdef Enumerator terminal_map = Enumerator()
        nonterminal_encoder = (lambda s: nonterminal_map.object_index(s)) if ENCODE_NONTERMINALS else lambda s: str(s)
        terminal_encoder = (lambda s: terminal_map.object_index(s)) if ENCODE_TERMINALS else lambda s: str(s)

        self.parser = PySDCPParser(grammar, term_labelling, lcfrs_parsing, debug)
        self.parser.set_sdcp(grammar_to_SDCP(grammar, nonterminal_encoder, terminal_encoder, lcfrs_parsing))
        self.parser.set_terminal_map(terminal_map)
        self.parser.set_nonterminal_map(nonterminal_map)

        cdef vector[NONTERMINAL] node_labels = range(0, self.parser.nonterminal_map.counter)
        cdef vector[size_t] edge_labels = range(0, len(grammar.rule_index()))

        self.trace_manager = build_trace_manager_ptr[NONTERMINAL, size_t](
            make_shared[vector[NONTERMINAL]](node_labels)
            , make_shared[vector[size_t]](edge_labels)
            , False)

    def compute_reducts(self, corpus):
        start_time = time.time()
        for i, tree in enumerate(corpus):
            self.parser.clear()
            self.parser.set_input(tree)
            self.parser.do_parse()
            if self.parser.recognized():
                add_trace_to_manager[NONTERMINAL,TERMINAL,int,size_t](self.parser.parser[0],self.trace_manager)
                # self.parser.print_trace()

            if i % 100 == 0:
                output_helper(str(i) + ' ' + str(time.time() - start_time))

    cpdef Enumerator get_nonterminal_map(self):
        return self.parser.nonterminal_map

def compute_reducts(grammar, corpus, term_labelling, debug=False):
    output_helper("creating trace")
    trace = PySDCPTraceManager(grammar, term_labelling, debug=debug)
    output_helper("computing reducts")
    trace.compute_reducts(corpus)
    return trace
