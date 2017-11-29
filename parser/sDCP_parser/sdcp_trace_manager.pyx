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
         , TraceManagerPtr[Nonterminal, TraceID],
           double frequency)

cdef extern from "util.h":
    cdef void output_helper(string)

cdef class PySDCPTraceManager(PyTraceManager):
    cdef PySDCPParser parser

    def __init__(self, grammar, term_labelling, PySDCPParser parser=None, Enumerator nont_map=None, lcfrs_parsing=True, debug=False):
        """
        :param grammar:
        :type grammar: gl.LCFRS
        :param lcfrs_parsing:
        :type lcfrs_parsing:
        :param debug:
        :type debug:
        """
        output_helper("initializing PyTraceManager")

        cdef Enumerator nonterminal_map = nont_map if nont_map is not None else Enumerator()
        cdef Enumerator terminal_map = Enumerator()

        if parser is None:
            nonterminal_encoder = (lambda s: nonterminal_map.object_index(s)) if ENCODE_NONTERMINALS else lambda s: str(s)
            terminal_encoder = (lambda s: terminal_map.object_index(s)) if ENCODE_TERMINALS else lambda s: str(s)

            self.parser = PySDCPParser(grammar, term_labelling, lcfrs_parsing, debug)
            self.parser.set_sdcp(grammar_to_SDCP(grammar, nonterminal_encoder, terminal_encoder, lcfrs_parsing))
            self.parser.set_terminal_map(terminal_map)
            self.parser.set_nonterminal_map(nonterminal_map)
        else:
            self.parser = parser

        cdef vector[NONTERMINAL] node_labels = range(0, self.parser.nonterminal_map.counter)
        cdef vector[size_t] edge_labels = range(0, len(grammar.rule_index()))

        self.trace_manager = build_trace_manager_ptr[NONTERMINAL, size_t](
            make_shared[vector[NONTERMINAL]](node_labels)
            , make_shared[vector[size_t]](edge_labels)
            , False)

    def compute_reducts(self, corpus, frequency=1.0):
        start_time = time.time()
        cdef int successful = 0
        cdef int fails = 0
        for i, tree in enumerate(corpus):
            self.parser.clear()
            self.parser.set_input(tree)
            self.parser.do_parse()
            if self.parser.recognized():
                add_trace_to_manager[NONTERMINAL,TERMINAL,int,size_t](self.parser.parser[0], self.trace_manager,
                                                                      frequency)
                successful += 1
                # self.parser.print_trace()
            else:
                fails += 1
                output_helper(str(i) + " " + str(tree) + str(tree.token_yield()) + " " + str(tree.full_yield()) )

            if (i + 1) % 100 == 0:
                output_helper(str(i + 1) + ' ' + str(time.time() - start_time))
        output_helper("Computed reducts for " + str(successful) + " out of " + str(successful + fails))

    cpdef Enumerator get_nonterminal_map(self):
        return self.parser.nonterminal_map

    cpdef PySDCPParser get_parser(self):
        return self.parser

def compute_reducts(grammar, corpus, term_labelling, PySDCPParser parser=None, Enumerator nont_map=None, debug=False,
                    frequency=1.0):
    output_helper("creating trace")
    trace = PySDCPTraceManager(grammar, term_labelling, parser=parser, nont_map=nont_map, debug=debug)
    output_helper("computing reducts")
    trace.compute_reducts(corpus, frequency=frequency)
    return trace
