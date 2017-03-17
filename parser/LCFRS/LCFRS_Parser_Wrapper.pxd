from libcpp.memory cimport shared_ptr, unique_ptr, make_unique
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from util.enumerator cimport Enumerator, unsigned_long

# Options:
DEF ENCODE_NONTERMINALS = True
DEF ENCODE_TERMINALS = True

cdef extern from "LCFR/LCFRS.h" namespace "LCFR":
    cdef cppclass LCFRS[Nonterminal, Terminal]:
        pass

cdef extern from "LCFR/LCFRS_Parser.h" namespace "LCFR":
    cdef cppclass HypergraphPtr[Nonterminal]:
        pass
    cdef cppclass LCFRS_Parser[Nonterminal, Terminal]:
        LCFRS_Parser(LCFRS grammar, vector[Terminal] word)
        void do_parse()
        void prune_trace()
        void print_trace()
        # HypergraphPtr[Nonterminal] convert_trace_to_hypergraph(
        #         shared_ptr[vector[Nonterminal]] nLabels
        #         , shared_ptr[vector[size_t]] eLabels)
        map[unsigned_long, pair[Nonterminal, vector[pair[unsigned_long, unsigned_long]]]] get_passive_items_map()
        map[unsigned_long, vector[pair[unsigned_long, vector[unsigned_long]]]] convert_trace()
        pair[Nonterminal, vector[pair[unsigned_long, unsigned_long]]] get_initial_passive_item()


cdef extern from "LCFR/LCFRS_util.h" namespace "LCFR":
    cdef cppclass LCFRSFactory[Nonterminal, Terminal]:
        LCFRSFactory(const Nonterminal initial)

        void new_rule(const Nonterminal nont)

        void add_terminal(const Terminal term)

        void add_variable(const unsigned long index, const unsigned long arg)

        void complete_argument()

        void add_rule_to_grammar(vector[Nonterminal] rhs, const unsigned long id)

        shared_ptr[LCFRS[Nonterminal, Terminal]] get_grammar()

IF ENCODE_NONTERMINALS:
    ctypedef unsigned_long NONTERMINAL
ELSE:
    ctypedef string NONTERMINAL

IF ENCODE_TERMINALS:
    ctypedef unsigned_long TERMINAL
ELSE:
    ctypedef string TERMINAL


cdef class PyLCFRSParser:
    cdef unique_ptr[LCFRS_Parser[NONTERMINAL,TERMINAL]] parser
    cdef shared_ptr[LCFRS[NONTERMINAL, TERMINAL]] grammar
    cdef Enumerator tMap
    cdef set_grammar(self, shared_ptr[LCFRS[NONTERMINAL, TERMINAL]] grammar)
    cpdef void do_parse(self, word)
    cpdef void prune_trace(self)
    cpdef map[unsigned_long, pair[NONTERMINAL, vector[pair[unsigned_long, unsigned_long]]]] get_passive_items_map(self)
    cpdef map[unsigned_long, vector[pair[unsigned_long, vector[unsigned_long]]]] convert_trace(self)

cdef class PyLCFRSFactory:
    cdef unique_ptr[LCFRSFactory[NONTERMINAL,TERMINAL]] _thisptr
    IF ENCODE_NONTERMINALS:
        cdef Enumerator ntMap
    cdef Enumerator tMap
    cpdef void new_rule(self, NONTERMINAL lhsNont)
    cpdef void add_terminal(self, TERMINAL term)
    cpdef void add_variable(self, unsigned long index, unsigned long arg)
    cpdef void complete_argument(self)
    cpdef void add_rule_to_grammar(self, vector[NONTERMINAL] rhs, const unsigned long ruleId)
    cpdef PyLCFRSParser build_parser(self)