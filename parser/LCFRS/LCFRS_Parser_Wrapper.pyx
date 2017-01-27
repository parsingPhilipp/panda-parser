
cimport cython

from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference as deref

from hybridtree.monadic_tokens import CoNLLToken



ctypedef unsigned long unsigned_long

cdef extern from "LCFRS_util.h" namespace "LCFR":
    cdef cppclass LCFRSFactory[Nonterminal, Terminal]:
        LCFRSFactory(const Nonterminal initial)

        void new_rule(const Nonterminal nont)

        void add_terminal(const Terminal term)

        void add_variable(const unsigned long index, const unsigned long arg)

        void complete_argument()

        void add_rule_to_grammar(vector[Nonterminal] rhs)

        void do_parse(vector[Terminal] word)

        map[unsigned_long, pair[Nonterminal, vector[pair[unsigned_long, unsigned_long]]]] get_passive_items_map()

        map[unsigned_long, vector[pair[unsigned_long, vector[unsigned_long]]]] convert_trace()



# Python classes:
ctypedef string Nonterminal
ctypedef string Terminal




cdef class PyLCFRSFactory:
    cdef LCFRSFactory[Nonterminal,Terminal] *_thisptr
    # cdef LCFRS *_grammar
    # cdef LCFRSParser *_parser
    # cdef map[PassiveItem[Nonterminal, Terminal], TraceItem[Nonterminal, Terminal]] trace

    def __cinit__(self, Nonterminal initial_nont):
        self._thisptr = new LCFRSFactory[Nonterminal,Terminal](initial_nont)
        # self._grammar = new LCFRS(initial_nont)

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr
        # if self._grammar != NULL:
        #     del self._grammar
        # if self._parser != NULL:
        #     del self._parser

    cpdef void new_rule(self, Nonterminal lhsNont):
        self._thisptr.new_rule(lhsNont)

    cpdef void add_terminal(self, Terminal term):
        self._thisptr.add_terminal(term)

    cpdef void add_variable(self, unsigned long index, unsigned long arg):
        self._thisptr.add_variable(index,arg)

    cpdef void complete_argument(self):
        self._thisptr.complete_argument()

    cpdef void add_rule_to_grammar(self, vector[Nonterminal] rhs):
        self._thisptr.add_rule_to_grammar(rhs)

    cpdef void do_parse(self, vector[Terminal] word):
        self._thisptr.do_parse(word);

    cpdef map[unsigned_long, pair[Nonterminal, vector[pair[unsigned_long, unsigned_long]]]] get_passive_items_map(self):
        return self._thisptr.get_passive_items_map()

    cpdef map[unsigned_long, vector[pair[unsigned_long, vector[unsigned_long]]]] convert_trace(self):
        return self._thisptr.convert_trace()






# cdef class TraceManager:
#     cdef map[PassiveItem[Nonterminal,Terminal],TraceItem[Nonterminal,Terminal]] *_trace
#     cdef __cinit__(self, trace):
#         self._trace = trace
#     cdef __dealloc__(self):
#         if _trace != NULL:
#             del _trace





# cdef class PyLCFRSParser:
#     cdef LCFRSParser *_thisptr
#     cdef __cinit__(self, grammar, word):
#         self._thisptr = new LCFRS(grammar, word)
#
#     cdef __dealloc__(self):
#         if self._thisptr != NULL:
#              del self._thisptr
#
#     cpdef do_parse(self):
#         return self._thisptr.do_parse()
#
#     cdef get_trace(self):
#         return TraceManager(self._thisptr.get_trace())
