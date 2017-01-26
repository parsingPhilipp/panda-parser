
cimport cython

from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference as deref


cdef extern from "LCFRS.h" namespace "LCFR":
    ctypedef unsigned Nonterminal
    ctypedef unsigned Terminal

cdef extern from "boost/variant/variant.hpp" namespace "boost":
    cdef cppclass variant[Terminal, Variable]:
        pass


cdef extern from "LCFRS.h" namespace "LCFR":
    ctypedef pair[unsigned long, unsigned long] Range

    cdef cppclass Variable:
        Variable(unsigned long ind, unsigned long arg)
        unsigned long get_index()
        unsigned long get_arg()

    # ctypedef variant[Terminal,Variable] TerminalOrVariable[Terminal]
    # ctypedef  unsigned long TerminalOrVariable

    cdef cppclass LHS[Nonterminal,Terminal]:
        LHS(Nonterminal nonterminal)
        Nonterminal get_nont()
        # vector[vector[TerminalOrVariable[Terminal]]] get_args()
        # void add_argument(vector[TerminalOrVariable[Terminal]] argument)

    cdef cppclass Rule[Nonterminal, Terminal]:
        Rule(LHS[Nonterminal, Terminal] lhs, vector[Nonterminal] rhs)
        LHS[Nonterminal,Terminal] get_lhs()
        vector[Nonterminal] get_rhs()

    cdef cppclass LCFRS[Nonterminal, Terminal]:
        LCFRS(Nonterminal init_nont, string name)
        Nonterminal get_initial_nont()
        map[Nonterminal, vector[shared_ptr[Rule[Nonterminal, Terminal]]]] get_rules()
        void add_rule(Rule[Nonterminal,Terminal] rule)



cdef extern from "LCFRS_Parser.h" namespace "LCFR":
    cdef cppclass PassiveItem[Nonterminal]:
        PassiveItem(Nonterminal nont, vector[Range] rs)
        Nonterminal get_nont()
        vector[Range] get_ranges()

    cdef cppclass TraceItem[Nonterminal, Terminal]:
        shared_ptr[PassiveItem[Nonterminal]] uniquePtr
        vector[pair[shared_ptr[Rule[Nonterminal,Terminal]]
               , vector[shared_ptr[PassiveItem[Nonterminal]]]
               ]
            ] parses

    cdef cppclass LCFRS_Parser:
        LCFRS_Parser(LCFRS grammar, vector[Terminal] word)
        void do_parse()
        map[PassiveItem,TraceItem] get_trace()


cdef extern from "LCFRS_util.h" namespace "LCFR":
    cdef cppclass RuleFactory[Nonterminal, Terminal]:
        void new_rule(const Nonterminal nont)

        void add_terminal(const Terminal term)

        void add_variable(const unsigned long index, const unsigned long arg)

        void complete_argument()

        Rule[Nonterminal,Terminal] get_rule(vector[Nonterminal] rhs)

cdef class PyRuleFactory:
    cdef RuleFactory *_thisptr
    def __cinit__(self):
        self._thisptr = new RuleFactory()

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    cpdef void new_rule(self, Nonterminal lhsNont):
        return self._thisptr.new_rule(lhsNont)

    cpdef void add_terminal(self, Terminal term):
        return self._thisptr.add_terminal(term)

    cpdef add_variable(self, unsigned long index, unsigned long arg):
        return self._thisptr.add_variable(index,arg)

    cpdef void complete_argument(self):
        return self._thisptr.complete_argument()

    cpdef Rule[Nonterminal, Terminal] get_rule(self, vector[Nonterminal] rhs):
        return self._thisptr.get_rule(rhs)


def test():
    print "test"