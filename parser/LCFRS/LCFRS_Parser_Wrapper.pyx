
cimport cython

from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference as deref

from hybridtree.monadic_tokens import CoNLLToken
from grammar.lcfrs import LCFRS as PyLCFRS, LCFRS_var as PyLCFRS_var



# Options:
DEF ENCODE_NONTERMINALS = True
DEF ENCODE_TERMINALS = True


ctypedef unsigned long unsigned_long

IF ENCODE_NONTERMINALS:
    ctypedef unsigned_long Nonterminal
ELSE:
    ctypedef string Nonterminal

IF ENCODE_TERMINALS:
    ctypedef unsigned_long Terminal
ELSE:
    ctypedef string Terminal






cdef extern from "LCFRS.h" namespace "LCFR":
    pass

cdef extern from "LCFRS_Parser.h" namespace "LCFR":
    pass

cdef extern from "LCFRS_util.h" namespace "LCFR":
    cdef cppclass LCFRSFactory[Nonterminal, Terminal]:
        LCFRSFactory(const Nonterminal initial)

        void new_rule(const Nonterminal nont)

        void add_terminal(const Terminal term)

        void add_variable(const unsigned long index, const unsigned long arg)

        void complete_argument()

        void add_rule_to_grammar(vector[Nonterminal] rhs, const unsigned long id)

        void do_parse(vector[Terminal] word)

        map[unsigned_long, pair[Nonterminal, vector[pair[unsigned_long, unsigned_long]]]] get_passive_items_map()

        map[unsigned_long, vector[pair[unsigned_long, vector[unsigned_long]]]] convert_trace()

        Nonterminal get_initial_nont()

        vector[Terminal] get_word()

        pair[Nonterminal, vector[pair[unsigned_long, unsigned_long]]] get_initial_passive_item()




cdef class Enumerator:
    cdef unsigned_long counter
    cdef dict obj_to_ind
    cdef dict ind_to_obj
    cdef unsigned_long first_index

    def __init__(self, first_index=0):
        self.first_index = first_index
        self.counter = first_index
        self.obj_to_ind = {}
        self.ind_to_obj = {}

    def index_object(self, int i):
        """
        :type i: int
        :return:
        """
        return self.ind_to_obj[i]

    cdef unsigned_long object_index(self, obj):
        if obj in self.obj_to_ind:
            return self.obj_to_ind[obj]
        else:
            self.obj_to_ind[obj] = self.counter
            self.ind_to_obj[self.counter] = obj
            self.counter += 1
            return self.counter - 1

    cdef objects_indices(self, objects):
        result = vector[unsigned_long]();
        for obj in objects:
            result += [self.object_index(obj)]
        return result

cdef class PyLCFRSFactory:
    cdef LCFRSFactory[Nonterminal,Terminal] *_thisptr
    cdef Enumerator ruleMap
    IF ENCODE_NONTERMINALS:
        cdef Enumerator ntMap
    IF ENCODE_TERMINALS:
        cdef Enumerator tMap

    def __cinit__(self, initial_nont):
        IF ENCODE_NONTERMINALS:
            self.ntMap = Enumerator()
            self._thisptr = new LCFRSFactory[Nonterminal,Terminal](self.ntMap.object_index(initial_nont))
        ELSE:
            self._thisptr = new LCFRSFactory[Nonterminal,Terminal](initial_nont)

        self.ruleMap = Enumerator()

        IF ENCODE_TERMINALS:
            self.tMap = Enumerator()


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

    cpdef void add_rule_to_grammar(self, vector[Nonterminal] rhs, const unsigned long ruleId):
        self._thisptr.add_rule_to_grammar(rhs, ruleId)

    cpdef void do_parse(self, word):
        IF ENCODE_TERMINALS:
            self._thisptr.do_parse(self.tMap.objects_indices(word));
        ELSE:
            self._thisptr.do_parse(word);

    cpdef map[unsigned_long, pair[Nonterminal, vector[pair[unsigned_long, unsigned_long]]]] get_passive_items_map(self):
        return self._thisptr.get_passive_items_map()

    cpdef map[unsigned_long, vector[pair[unsigned_long, vector[unsigned_long]]]] convert_trace(self):
        return self._thisptr.convert_trace()

    def import_grammar(self, grammar):
        # :type grammar PyLCFRS
        # :return:

        for rule in grammar.rules():
            IF ENCODE_NONTERMINALS:
                self.new_rule(self.ntMap.object_index(rule.lhs().nont()))
            ELSE:
                self.new_rule(rule.lhs().nont())
            for argument in rule.lhs().args():
                for symbol in argument:
                    if type(symbol) is PyLCFRS_var:
                        self.add_variable(symbol.mem, symbol.arg)
                    else:
                        IF ENCODE_NONTERMINALS:
                            self.add_terminal(self.tMap.object_index(symbol))
                        ELSE:
                            self.add_terminal(symbol)
                self.complete_argument()
            IF ENCODE_NONTERMINALS:
                self.add_rule_to_grammar(self.ntMap.objects_indices(rule.rhs()), self.ruleMap.object_index(rule))
            ELSE:
                self.add_rule_to_grammar(rule.rhs(), self.ruleMap.object_index(rule))

    def get_initial_passive_item(self):
        return self._thisptr.get_initial_passive_item()




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
