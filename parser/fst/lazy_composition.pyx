from libcpp.vector cimport vector
from libcpp.string cimport string
import pynini

cdef extern from "lazy_composition.h":
    cdef cppclass Fst:
        pass
    cdef cppclass StdFst:
        pass
    cdef cppclass StdVectorFst:
        pass
    StdFst* readFst(string)
    cdef void lazy_compose(StdVectorFst & fst_a, StdFst & fst_b)
    cdef vector[unsigned] lazy_compose_(const StdVectorFst & fst_a, const StdFst & fst_b)
    cdef StdVectorFst construct_fa(vector[string] sequence, StdFst & fst)


def encode_list_of_symbols(input, symbol_table):
    """
    :param input:
    :type input:
    :param symbol_table:
    :type symbol_table: SymbolTable
    :return: An acceptor for the given list of tokens.
    :rtype: Fst
    The symbol table gets extended, if new tokens occur in the input.
    """
    output = []
    for x in input:
        try:
            output.append(symbol_table.find(x))
        except KeyError:
            output.append(symbol_table.add_symbol(x))
    return output


cdef class DelayedFstComposer:
    cdef object __fst
    cdef StdFst* __cpp_fst
    def __init__(self, fst):
        assert(isinstance(fst, pynini.Fst))
        fst.write('/tmp/main_fst.fst')
        self.__fst = fst
        self.__cpp_fst = readFst('/tmp/main_fst.fst')

    def compose(self, input):
        # input_encoded = encode_list_of_symbols(input, self.__fst.mutable_input_symbols())
        cdef StdVectorFst input_fsa = construct_fa(input, self.__cpp_fst[0])
        lazy_compose(input_fsa, self.__cpp_fst[0])
        return pynini.Fst.read("/tmp/shortest_path.fst")

    def compose_(self, input):
        cdef StdVectorFst input_fsa = construct_fa(input, self.__cpp_fst[0])
        return lazy_compose_(input_fsa, self.__cpp_fst[0])
