from libcpp.string cimport string
# this typedef seems necessary,
# since the compiler does not accept "vector[unsigned int]" or "vector[unsigned]"
# but accepts "vector[unsigned_int]"
ctypedef unsigned int unsigned_int

DEF ENCODE_NONTERMINALS = True
DEF ENCODE_TERMINALS = True

IF ENCODE_NONTERMINALS:
    ctypedef size_t NONTERMINAL
ELSE:
    ctypedef string NONTERMINAL

IF ENCODE_TERMINALS:
    ctypedef size_t TERMINAL
ELSE:
    ctypedef string TERMINAL

cdef extern from "util.h":
    cdef void output_helper(string)

cpdef void output_helper_utf8(str s)