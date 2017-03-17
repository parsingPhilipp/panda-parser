from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from util.enumerator cimport Enumerator

# this typedef seems necessary,
# since the compiler does not accept "vector[unsigned int]" or "vector[unsigned]"
# but accepts "vector[unsigned_int]"
ctypedef unsigned int unsigned_int

# this needs to be consistent
DEF ENCODE_NONTERMINALS = True
ctypedef size_t NONTERMINAL
DEF ENCODE_TERMINALS = True
ctypedef size_t TERMINAL


cdef extern from "DCP/SDCP.h" namespace "DCP":
    cdef cppclass Rule[Nonterminal, Terminal]:
         Rule(Nonterminal)
         void add_nonterminal(Nonterminal)
         void next_inside_attribute()
         void set_id(int)
         int get_id()
         void next_word_function_argument()
         void add_var_to_word_function(int,int)
         void add_terminal_to_word_function(Terminal)

    cdef cppclass STermBuilder[Nonterminal, Terminal]:
         void add_var(int, int)
         void add_terminal(Terminal, int)
         void add_terminal(Terminal)
         bint add_children()
         bint move_up()
         void clear()
         void add_to_rule(Rule*)

    cdef cppclass Variable:
         Variable(int, int)

    cdef cppclass SDCP[Nonterminal, Terminal]:
        SDCP()
        bint add_rule(Rule[Nonterminal, Terminal])
        bint set_initial(Nonterminal)
        void output()


cdef extern from "DCP/SDCP.h" namespace "boost":
    cdef cppclass variant

cdef extern from "DCP/HybridTree.h" namespace "DCP":
    cdef cppclass HybridTree[Terminal, Position]:
        void add_node(Position, Terminal, Position)
        void add_node(Position, Terminal, Terminal, Position)
        void add_child(Position, Position)
        void set_entry(Position)
        void set_exit(Position)
        void is_initial(Position)
        Terminal get_label(Position)
        Position get_next(Position)
        void output()
        void set_linearization(vector[Position])

    cdef void output_helper(string)

cdef extern from "DCP/SDCP_Parser.h" namespace "DCP":
    cdef cppclass SDCPParser[Nonterminal,Terminal,Position]:
        SDCPParser()
        SDCPParser(bint,bint,bint,bint)
        void do_parse()
        void clear()
        void set_input(HybridTree[Terminal,Position])
        HybridTree input;
        void set_sDCP(SDCP[Nonterminal, Terminal])
        void set_goal()
        void reachability_simplification()
        void print_chart()
        void print_trace()
        bint recognized()
        ParseItem* goal
        vector[pair[Rule,vector[ParseItem]]] query_trace(ParseItem)

    cdef cppclass ParseItem[Nonterminal,Position]:
        Nonterminal nonterminal
        vector[pair[Position,Position]] spans_inh
        vector[pair[Position,Position]] spans_syn

# cdef class Enumerator:
#     cdef size_t counter
#     cdef dict obj_to_ind
#     cdef dict ind_to_obj
#     cdef size_t first_index
#     cdef size_t object_index(self, obj)


cdef class PySDCPParser(object):
    cdef SDCP[NONTERMINAL,TERMINAL] sdcp
    cdef SDCPParser[NONTERMINAL,TERMINAL,int]* parser
    cdef Enumerator terminal_map, nonterminal_map
    cdef bint debug
    cdef void set_sdcp(self, SDCP[NONTERMINAL,TERMINAL] sdcp)
    cdef void set_terminal_map(self, Enumerator terminal_map)
    cdef void set_nonterminal_map(self, Enumerator nonterminal_map)
    cpdef void do_parse(self)
    cpdef bint recognized(self)
    cpdef void clear(self)
    cpdef void print_trace(self)

cdef SDCP[NONTERMINAL, TERMINAL] grammar_to_SDCP(grammar, nonterminal_encoder, terminal_encoder, lcfrs_conversion=?) except *