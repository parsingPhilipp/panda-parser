from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, weak_ptr
from libcpp.string cimport string
from util.enumerator cimport Enumerator
from libcpp.pair cimport pair
from libcpp.map cimport map as cmap
from libcpp.unordered_map cimport unordered_map
from parser.trace_manager.sm_trainer_util cimport PyGrammarInfo, GrammarInfo2
from parser.trace_manager.sm_trainer cimport LatentAnnotation

cdef extern from "Manage/Manager.h":
    cppclass Element[InfoT]:
        InfoT* get "operator->"()
        bint equals "operator=="(const Element[InfoT]& other)
        size_t hash()

cdef extern from "Manage/Manager.h" namespace "Manage":
    cppclass Manager[InfoT]:
        const InfoT operator[](size_t) const
        unsigned long size() const
#     cppclass hasher "hash"[InfoT, isConst]:
#         size_t hash_it "operator()"(const Element[InfoT]& element)

cdef extern from "Manage/Hypergraph.h" namespace "Manage":
    cppclass Node[NodeLabelT]:
        size_t get_label_id()
        NodeLabelT get_label()

    cppclass HyperEdge[NodeT, LabelT]:
        size_t get_label_id()
        LabelT get_label()
        Element[NodeT] get_target()
        const vector[Element[NodeT]] get_sources() const

    cppclass Hypergraph[NodeLabelT, EdgeLabelT]:
        Hypergraph(shared_ptr[vector[NodeLabelT]] nLabels
                   , shared_ptr[vector[EdgeLabelT]] eLabels)
        Element[Node[NodeLabelT]] create(NodeLabelT nLabel)
        Element[HyperEdge[Node[NodeLabelT], EdgeLabelT]] add_hyperedge(
                EdgeLabelT eLabel
                , Element[Node[NodeLabelT]]& target
                , vector[Element[Node[NodeLabelT]]]& sources
                )
        vector[Element[HyperEdge[Node[NodeLabelT], EdgeLabelT]]] get_incoming_edges(Element[Node[NodeLabelT]] e)
        const vector[pair[Element[HyperEdge[Node[NodeLabelT], EdgeLabelT]], size_t]]& get_outgoing_edges(Element[Node[NodeLabelT]])
        const weak_ptr[Manager[HyperEdge[Node[NodeLabelT], EdgeLabelT]]] get_edges()


cdef extern from "Trainer/TraceManager.h" namespace "Trainer":
    cdef cppclass Trace[Nonterminal, oID]:
        const Element[Node[Nonterminal]]& get_goal()
        bint has_topological_order()
        shared_ptr[Hypergraph[Nonterminal, size_t]] get_hypergraph()
        double get_frequency()
        bint is_consistent_with_grammar(GrammarInfo2 & grammarInfo)
        pair[size_t, unordered_map[pair[Element[Node[Nonterminal]], size_t], pair[Element[HyperEdge[Node[Nonterminal], size_t]], vector[size_t]]]] computeViterbiPath(LatentAnnotation & latentAnnotation)

    cdef cppclass TraceManager2[Nonterminal, TraceID]:
        Trace[Nonterminal, TraceID] operator[](size_t)
        void set_io_cycle_limit(unsigned int io_cycle_limit)
        void set_io_precision(double io_precision)
    cdef cppclass TraceManagerPtr[Nonterminal, TraceID]:
         pass
    cdef TraceManagerPtr[Nonterminal, TraceID] build_trace_manager_ptr[Nonterminal, TraceID](
            shared_ptr[vector[Nonterminal]]
            , shared_ptr[vector[size_t]]
            , bint)
    cdef void serialize_trace[Nonterminal, TraceID](TraceManagerPtr[Nonterminal, TraceID] traceManager, string path)
    cdef TraceManagerPtr[Nonterminal, TraceID] load_trace_manager[Nonterminal, TraceID](string path)

    cdef shared_ptr[TraceManager2[Nonterminal, TraceID]] fool_cython_unwrap[Nonterminal, TraceID](TraceManagerPtr[Nonterminal, TraceID] tmp)

ctypedef size_t NONTERMINAL

cdef class PyTraceManager:
    cdef TraceManagerPtr[NONTERMINAL, size_t] trace_manager
    cpdef serialize(self, string path)
    cpdef void load_traces_from_file(self, string path)
    cpdef Enumerator get_nonterminal_map(self)
    cpdef is_consistent_with_grammar(self, PyGrammarInfo grammarInfo, size_t traceId=*)
    cdef DerivationTree __build_viterbi_derivation_tree_rec(self,
                                                            Element[Node[NONTERMINAL]] node, # dict node_best_edge,
                                                            cmap[Element[Node[NONTERMINAL]], size_t] node_best_edge,
                                                            shared_ptr[Manager[HyperEdge[Node[NONTERMINAL], size_t]]] edges)
    cdef DerivationTree __build_viterbi_derivation_tree_rec_(
            self
            , Element[Node[NONTERMINAL]] node
            , size_t sub
            , unordered_map[  pair[Element[Node[NONTERMINAL]], size_t]
                            , pair[Element[HyperEdge[Node[NONTERMINAL], size_t]], vector[size_t]]]
                node_best_edge
        )
    cpdef void set_io_cycle_limit(self, unsigned int io_cycle_limit)
    cpdef void set_io_precision(self, double io_precision)


cdef class PyElement:
    cdef shared_ptr[Element[Node[NONTERMINAL]]] element

cdef class DerivationTree:
    cdef size_t rule_id
    cdef list children