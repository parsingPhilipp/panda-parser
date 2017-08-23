from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from util.enumerator cimport Enumerator

cdef extern from "Manage/Manager.h":
    cppclass Element[InfoT]:
        InfoT* get "operator->"()
        bint equals "operator=="(const Element[InfoT]& other)
        size_t hash()

# cdef extern from "Manage/Mangaer.h" namespace "std":
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
        vector[Element[NodeT]] get_sources()

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

cdef extern from "Trainer/TraceManager.h" namespace "Trainer":
    cdef cppclass Trace[Nonterminal, oID]:
        const Element[Node[Nonterminal]]& get_goal()
        bint has_topological_order()
        shared_ptr[Hypergraph[Nonterminal, size_t]] get_hypergraph()
    cdef cppclass TraceManager2[Nonterminal, TraceID]:
        Trace[Nonterminal, TraceID] operator[](size_t)
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