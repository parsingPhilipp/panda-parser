from parser.derivation_interface import AbstractDerivation
from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared
from grammar.lcfrs import LCFRS_rule
from itertools import product

cdef extern from "util.h":
    cdef void output_helper(string)

cdef class PyTraceManager:
    cpdef serialize(self, string path):
        serialize_trace(self.trace_manager, path)

    cpdef void load_traces_from_file(self, string path):
        cdef TraceManagerPtr[NONTERMINAL, size_t] tm = load_trace_manager[NONTERMINAL, size_t](path)
        self.trace_manager = tm

    cpdef Enumerator get_nonterminal_map(self):
        pass

    def enumerate_derivations(self, size_t traceId, grammar):
        cdef Trace[NONTERMINAL, size_t]* trace = &(deref(fool_cython_unwrap(self.trace_manager))[traceId])
        cdef PyElement goal = PyElement()
        # cdef size_t label = deref(deref(trace).get_goal().get()).get_label()
        if deref(trace).has_topological_order():
            # output_helper(str(label))
            goal.element = make_shared[Element[Node[NONTERMINAL]]](deref(trace).get_goal())
            for tree in self.__enumerate_derivations_rec(traceId, goal):
                yield TraceManagerDerivation(tree, grammar)

    def __enumerate_derivations_rec(self, size_t traceId, PyElement node):
        cdef Trace[NONTERMINAL, size_t]* trace = &(deref(fool_cython_unwrap(self.trace_manager))[traceId])
        cdef PyElement root
        cdef PyElement child
        cdef NONTERMINAL root_nonterminal
        cdef size_t rule_id
        cdef Element[HyperEdge[Node[NONTERMINAL], size_t]]* edge
        cdef vector[Element[HyperEdge[Node[NONTERMINAL], size_t]]] incoming_edges
        cdef size_t i
        cdef size_t c
        # for edge in deref(deref(trace).get_hypergraph()).get_incoming_hyperedges(deref(node.element)):
        incoming_edges = deref(deref(trace).get_hypergraph()).get_incoming_edges(deref(node.element))

        for i in range(incoming_edges.size()):
            edge = &(incoming_edges[i])
            childrens = []
            for c in range(deref(deref(edge).get()).get_sources().size()):
                child = PyElement()
                child.element = make_shared[Element[Node[NONTERMINAL]]](deref(deref(edge).get()).get_sources()[c])
                childrens.append([tree for tree in self.__enumerate_derivations_rec(traceId, child)])
            child_combinations = product(*childrens)
            for children in child_combinations:
                root = PyElement()
                root.element = make_shared[Element[Node[NONTERMINAL]]](deref(deref(edge).get()).get_target())
                root_nonterminal = deref(deref(deref(edge).get()).get_target().get()).get_label()
                rule_id = deref(deref(edge).get()).get_label()
                yield DerivationTree(root, root_nonterminal, rule_id, list(children))


cdef class PyElement:
    cdef shared_ptr[Element[Node[NONTERMINAL]]] element
    def __cinit__(self):
        self.element = shared_ptr[Element[Node[NONTERMINAL]]]()

    def __hash__(self):
        return deref(self.element).hash()

    def __richcmp__(self, other, select):
         if select == 2:
             return deref((<PyElement> self).element)\
                 .equals(deref((<PyElement> other).element))
         elif select == 3:
             return not deref((<PyElement> self).element)\
                 .equals(deref((<PyElement> other).element))

class TraceManagerDerivation(AbstractDerivation):
    # cdef Element[Node[NONTERMINAL]] root_idx
    # cdef vector[Element[Node[NONTERMINAL]]] __ids

    def getRule(self, idx):
        return self.__rules[idx]

    def ids(self):
        return self.__ids

    def position_relative_to_parent(self, idx):
        return self.__relative_positions[idx]

    def child_id(self, idx, i):
        return self.__child_ids[idx][i]

    def child_ids(self, idx):
        return self.__child_ids[idx]

    def root_id(self):
        return self.__root_idx

    def __init__(self, DerivationTree tree, grammar):
        self.__root_idx = tree.root_id
        self.__ids = []
        self.__rules = {}
        self.__child_ids = {}
        self.__relative_positions = {}
        self.spans = None
        self.__fill_recursive(tree, grammar)

    def __fill_recursive(self, DerivationTree tree, grammar):
        self.__ids.append(tree.root_id)
        self.__rules[tree.root_id] = grammar.rule_index(tree.rule_id)
        self.__child_ids[tree.root_id] = []
        cdef size_t i = 0
        cdef DerivationTree child_

        for child in tree.children:
            child_ = <DerivationTree> child
            self.__child_ids[tree.root_id].append(child_.root_id)
            self.__relative_positions[child_.root_id] = (tree.root_id, i)
            self.__fill_recursive(child_, grammar)
            i = i + 1


cdef class DerivationTree:
    cdef PyElement root_id
    cdef NONTERMINAL root_nonterminal
    cdef size_t rule_id
    cdef list children

    def __init__(self, PyElement root_id, NONTERMINAL root_nonterminal, size_t rule_id, list children):
        self.root_id = root_id
        self.root_nonterminal = root_nonterminal
        self.rule_id = rule_id
        self.children = children