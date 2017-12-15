from __future__ import print_function
from parser.derivation_interface import AbstractDerivation
from cython.operator cimport dereference as deref
from libcpp.memory cimport make_shared
from libcpp.map cimport map as cmap
from grammar.lcfrs import LCFRS_rule, LCFRS
from itertools import product

cdef extern from "util.h":
    cdef void output_helper(string)


cpdef double prod(double x, double y):
    return x * y


cpdef double add(double x, double y):
    return x + y


cdef class PyTraceManager:
    cpdef serialize(self, string path):
        serialize_trace(self.trace_manager, path)

    cpdef void load_traces_from_file(self, string path):
        cdef TraceManagerPtr[NONTERMINAL, size_t] tm = load_trace_manager[NONTERMINAL, size_t](path)
        self.trace_manager = tm

    cpdef Enumerator get_nonterminal_map(self):
        raise NotImplementedError()

    def viterbi_derivation(self, size_t traceId, vector[double] edge_weights, grammar, op=prod):
        """
        :param traceId: trace of which the viterbi derivation shall be computed
        :type traceId: size_t
        :param edge_weights: non-negative weights for each edge (ordered according to edge ordering in hypergraph)
        :type edge_weights: list[double]
        :param grammar:
        :type grammar: LCFRS
        :return: the Viterbi derivation
        :param op: path operation (binary operation on double)
        :rtype: AbstractDerivation
        Uses Knuth's generalization of Dijkstra's algorithm to compute the best derivation of some hypergraph.
        cf. https://doi.org/10.1016/0020-0190(77)90002-3
        """
        cdef Trace[NONTERMINAL, size_t]* trace = &(deref(fool_cython_unwrap(self.trace_manager))[traceId])

        cdef dict node_best_weight = {}
        cdef node_best_edge = {}
        cdef cmap[void*, size_t] edge_weights_dict
        cdef edge_idx
        cdef HyperEdge[Node[NONTERMINAL], size_t]* edge

        cdef shared_ptr[Manager[HyperEdge[Node[NONTERMINAL], size_t]]] edges
        edges = deref(deref(trace).get_hypergraph()).get_edges().lock()

        cdef NONTERMINAL target
        cdef const Element[Node[NONTERMINAL]]* node
        cdef size_t source_list_idx

        cdef PyElement target_element
        cdef size_t sources_size

        cdef set U = set()
        cdef set Q = set()

        for edge_idx in range(deref(edges).size()):
            edge = &deref(edges)[edge_idx]
            edge_weights_dict[edge] = edge_idx

            target_element = PyElement()
            target_element.element = make_shared[Element[Node[NONTERMINAL]]](deref(edge).get_target())

            U.add(target_element)

            sources_size = deref(edge).get_sources().size()
            # print("iterated over edge", edge_idx, sources_size)
            if sources_size == 0:
                if target_element not in node_best_weight\
                        or node_best_weight[target_element] < edge_weights[edge_idx]:
                    node_best_weight[target_element] = edge_weights[edge_idx]
                    node_best_edge[target_element] = edge_idx
            else:
                if target_element not in node_best_weight:
                    node_best_weight[target_element] = 0.0
                    node_best_edge[target_element] = None

        cdef PyElement A
        cdef size_t position
        cdef PyElement source
        cdef bint all_sources_in_Q
        cdef double weight

        while U:
            A = max(U, key=lambda x: node_best_weight[x])
            U.remove(A)
            Q.add(A)
            for edge_idx in range(deref(deref(trace).get_hypergraph()).get_outgoing_edges(deref(A.element)).size()):
                edge = deref(deref(trace).get_hypergraph()).get_outgoing_edges(deref(A.element))[edge_idx].first.get()
                position = deref(deref(trace).get_hypergraph()).get_outgoing_edges(deref(A.element))[edge_idx].second

                weight = edge_weights[edge_weights_dict[edge]]
                all_sources_in_Q = True

                for source_list_idx in range(deref(edge).get_sources().size()):
                    source = PyElement()
                    source.element = make_shared[Element[Node[NONTERMINAL]]](deref(edge).get_sources()[source_list_idx])
                    if source not in Q:
                        all_sources_in_Q = False
                        break
                    weight = op(weight, node_best_weight[source])

                if all_sources_in_Q:
                    target_element = PyElement()
                    target_element.element = make_shared[Element[Node[NONTERMINAL]]](deref(edge).get_target())

                    if weight >= node_best_weight[target_element]:
                        node_best_weight[target_element] = weight
                        node_best_edge[target_element] = edge_weights_dict[edge]

        # print("best weights and incoming edges")
        # for i, target_element in enumerate(sorted(node_best_weight)):
        #     print(i, node_best_weight[target_element], node_best_edge[target_element])

        cdef PyElement root = PyElement()
        root.element = make_shared[Element[Node[NONTERMINAL]]](deref(trace).get_goal())
        cdef DerivationTree tree = self.__build_viterbi_derivation_tree_rec(root, node_best_edge, edges)
        return TraceManagerDerivation(tree, grammar)

    cdef DerivationTree __build_viterbi_derivation_tree_rec(self, PyElement node,
                                                            dict node_best_edge,
                                                            shared_ptr[Manager[HyperEdge[Node[NONTERMINAL], size_t]]] edges):
        cdef size_t best_edge_idx = node_best_edge[node]
        cdef HyperEdge[Node[NONTERMINAL], size_t]* edge = &(deref(edges)[best_edge_idx])
        cdef NONTERMINAL root_nonterminal = deref(deref(edge).get_target().get()).get_label()
        cdef list children = []
        cdef size_t child_list_idx
        for child_list_idx in range(deref(edge).get_sources().size()):
            source = PyElement()
            source.element = make_shared[Element[Node[NONTERMINAL]]](deref(edge).get_sources()[child_list_idx])
            children.append(self.__build_viterbi_derivation_tree_rec(source, node_best_edge, edges))
        cdef size_t rule_id = deref(edge).get_label()

        return DerivationTree(node, root_nonterminal, rule_id, children)

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
    def __init__(self, PyElement root_id, NONTERMINAL root_nonterminal, size_t rule_id, list children):
        self.root_id = root_id
        self.root_nonterminal = root_nonterminal
        self.rule_id = rule_id
        self.children = children