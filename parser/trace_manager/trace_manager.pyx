from __future__ import print_function
from parser.derivation_interface import AbstractDerivation
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.memory cimport make_shared
from libcpp.map cimport map as cmap
from libcpp.set cimport set as cset
from grammar.lcfrs import LCFRS_rule, LCFRS
from itertools import product
from parser.trace_manager.sm_trainer cimport PyLatentAnnotation
from libc.math cimport log, NAN, INFINITY, isnan, isinf
from libcpp cimport bool
from grammar.rtg import RTG_like

cdef extern from "cpp_priority_queue.hpp":
    cdef cppclass intp_std_func_priority_queue[T]:
        intp_std_func_priority_queue(...)
        void push(Element[T])
        Element[T] top()
        void pop()
        bool empty()

    cdef cppclass comperator_function[T]:
        pass

    cdef comperator_function[T] construct_comparator[T](...)

cpdef double prod(double x, double y):
    return x * y


cpdef double add(double x, double y):
    return x + y


cdef bint in_vector(Element[Node[NONTERMINAL]] t, vector[Element[Node[NONTERMINAL]]] vec):
    cdef size_t idx = 0
    while idx < vec.size():
        if t.equals(vec[idx]):
            return True
        else:
            idx = idx + 1
    return False


cdef class PyTraceManager:
    cpdef serialize(self, string path):
        serialize_trace(self.trace_manager, path)

    cpdef void load_traces_from_file(self, string path):
        cdef TraceManagerPtr[NONTERMINAL, size_t] tm = load_trace_manager[NONTERMINAL, size_t](path)
        self.trace_manager = tm

    cpdef Enumerator get_nonterminal_map(self):
        raise NotImplementedError()

    cpdef is_consistent_with_grammar(self, PyGrammarInfo grammarInfo, size_t traceId=0):
        cdef Trace[NONTERMINAL, size_t]* trace = &(deref(fool_cython_unwrap(self.trace_manager))[traceId])
        cdef shared_ptr[GrammarInfo2] gI = grammarInfo.grammarInfo
        return deref(trace).is_consistent_with_grammar(deref(gI))

    cpdef void set_io_cycle_limit(self, unsigned int io_cycle_limit):
        deref(fool_cython_unwrap(self.trace_manager)).set_io_cycle_limit(io_cycle_limit)

    cpdef void set_io_precision(self, double io_precision):
        deref(fool_cython_unwrap(self.trace_manager)).set_io_precision(io_precision)

    def viterbi_derivation(self, size_t traceId, vector[double] edge_weights, grammar, op=prod, log_mode=True):
        """
        :param traceId: trace of which the viterbi derivation shall be computed
        :type traceId: size_t
        :param edge_weights: non-negative weights for each edge (ordered according to edge ordering in hypergraph)
        :type edge_weights: list[double]
        :param grammar:
        :type grammar: RTG_like
        :return: the Viterbi derivation
        :param op: path operation (binary operation on double)
        :rtype: AbstractDerivation
        Uses Knuth's generalization of Dijkstra's algorithm to compute the best derivation of some hypergraph.
        cf. https://doi.org/10.1016/0020-0190(77)90002-3
        """

        cdef Trace[NONTERMINAL, size_t]* trace = &(deref(fool_cython_unwrap(self.trace_manager))[traceId])

        cdef cmap[Element[Node[NONTERMINAL]], double] node_best_weight
        cdef cmap[Element[Node[NONTERMINAL]], size_t] node_best_edge
        cdef cmap[void*, size_t] edge_weights_dict
        cdef edge_idx
        cdef HyperEdge[Node[NONTERMINAL], size_t]* edge

        cdef shared_ptr[Manager[HyperEdge[Node[NONTERMINAL], size_t]]] edges
        edges = deref(deref(trace).get_hypergraph()).get_edges().lock()

        cdef size_t source_list_idx
        cdef size_t sources_size

        # cdef cset[Element[Node[NONTERMINAL]]] U
        cdef intp_std_func_priority_queue[Node[NONTERMINAL]] U = \
            intp_std_func_priority_queue[Node[NONTERMINAL]](construct_comparator[Node[NONTERMINAL]](node_best_weight))
        cdef cset[Element[Node[NONTERMINAL]]] Q

        for edge_idx in range(deref(edges).size()):
            edge = &deref(edges)[edge_idx]
            edge_weights_dict[edge] = edge_idx


            sources_size = deref(edge).get_sources().size()
            # print("iterated over edge", edge_idx, sources_size)
            if sources_size == 0:
                if node_best_weight.count(deref(edge).get_target()) == 0\
                        or node_best_weight[deref(edge).get_target()] < edge_weights[edge_idx]:
                    node_best_weight[deref(edge).get_target()] = edge_weights[edge_idx]
                    node_best_edge[deref(edge).get_target()] = edge_idx
                    U.push(deref(edge).get_target())
            else:
                if node_best_weight.count(deref(edge).get_target()) == 0:
                    node_best_weight[deref(edge).get_target()] = 0.0 if not log_mode else -INFINITY
                    node_best_edge[deref(edge).get_target()] = <size_t> (-1)

            # U.insert(deref(edge).get_target())

        cdef cset[Element[Node[NONTERMINAL]]].iterator A
        cdef cset[Element[Node[NONTERMINAL]]].iterator it

        # cdef size_t position

        cdef bint all_sources_in_Q
        cdef double weight

        while not U.empty(): # U.size() > 0:

            # finding element with maximum weight in U, i.e.,
            #  A = max(U, key=lambda x: node_best_weight[x])
            if Q.count(U.top()):
                U.pop()
                continue
            A = Q.insert(U.top()).first
            U.pop()

            # A = U.begin()
            # it = U.begin()
            # while it != U.end():
            #     if node_best_weight[deref(A)] <= node_best_weight[deref(it)]:
            #         A = it
            #     inc(it)
            #
            # it = A
            # A = Q.insert(deref(A)).first
            # U.erase(it)

            # abort early since root has been found
            if deref(A).equals(deref(trace).get_goal()):
                break

            for edge_idx in range(deref(deref(trace).get_hypergraph()).get_outgoing_edges(deref(A)).size()):
                edge = deref(deref(trace).get_hypergraph()).get_outgoing_edges(deref(A))[edge_idx].first.get()
                # position = deref(deref(trace).get_hypergraph()).get_outgoing_edges(deref(A))[edge_idx].second

                weight = edge_weights[edge_weights_dict[edge]]
                    # if not log_mode else edge_weights[edge_weights_dict[edge]]
                all_sources_in_Q = True

                for source_list_idx in range(deref(edge).get_sources().size()):
                    if Q.count(deref(edge).get_sources()[source_list_idx]) == 0:
                        all_sources_in_Q = False
                        break
                    if log_mode:
                        weight = weight + node_best_weight[deref(edge).get_sources()[source_list_idx]]
                    else:
                        weight = op(weight, node_best_weight[deref(edge).get_sources()[source_list_idx]])

                if all_sources_in_Q:
                    if False and (isnan(weight) or isinf(weight)):
                        print("Weight:", weight, "=", edge_weights[edge_weights_dict[edge]], end=" ")
                        for source_list_idx in range(deref(edge).get_sources().size()):
                            print("*", node_best_weight[deref(edge).get_sources()[source_list_idx]], end=" ")
                        print()
                    if weight > node_best_weight[deref(edge).get_target()] \
                            or (node_best_edge[deref(edge).get_target()] == <size_t> (-1)
                                and not in_vector(deref(edge).get_target(), deref(edge).get_sources())):
                        node_best_weight[deref(edge).get_target()] = weight
                        node_best_edge[deref(edge).get_target()] = edge_weights_dict[edge]
                        U.push(deref(edge).get_target())

        # print("best weights and incoming edges")
        # for i, target_element in enumerate(sorted(node_best_weight)):
        #     print(i, node_best_weight[target_element], node_best_edge[target_element])

        cdef DerivationTree tree
        cdef vector[Element[Node[NONTERMINAL]]] history

        if node_best_edge[deref(trace).get_goal()] != <size_t> (-1):
            try:
                tree = self.__build_viterbi_derivation_tree_rec(deref(trace).get_goal(), node_best_edge, edges, history)
                return TraceManagerDerivation(tree, grammar)
            except Exception:
                pass
        return None

    cdef DerivationTree __build_viterbi_derivation_tree_rec(self,
                                                            Element[Node[NONTERMINAL]] node,
                                                            cmap[Element[Node[NONTERMINAL]], size_t] node_best_edge,
                                                            shared_ptr[Manager[HyperEdge[Node[NONTERMINAL], size_t]]] edges,
                                                            vector[Element[Node[NONTERMINAL]]] history):
        cdef size_t best_edge_idx = node_best_edge[node]
        if best_edge_idx == <size_t> (-1):
            print("Unexpected edge idx")
            raise Exception()
        if in_vector(node, history):
            print("Cyclic derivation")
            raise Exception()
        history.push_back(node)

        cdef HyperEdge[Node[NONTERMINAL], size_t]* edge = &(deref(edges)[best_edge_idx])
        cdef list children = []
        cdef size_t child_list_idx

        for child_list_idx in range(deref(edge).get_sources().size()):
            children.append(self.__build_viterbi_derivation_tree_rec(deref(edge).get_sources()[child_list_idx], node_best_edge, edges, history))
        cdef size_t rule_id = deref(edge).get_label()

        history.pop_back()

        return DerivationTree(rule_id, children)


    def latent_viterbi_derivation(self, size_t traceID, PyLatentAnnotation latentAnnotation, grammar, bint debug=False):
        cdef Trace[NONTERMINAL, size_t]* trace = &(deref(fool_cython_unwrap(self.trace_manager))[traceID])
        cdef pair[size_t, unordered_map[pair[Element[Node[NONTERMINAL]], size_t],
                                   pair[Element[HyperEdge[Node[NONTERMINAL], size_t]], vector[size_t]]]] \
                result = deref(trace).computeViterbiPath(deref(latentAnnotation.latentAnnotation), debug)

        if result.first == <size_t> (-1):
            return None
        return TraceManagerDerivation(
            self.__build_viterbi_derivation_tree_rec_(
              trace.get_goal()
            , result.first
            , result.second
            )
            , grammar
        )

    cdef DerivationTree __build_viterbi_derivation_tree_rec_(
            self, Element[Node[NONTERMINAL]] node
                , size_t sub
                , unordered_map[  pair[Element[Node[NONTERMINAL]], size_t]
                            , pair[Element[HyperEdge[Node[NONTERMINAL], size_t]], vector[size_t]]]
                  node_best_edge
        ):
        # cdef size_t best_edge_idx = node_best_edge[node]
        # if best_edge_idx == <size_t> (-1):
        #     print("Unexpected edge idx")
        #     raise Exception()
        # cdef pair[Element[HyperEdge[NONTERMINAL, size_t]], vector[size_t]]& best = node_best_edge.at(node)
        cdef pair[Element[Node[NONTERMINAL]], size_t]* sub_node = new pair[Element[Node[NONTERMINAL]], size_t](node, sub)
        cdef Element[HyperEdge[Node[NONTERMINAL], size_t]]* edge = &node_best_edge.at(deref(sub_node)).first
        cdef vector[size_t] index = node_best_edge.at(deref(sub_node)).second
        cdef list children = []
        cdef size_t child_list_idx
        cdef Element[Node[NONTERMINAL]]* child

        for child_list_idx in range(deref(edge).get().get_sources().size()):
            children.append (
                self.__build_viterbi_derivation_tree_rec_
                    ( deref(edge).get().get_sources()[child_list_idx]
                    , index[child_list_idx + 1]
                    , node_best_edge
                    )
                )
        cdef size_t rule_id = deref(edge).get().get_label()

        return DerivationTree(rule_id, children)


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
                rule_id = deref(deref(edge).get()).get_label()
                yield DerivationTree(rule_id, list(children))


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
        self.__root_idx = 0 # tree.root_id
        self.__ids = []
        self.__rules = {}
        self.__child_ids = {}
        self.__relative_positions = {}
        self.spans = None
        self.__fill_recursive(tree, grammar, 0)

    def __fill_recursive(self, DerivationTree tree, grammar, size_t next_pos):
        cdef size_t current_pos = next_pos
        self.__ids.append(current_pos)
        self.__rules[current_pos] = grammar.rule_index(tree.rule_id)
        self.__child_ids[current_pos] = []
        cdef DerivationTree child_

        for i, child in enumerate(tree.children):
            child_ = <DerivationTree> child
            self.__child_ids[current_pos].append(next_pos + 1)
            self.__relative_positions[next_pos + 1] = (current_pos, i)
            next_pos = self.__fill_recursive(child_, grammar, next_pos + 1)

        return next_pos


cdef class DerivationTree:
    def __init__(self, size_t rule_id, list children):
        self.rule_id = rule_id
        self.children = children