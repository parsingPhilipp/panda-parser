#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cython.operator cimport dereference as deref
from parser.derivation_interface import AbstractDerivation
from grammar.rtg import RTG

cdef extern from "Trainer/TraceManager.h" namespace "Trainer":
    cdef void add_hypergraph_to_trace[Nonterminal, TraceID](
            TraceManagerPtr[Nonterminal, TraceID] manager
            , shared_ptr[Hypergraph[Nonterminal, size_t]] hypergraph
            , Element[Node[Nonterminal]] root
            , double frequency)


cdef class PyElement:
    cdef shared_ptr[Element[Node[NONTERMINAL]]] element
    def __cinit__(self):
        self.element = shared_ptr[Element[Node[NONTERMINAL]]]()
         # self.element = make_shared[Element[Node[NONTERMINAL]]](element)

cdef class PyDerivationManager(PyTraceManager):
    def __init__(self, grammar, Enumerator nonterminal_map=None):
        """
        :param grammar:
        :type grammar: PyLCFRS
        """
        if nonterminal_map is None:
            nonterminal_map = Enumerator()
            for nont in grammar.nonts():
                nonterminal_map.object_index(nont)
        cdef vector[NONTERMINAL] node_labels = range(0, nonterminal_map.counter)
        self.node_labels = make_shared[vector[NONTERMINAL]](node_labels)
        cdef vector[size_t] edge_labels = range(0, len(grammar.rule_index()))
        self.edge_labels = make_shared[vector[size_t]](edge_labels)
        self.nonterminal_map = nonterminal_map

        self.trace_manager = build_trace_manager_ptr[NONTERMINAL, size_t](self.node_labels
            , self.edge_labels
            , False)

    cpdef void convert_derivations_to_hypergraphs(self, corpus):
        cdef shared_ptr[Hypergraph[NONTERMINAL, size_t]] hg
        cdef vector[Element[Node[NONTERMINAL]]] sources
        cdef PyElement pyElement

        for derivation in corpus:
            assert(isinstance(derivation, AbstractDerivation))
            hg = make_shared[Hypergraph[NONTERMINAL, size_t]](self.node_labels, self.edge_labels)
            nodeMap = {}

            # create nodes
            for node in derivation.ids():
                nont = derivation.getRule(node).lhs().nont()
                nLabel = self.nonterminal_map.object_index(nont)
                pyElement2 = PyElement()
                pyElement2.element = make_shared[Element[Node[NONTERMINAL]]](deref(hg).create(nLabel))
                nodeMap[node] = pyElement2

            # create edges
            for node in derivation.ids():
                eLabel = derivation.getRule(node).get_idx()
                for child in derivation.child_ids(node):
                    # nont = derivation.getRule(nont).lhs().nont()
                    pyElement = nodeMap[child]
                    sources.push_back(deref(pyElement.element))

                # target
                pyElement = nodeMap[node]
                deref(hg).add_hyperedge(eLabel, deref(pyElement.element), sources)
                sources.clear()

            # root
            pyElement = nodeMap[derivation.root_id()]
            add_hypergraph_to_trace[NONTERMINAL, size_t](self.trace_manager, hg, deref(pyElement.element), 1.0)
            # nodeMap.clear()

    cpdef Enumerator get_nonterminal_map(self):
        return self.nonterminal_map

    cpdef void convert_rtgs_to_hypergraphs(self, rtgs):
        """
        :param rtgs: a sequence of reduct RTG of the grammar passed in the constructor, i.e.,\n 
                     - nonterminals are of the the form (N, X) where N is a nonterminal in grammar,\n 
                     - the initial nonterminal is the start symbol of grammar,\n 
                     - rules are of the from (N, X) -> n((N_1, X_1) … (N_k, X_k)) where 
                       N -> N_1 … N_k is the rule with idx n in grammar
        :type rtgs: iterable[RTG]
        """
        cdef shared_ptr[Hypergraph[NONTERMINAL, size_t]] hg
        cdef vector[Element[Node[NONTERMINAL]]] sources
        cdef PyElement pyElement

        for rtg in rtgs:
            assert(isinstance(rtg, RTG))
            hg = make_shared[Hypergraph[NONTERMINAL, size_t]](self.node_labels, self.edge_labels)
            nodeMap = {}

            # create nodes
            for nont in rtg.nonterminals:
                orig_nont = nont[0]
                nLabel = self.nonterminal_map.object_index(orig_nont)
                pyElement2 = PyElement()
                pyElement2.element = make_shared[Element[Node[NONTERMINAL]]](deref(hg).create(nLabel))
                nodeMap[nont] = pyElement2

            # create edges
            for rule in rtg.rules:
                eLabel = rule.symbol
                for rhs_nont in rule.rhs:
                    pyElement = nodeMap[rhs_nont]
                    sources.push_back(deref(pyElement.element))

                # target
                pyElement = nodeMap[rule.lhs]
                deref(hg).add_hyperedge(eLabel, deref(pyElement.element), sources)
                sources.clear()

            # root
            pyElement = nodeMap[rtg.initial]
            add_hypergraph_to_trace[NONTERMINAL, size_t](self.trace_manager, hg, deref(pyElement.element), 1.0)
            # nodeMap.clear()

