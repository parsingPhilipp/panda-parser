from libcpp.memory cimport make_shared
from util.enumerator cimport Enumerator

cdef class PyGrammarInfo:
    def __init__(self, grammar, Enumerator nont_map):
        """
        :type grammar: gl.LCFRS
        """
        cdef vector[vector[size_t]] rule_to_nonterminals = []
        cdef size_t i
        # for i in range(traceManager.parser.rule_map.first_index, traceManager.parser.rule_map.counter):
        for i in range(0, len(grammar.rule_index())):
            rule = grammar.rule_index(i)
            nonts = [nont_map.object_index(rule.lhs().nont())] + [nont_map.object_index(nont) for nont in rule.rhs()]
            rule_to_nonterminals.push_back(nonts)

        self.grammarInfo = make_shared[GrammarInfo2](rule_to_nonterminals, nont_map.object_index(grammar.start()))

cdef class PyStorageManager:
    def __init__(self, bint selfMalloc=False):
        self.storageManager = make_shared[StorageManager](selfMalloc)