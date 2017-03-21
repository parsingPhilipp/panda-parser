from cython.operator cimport dereference as deref
from grammar.lcfrs import LCFRS as PyLCFRS, LCFRS_var as PyLCFRS_var

# Options:
DEF ENCODE_NONTERMINALS = True
DEF ENCODE_TERMINALS = True

cdef class PyLCFRSFactory:
    def __cinit__(self, initial_nont, Enumerator ntMap=Enumerator()):
        IF ENCODE_NONTERMINALS:
            self.ntMap = ntMap
            self._thisptr = make_unique[LCFRSFactory[NONTERMINAL,TERMINAL]](self.ntMap.object_index(initial_nont))
        ELSE:
            self._thisptr = make_unique[LCFRSFactory[NONTERMINAL,TERMINAL]](initial_nont)

        # IF ENCODE_TERMINALS:
        self.tMap = Enumerator()


    cpdef void new_rule(self, NONTERMINAL lhsNont):
        deref(self._thisptr).new_rule(lhsNont)

    cpdef void add_terminal(self, TERMINAL term):
        deref(self._thisptr).add_terminal(term)

    cpdef void add_variable(self, unsigned long index, unsigned long arg):
        deref(self._thisptr).add_variable(index,arg)

    cpdef void complete_argument(self):
        deref(self._thisptr).complete_argument()

    cpdef void add_rule_to_grammar(self, vector[NONTERMINAL] rhs, const unsigned long ruleId):
        deref(self._thisptr).add_rule_to_grammar(rhs, ruleId)

    cpdef PyLCFRSParser build_parser(self):
        cdef PyLCFRSParser parser = PyLCFRSParser(#deref(self._thisptr).get_grammar(),
                             self.tMap)
        parser.set_grammar(deref(self._thisptr).get_grammar())
        return parser

    def import_grammar(self, grammar):
        # :type grammar PyLCFRS
        # :return:

        for rule in grammar.rules():
            IF ENCODE_NONTERMINALS:
                self.new_rule(self.ntMap.object_index(rule.lhs().nont()))
            ELSE:
                self.new_rule(rule.lhs().nont())
            for argument in rule.lhs().args():
                for symbol in argument:
                    if type(symbol) is PyLCFRS_var:
                        self.add_variable(symbol.mem, symbol.arg)
                    else:
                        IF ENCODE_NONTERMINALS:
                            self.add_terminal(self.tMap.object_index(symbol))
                        ELSE:
                            self.add_terminal(symbol)
                self.complete_argument()
            IF ENCODE_NONTERMINALS:
                self.add_rule_to_grammar(<vector[NONTERMINAL]> self.ntMap.objects_indices(rule.rhs()), rule.get_idx())
            ELSE:
                self.add_rule_to_grammar(rule.rhs(), rule.get_idx())


cdef class PyLCFRSParser:
    def __cinit__(self, Enumerator tMap):
        self.tMap = tMap

    cdef set_grammar(self, shared_ptr[LCFRS[NONTERMINAL, TERMINAL]] grammar):
        self.grammar = grammar

    cpdef void do_parse(self, word):
        IF ENCODE_TERMINALS:
            cdef vector[TERMINAL] words_encoded = <vector[TERMINAL]> self.tMap.objects_indices(word)
            self.parser = make_unique[LCFRS_Parser[NONTERMINAL, TERMINAL]](deref(self.grammar), words_encoded)
        ELSE:
            self.parser = make_unique[LCFRS_Parser[NONTERMINAL, TERMINAL]](deref(self.grammar), word)
        deref(self.parser).do_parse()

    cpdef void prune_trace(self):
        deref(self.parser).prune_trace()

    cpdef map[unsigned_long, pair[NONTERMINAL, vector[pair[unsigned_long, unsigned_long]]]] get_passive_items_map(self):
        return deref(self.parser).get_passive_items_map()

    cpdef map[unsigned_long, vector[pair[unsigned_long, vector[unsigned_long]]]] convert_trace(self):
        return deref(self.parser).convert_trace()

    def get_initial_passive_item(self):
        return deref(self.parser).get_initial_passive_item()