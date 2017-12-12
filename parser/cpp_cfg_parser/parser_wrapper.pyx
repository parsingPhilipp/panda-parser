from grammar.lcfrs import LCFRS, LCFRS_rule, LCFRS_lhs, LCFRS_var
from util.enumerator cimport Enumerator
from parser.derivation_interface import AbstractDerivation
from parser.parser_interface import AbstractParser
cimport cython
from libc.stdlib cimport malloc, free
from math import log

cdef extern from "cfg.h":
    ctypedef unsigned Terminal
    ctypedef unsigned Nonterminal

cdef extern from "cfg.h" namespace "cyk":
    cdef cppclass CFG:
        CFG()
        CFG(Nonterminal initial)
        # void add_rule(Rule rule)
        void add_lex_rule(Nonterminal lhn, int idx, double weight, Terminal terminal);
        void add_chain_rule(Nonterminal lhn, int idx, double weight, Nonterminal rhs);
        void add_binary_rule(Nonterminal lhn, int idx, double weight, Nonterminal left_child, Nonterminal right_child);
        void set_initial(Nonterminal initial)

cdef extern from "parser.h" namespace "cyk":
    cdef cppclass CYKParser:
        CYKParser();
        CYKParser(CFG grammar)
        void parse_input(Terminal *input, unsigned length);
        CYKItem* get_goal();

    cdef cppclass CYKItem:
        unsigned left;
        unsigned right;
        double weight
        CYKItem * left_child
        CYKItem * right_child;
        int rule_idx() const;


cdef class PyCFG:
    cdef CFG cfg
    cdef Enumerator nonterminal_map, terminal_map, rule_map

    cdef void set_cfg(self, CFG cfg):
        self.cfg = cfg

    cpdef void set_enumerators(self, Enumerator nonterminal_map, Enumerator terminal_map, Enumerator rule_map):
        self.nonterminal_map = nonterminal_map
        self.terminal_map = terminal_map
        self.rule_map = rule_map

cdef PyCFG grammar_to_cfg(grammar, Enumerator terminal_map, Enumerator nonterminal_map, Enumerator rule_map):
    """
    :param grammar:
    :type grammar: LCFRS
    :return:
    :rtype: CFG
    """
    assert(isinstance(grammar, LCFRS))
    assert(isinstance(terminal_map, Enumerator))
    assert(isinstance(nonterminal_map, Enumerator))
    assert(isinstance(rule_map, Enumerator))

    cdef CFG cfg = CFG()

    for rule in grammar.rules():
        if len(rule.lhs().args()) > 1:
            raise Exception("Only CFGs in Chomsky normal form are supported, "
                            + "but grammar contained rule with higher fanout: "
                            + str(rule))

        nont_index = nonterminal_map.object_index(str(rule.lhs().nont()))
        rule_idx = rule_map.object_index(rule)
        if rule.weight() == 0.0:
            rule_weight = float("-inf")
        else:
            rule_weight = log(rule.weight())

        # lexical rules
        if rule.rank() == 0 and len(rule.lhs().arg(0)) == 1 and isinstance(rule.lhs().arg(0)[0], str):
            terminal = rule.lhs().arg(0)[0]
            term_idx = terminal_map.object_index(terminal)
            # print "Rule::LexRule(", nont_index, ", ", rule_idx, ", ", rule_weight, ", ", term_idx, ")"
            cfg.add_lex_rule(nont_index, rule_idx, rule_weight, term_idx)

        # chain rules
        elif rule.rank() == 1 and len(rule.lhs().arg(0)) == 1 and rule.lhs().arg(0)[0] == LCFRS_var(0, 0):
            rhs_index = nonterminal_map.object_index(str(rule.rhs_nont(0)))
            # print "Rule::ChainRule(", nont_index, ", ", rule_idx, ", ", rule_weight, ", ", rhs_index, ")"
            cfg.add_chain_rule(nont_index, rule_idx, rule_weight, rhs_index)

        # binary rules
        elif rule.rank() == 2 \
                and len(rule.lhs().arg(0)) == 2 \
                and rule.lhs().arg(0)[0] == LCFRS_var(0, 0)\
                and rule.lhs().arg(0)[1] == LCFRS_var(1, 0):
            rhs_index1 = nonterminal_map.object_index(str(rule.rhs_nont(0)))
            rhs_index2 = nonterminal_map.object_index(str(rule.rhs_nont(1)))
            # print "Rule::BinaryRule(", nont_index, ", ", rule_idx, ", ", rule_weight, ", ", rhs_index1, ", ", rhs_index2, ")"
            cfg.add_binary_rule(nont_index, rule_idx, rule_weight, rhs_index1, rhs_index2)

        else:
            raise Exception("Only CFGs in Chomsky normal form are supported, but grammar contained rule " + str(rule))

    start = nonterminal_map.object_index(str(grammar.start()))
    print(grammar.start(), start)
    cfg.set_initial(start)

    py_cfg = PyCFG()
    py_cfg.set_cfg(cfg)
    py_cfg.set_enumerators(nonterminal_map, terminal_map, rule_map)
    return py_cfg

cdef class PyCFGParser:
    cdef CYKParser cpp_parser
    cdef PyCFG py_cfg

    def __cinit__(self, PyCFG py_cfg):
        self.py_cfg = py_cfg
        self.cpp_parser = CYKParser(py_cfg.cfg)

    def parse_sentence(self, sentence):
        # todo: cython.sizeof(unsigned) does not compile?!
        my_ints = <unsigned *>malloc(len(sentence)*cython.sizeof(int))
        if my_ints is NULL:
            raise MemoryError()

        for i in range(len(sentence)):

            my_ints[i] = self.py_cfg.terminal_map.object_index(sentence[i])

        self.cpp_parser.parse_input(my_ints, len(sentence))

    def recognized(self):
        return not self.cpp_parser.get_goal() is NULL

    def goal(self):
        return convert_items(self.cpp_parser.get_goal()[0])

    def goal_weight(self):
        return self.cpp_parser.get_goal().weight

    def rule_map(self):
        return self.py_cfg.rule_map


class CFGParser(AbstractParser):
    def recognized(self):
        return self.parser.recognized()

    def all_derivation_trees(self):
        pass

    def __init__(self, grammar, input=None, save_preprocess=None, load_preprocess=None):
        self.input = input
        self.goal = None
        if input is not None:
            self.parser = grammar.tmp
            self.parse()
        else:
            self.parser = CFGParser.__preprocess(grammar)

    def best_derivation_tree(self):
        if self.recognized():
            return CFGDerivation(self.goal, self.parser.rule_map())
        else:
            return None

    def best(self):
        return self.parser.goal_weight()

    def clear(self):
        self.input = None
        self.goal = None

    def parse(self):
        self.parser.parse_sentence(self.input)
        if self.recognized():
            self.goal = self.parser.goal()

    def set_input(self, input):
        self.input = input

    @staticmethod
    def __preprocess(grammar):
        terminal_map, nonterminal_map, rule_map = Enumerator(0), Enumerator(0), Enumerator(0)
        pycfg = grammar_to_cfg(grammar, terminal_map, nonterminal_map, rule_map)
        return PyCFGParser(pycfg)

    @staticmethod
    def preprocess_grammar(grammar):
        grammar.tmp = CFGParser.__preprocess(grammar)


cdef class PyCYKItem:
    cdef public list children
    cdef public unsigned rule_idx
    cdef public unsigned left, right

    def __init__(self, left, right, rule_idx):
        self.rule_idx = rule_idx
        self.left = left
        self.right = right
        self.children = []


cdef PyCYKItem convert_items(CYKItem root):
    # rule = rule_map.index_object(root.rule_idx())
    cdef PyCYKItem root_ = PyCYKItem(root.left, root.right, root.rule_idx())
    if root.left_child != NULL:
        root_.children.append(convert_items(root.left_child[0]))
    if root.right_child != NULL:
        root_.children.append(convert_items(root.right_child[0]))
    return root_


class CFGDerivation(AbstractDerivation):
    def __init__(self, root_item, rule_map):
        """
        :param root_item:
        :type root_item: PyCYKItem
        """
        self.root_item = root_item
        self.parent = {}
        self.rule_map = rule_map
        self.populate(root_item)

    def populate(self, item):
        for child in item.children:
            self.populate(child)
            self.parent[child] = item

    def getRule(self, id):
        """
        :param id:
        :type id:
        :return:
        :rtype:
        """
        assert(isinstance(id, PyCYKItem))
        return self.rule_map.index_object(id.rule_idx)

    def child_id(self, id, i):
        return id.children[i]

    def child_ids(self, id):
        return id.children

    def terminal_positions(self, id):
        if not id.children:
            return [id.right]
        else:
            return []

    def position_relative_to_parent(self, id):
        return self.parent[id], self.parent[id].children.index(id)

    def root_id(self):
        return self.root_item

    def ids(self):
        return [self.root_item] + [key for key in self.parent.keys()]

    def __str__(self):
        return self.der_to_str_rec(self.root_id(), 0)

    def der_to_str_rec(self, item, indentation):
        s = ' ' * indentation * 2 + str(self.getRule(item)) + '\t(' + str(item.left) + ',' + str(item.right) + ')\n'
        for child in self.child_ids(item):
            s += self.der_to_str_rec(child, indentation + 1)
        return s