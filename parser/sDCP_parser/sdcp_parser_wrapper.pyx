import grammar.lcfrs as gl
import grammar.dcp as gd
import hybridtree.general_hybrid_tree as gh
import parser.parser_interface as pi
import parser.derivation_interface as di
import copy
from collections import defaultdict

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "SDCP.h":
    cdef cppclass Rule[Nonterminal, Terminal]:
         Rule(Nonterminal)
         void add_nonterminal(Nonterminal)
         void next_outside_attribute()
         void set_id(int)
         int get_id()
         # void add_outside_attribute(STerm)
         # void add_sterm_from_builder(STermBuilder[Terminal])

    cdef cppclass STermBuilder[Nonterminal, Terminal]:
         void add_var(int, int)
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


cdef extern from "SDCP.h" namespace "boost":
    cdef cppclass variant

cdef extern from "HybridTree.h":
    cdef cppclass HybridTree[Terminal, Position]:
        void add_node(Position, Terminal, Position)
        void add_child(Position, Position)
        void set_entry(Position)
        void set_exit(Position)
        void is_initial(Position)
        Terminal get_label(Position)
        Position get_next(Position)
        void output()

    cdef void output_helper(string)

cdef extern from "SDCP_Parser.h":
    cdef cppclass SDCPParser[Nonterminal,Terminal,Position]:
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
        # vector[pair[Rule[Nonterminal,Terminal], vector[ParseItem[Nonterminal,Position]]]] \
        vector[pair[Rule,vector[ParseItem]]] query_trace(ParseItem)

    cdef cppclass ParseItem[Nonterminal,Position]:
        Nonterminal nonterminal
        vector[pair[Position,Position]] spans_inh
        vector[pair[Position,Position]] spans_syn

cdef HybridTree[string, int]* convert_hybrid_tree(p_tree):
    cdef HybridTree[string, int]* c_tree = new HybridTree[string, int]()
    assert isinstance(p_tree, gh.HybridTree)
    c_tree[0].set_entry(0)
    output_helper(str(p_tree.root))
    (last, _) = insert_nodes_recursive(p_tree, c_tree, p_tree.root, 0, False, 0, 0)
    c_tree[0].set_exit(last)
    return c_tree


cdef pair[int,int] insert_nodes_recursive(p_tree, HybridTree[string, int]* c_tree, p_ids, int pred_id, attach_parent, int parent_id, int max_id):
    output_helper(str(p_ids))
    if p_ids == []:
        return pred_id, max_id
    p_id = p_ids[0]
    cdef c_id = max_id + 1
    max_id += 1
    c_tree[0].add_node(pred_id, str(p_tree.node_token(p_id).pos()) + " : " + str(p_tree.node_token(p_id).deprel()), c_id)
    if attach_parent:
        c_tree[0].add_child(parent_id, c_id)
    if p_tree.children(p_id):
        c_tree[0].add_child(c_id, c_id + 1)
        (_, max_id) = insert_nodes_recursive(p_tree, c_tree, p_tree.children(p_id), c_id + 1, True, c_id, c_id + 1)
    return insert_nodes_recursive(p_tree, c_tree, p_ids[1:], c_id, attach_parent, parent_id, max_id)


cdef class Enumerator:
    cdef unsigned counter
    cdef dict obj_to_ind
    cdef dict ind_to_obj

    def __init__(self, first_index=1):
        self.counter = first_index
        self.obj_to_ind = {}
        self.ind_to_obj = {}

    def index_object(self, i):
        """
        :type i: int
        :return:
        """
        return self.ind_to_obj[i]

    cdef int object_index(self, obj):
        i = self.obj_to_ind.get(obj, None)
        if i:
            return i
        else:
            self.obj_to_ind[obj] = self.counter
            self.ind_to_obj[self.counter] = obj
            self.counter += 1
            return self.counter - 1


cdef SDCP[string, string] grammar_to_SDCP(grammar, Enumerator rule_map):
    cdef SDCP[string, string] sdcp
    cdef Rule[string, string]* c_rule
    cdef int arg
    cdef PySTermBuilder py_builder = PySTermBuilder()
    converter = STermConverter(py_builder)

    assert isinstance(grammar, gl.LCFRS)

    for rule in grammar.rules():
        converter.set_rule(rule)
        c_rule = new Rule[string,string](rule.lhs().nont())
        c_rule[0].set_id(rule_map.object_index(rule))
        for nont in rule.rhs():
            c_rule[0].add_nonterminal(nont)
        mem = -3
        arg = 0
        for equation in rule.dcp():
            assert isinstance(equation, gd.DCP_rule)
            assert mem < equation.lhs().mem()
            while mem < equation.lhs().mem() - 1:
                c_rule[0].next_outside_attribute()
                mem += 1
                arg = 0
            assert mem == equation.lhs().mem() - 1

            converter.evaluateSequence(equation.rhs())
            py_builder.add_to_rule(c_rule)
            converter.clear()
            arg += 1
        # create remaining empty attributes
        while mem < len(rule.rhs()) - 2:
            c_rule[0].next_outside_attribute()
            mem += 1

        assert sdcp.add_rule(c_rule[0])
        del c_rule

    sdcp.set_initial(grammar.start())
    return sdcp


def print_grammar(grammar):
    cdef Enumerator rule_map = Enumerator()
    cdef SDCP[string,string] sdcp = grammar_to_SDCP(grammar, rule_map)
    sdcp.output()

def print_grammar_and_parse_tree(grammar, tree):
    cdef Enumerator rule_map = Enumerator()
    cdef SDCP[string,string] sdcp = grammar_to_SDCP(grammar, rule_map)
    sdcp.output()

    cdef HybridTree[string,int]* c_tree = convert_hybrid_tree(tree)
    c_tree[0].output()

    cdef SDCPParser[string,string,int] parser
    parser.set_input(c_tree[0])
    parser.set_sDCP(sdcp)
    parser.do_parse()
    parser.set_goal()
    parser.reachability_simplification()
    parser.print_trace()
    del c_tree


cdef class PySTermBuilder:
    cdef STermBuilder[string, string] builder
    cdef STermBuilder[string, string] get_builder(self):
        return self.builder
    def add_terminal(self, string term):
        self.builder.add_terminal(term)
    def add_var(self, int mem, int arg):
        self.builder.add_var(mem, arg)
    def add_children(self):
        self.builder.add_children()
    def move_up(self):
        self.builder.move_up()
    def clear(self):
        self.builder.clear()
    cdef void add_to_rule(self, Rule[string, string]* rule):
        self.builder.add_to_rule(rule)





class STermConverter(gd.DCP_evaluator):
    def evaluateIndex(self, index, id):
        # print index
        cdef int i = index.index()
        rule = self.rule
        assert isinstance(rule, gl.LCFRS_rule)
        cdef int j = 0
        pos = None
        for arg in rule.lhs().args():
            for obj in arg:
                if isinstance(obj, gl.LCFRS_var):
                    continue
                if isinstance(obj, str):
                    if i == j:
                        pos = obj
                        break
                    j += 1
            if pos:
               break
        self.builder.add_terminal(str(pos) + " : " + str(index.dep_label()))

    def evaluateString(self, s, id):
        # print s
        self.builder.add_terminal(s)

    def evaluateVariable(self, var, id):
        # print var
        cdef int offset = 0
        if var.mem() >= 0:
            for dcp_eq in self.rule.dcp():
                if dcp_eq.lhs().mem() == var.mem():
                    offset += 1
        self.builder.add_var(var.mem() + 1, var.arg() + 1 - offset)

    def evaluateTerm(self, term, id):
        term.head().evaluateMe(self)
        if term.arg():
            self.builder.add_children()
            self.evaluateSequence(term.arg())
            self.builder.move_up()

    def evaluateSequence(self, sequence):
        for element in sequence:
            element.evaluateMe(self)

    def __init__(self, py_builder):
        self.builder = py_builder

    def set_rule(self, rule):
        self.rule = rule

    def get_evaluation(self):
        return self.builder.get_sTerm()

    def  get_pybuilder(self):
        return self.builder

    def clear(self):
        self.builder.clear()


cdef class PyParseItem:
    cdef ParseItem[string,int] item

    cdef set_item(self, ParseItem[string,int] item):
        self.item = item

    @property
    def nonterminal(self):
        return self.item.nonterminal

    @property
    def inherited(self):
        ranges = []
        for range in self.item.spans_inh:
            ranges.append((range.first, range.second))
        return ranges

    @property
    def synthesized(self):
        ranges = []
        for range in self.item.spans_syn:
            ranges.append((range.first, range.second))
        return ranges

    cdef ParseItem[string,int] get_c_item(self):
        return self.item

    def __str__(self):
        return self.nonterminal + " " + str(self.inherited) + " " + str(self.synthesized)


cdef class PySDCPParser(object):
    cdef SDCP[string,string] sdcp
    cdef SDCPParser[string,string,int] parser
    cdef Enumerator rule_map

    cdef void set_sdcp(self, SDCP[string,string] sdcp):
        self.sdcp = sdcp
        self.parser.set_sDCP(sdcp)

    cdef void set_rule_map(self, Enumerator rule_map):
        self.rule_map = rule_map

    def do_parse(self):
        self.parser.do_parse()
        self.parser.reachability_simplification()
        self.parser.print_trace()

    def recognized(self):
        return self.parser.recognized()

    def set_input(self, tree):
        cdef HybridTree[string,int]* c_tree = convert_hybrid_tree(tree)
        self.parser.set_input(c_tree[0])
        self.parser.set_goal()
        c_tree[0].output()

    def query_trace(self, PyParseItem item):
        result = []
        trace_items = self.parser.query_trace(item.item)
        for p in trace_items:
            children = []
            for item_ in p.second:
                py_item_ = PyParseItem()
                py_item_.set_item(item_)
                children.append(py_item_)
            result.append((p.first.get_id(), children))
        return result

    def all_derivation_trees(self):
        self.parser.input.output()
        der = SDCPDerivation(0, self.rule_map)
        goal_py = PyParseItem()
        goal_py.set_item(self.parser.goal[0])
        der.max_idx = 1
        return self.derivations_rec([goal_py], [1], der)

    # this expands the packed forest of parse items into an iterator over derivation trees
    def derivations_rec(self, list items, positions, derivation):
        assert isinstance(derivation, SDCPDerivation)

        # output_helper("items = [" + ', '.join(map(str, items)) +  ']' + '\n')
        # output_helper("positions = " + str(positions) + "\n")

        if len(items) == 0:
            yield derivation
            return

        position = positions[0]
        for rule_id, children in self.query_trace(items[0]):
            # output_helper("children = [" + ', '.join(map(str, children)) +  ']' + '\n')
            extended_derivation, child_positions = derivation.extend_by(position, rule_id, len(children))
            for vertical_extension in self.derivations_rec(children, child_positions, extended_derivation):
                for horizontal_extension in self.derivations_rec(items[1:], positions[1:], vertical_extension):
                    yield horizontal_extension

    def clear(self):
        self.parser.clear()




class SDCPDerivation(di.AbstractDerivation):
    def __init__(self, max_idx, rule_map, idx_to_rule=defaultdict(lambda: None), children=defaultdict(lambda: []), parent=defaultdict(lambda: None)):
        self.max_idx = max_idx
        self.idx_to_rule = idx_to_rule.copy()
        self.children = children.copy()
        self.parent = parent.copy()
        self.rule_map = rule_map
        self.spans = None

    def root_id(self):
        return min(self.max_idx, 1)

    def child_id(self, idx, i):
        return self.children[idx][i]

    def child_ids(self, idx):
        return self.children[idx]

    def ids(self):
        return range(1, self.max_idx + 1)

    def getRule(self, idx):
        return self.idx_to_rule[idx]

    def position_relative_to_parent(self, idx):
        p = self.parent[idx]
        return p, self.children[p].index(idx)

    def extend_by(self, int idx, int rule_id, int n_children):
        new_deriv = SDCPDerivation(self.max_idx, self.rule_map, self.idx_to_rule, self.children, self.parent)

        new_deriv.idx_to_rule[idx] = self.rule_map.index_object(rule_id)

        child_idx = new_deriv.max_idx
        first = child_idx + 1
        for child in range(n_children):
            child_idx += 1
            new_deriv.children[idx].append(child_idx)
            new_deriv.parent[child_idx] = idx
        new_deriv.max_idx = child_idx

        return new_deriv, range(first, child_idx + 1)




class PysDCPParser(pi.AbstractParser):
    def __init__(self, grammar, input):
        self.parser = grammar.sdcp_parser
        self.parser.clear()
        self.parser.set_input(input)
        self.parser.do_parse()

    def recognized(self):
        return self.parser.recognized()

    def best_derivation_tree(self):
        pass

    def best(self):
        pass

    def all_derivation_trees(self):
        if self.recognized():
            return self.parser.all_derivation_trees()
        else:
            return []

    @staticmethod
    def preprocess_grammar(grammar):
        """
        :type grammar: LCFRS
        """
        cdef Enumerator enum = Enumerator()
        cdef SDCP[string,string] sdcp = grammar_to_SDCP(grammar, enum)
        parser = PySDCPParser()
        parser.set_sdcp(sdcp)
        parser.set_rule_map(enum)
        grammar.sdcp_parser = parser

