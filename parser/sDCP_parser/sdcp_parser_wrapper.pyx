import grammar.lcfrs as gl
import grammar.dcp as gd
import hybridtree.general_hybrid_tree as gh

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "SDCP.h":
    cdef cppclass Rule[Nonterminal, Terminal]:
         Rule(Nonterminal)
         void add_nonterminal(Nonterminal)
         void next_outside_attribute()
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
        void set_input(HybridTree[Terminal,Position])
        void set_sDCP(SDCP[Nonterminal, Terminal])
        void set_goal()
        void reachability_simplification()
        void print_chart()
        void print_trace()


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


cdef SDCP[string, string] grammar_to_SDCP(grammar):
    cdef SDCP[string, string] sdcp
    cdef Rule[string, string]* c_rule
    cdef int arg
    cdef PySTermBuilder py_builder = PySTermBuilder()
    converter = STermConverter(py_builder)

    assert isinstance(grammar, gl.LCFRS)

    for rule in grammar.rules():
        converter.set_rule(rule)
        c_rule = new Rule[string,string](rule.lhs().nont())
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


def print_grammar(grammar, tree):
    cdef SDCP[string,string] sdcp = grammar_to_SDCP(grammar)
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









        # c_rule[0]

    # return sdcp

