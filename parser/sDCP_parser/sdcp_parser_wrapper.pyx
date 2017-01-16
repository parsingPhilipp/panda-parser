import grammar.lcfrs as gl
import grammar.dcp as gd
import hybridtree.general_hybrid_tree as gh
import parser.parser_interface as pi
import parser.derivation_interface as di
import random
import itertools
import time
from collections import defaultdict

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
# from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from math import exp

# this typedef seems necessary,
# since the compiler does not accept "vector[unsigned int]" or "vector[unsigned]"
# but accepts "vector[unsigned_int]"
ctypedef unsigned int unsigned_int

# this needs to be consistent
DEF ENCODE_NONTERMINALS = True
ctypedef unsigned_int NONTERMINAL
DEF ENCODE_TERMINALS = True
ctypedef unsigned_int TERMINAL

cdef extern from "SplitMergeUtil.h":
    unsigned_int indexation(vector[unsigned_int], vector[unsigned_int])

cdef extern from "SDCP.h":
    cdef cppclass Rule[Nonterminal, Terminal]:
         Rule(Nonterminal)
         void add_nonterminal(Nonterminal)
         void next_inside_attribute()
         void set_id(int)
         int get_id()
         void next_word_function_argument()
         void add_var_to_word_function(int,int)
         void add_terminal_to_word_function(Terminal)

    cdef cppclass STermBuilder[Nonterminal, Terminal]:
         void add_var(int, int)
         void add_terminal(Terminal, int)
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
        void add_node(Position, Terminal, Terminal, Position)
        void add_child(Position, Position)
        void set_entry(Position)
        void set_exit(Position)
        void is_initial(Position)
        Terminal get_label(Position)
        Position get_next(Position)
        void output()
        void set_linearization(vector[Position])

    cdef void output_helper(string)

cdef extern from "SDCP_Parser.h":
    cdef cppclass SDCPParser[Nonterminal,Terminal,Position]:
        SDCPParser()
        SDCPParser(bint,bint,bint,bint)
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

cdef extern from "Trace.h":
    cdef cppclass Double
    cdef cppclass LogDouble
    cdef cppclass GrammarInfo[Nonterminal]:
        GrammarInfo()
    cdef cppclass TraceManager[Nonterminal,Terminal,Position]:
        TraceManager(bint)
        void add_trace_from_parser(SDCPParser[Nonterminal, Terminal, Position], unsigned)
        vector[double] do_em_training[Val](vector[double], vector[vector[unsigned_int]], unsigned)
        pair[vector[unsigned_int], vector[vector[double]]] split_merge_id[Val](
                  vector[double]
                , vector[vector[unsigned_int]]
                , unsigned_int
                , unsigned_int
                , unsigned_int
                , double
        )
        pair[vector[unsigned_int], vector[vector[double]]] split_merge[Val](
                  vector[double]
                , vector[vector[unsigned_int]]
                , unsigned_int
                , unordered_map[Nonterminal,unsigned_int]
                , unsigned_int
                , double
        )
        pair [vector [pair [ Nonterminal
                            , pair[vector [pair[Position,Position]]
                            , pair[vector [pair[Position,Position]]
                            , vector [pair[Position,Position]]]]
                            ]
                            ]
             , pair [vector[vector[pair[unsigned_int, vector[unsigned_int]]]]
             , unsigned_int]] serialize(unsigned_int)
        void deserialize(
                pair [vector [pair [ Nonterminal
                            , pair[vector [pair[Position,Position]]
                            , pair[vector [pair[Position,Position]]
                            , vector [pair[Position,Position]]]]
                            ]
                            ]
             , pair [vector[vector[pair[unsigned_int, vector[unsigned_int]]]]
             , unsigned_int]]
             , SDCP[Nonterminal, Terminal]
        )
        unsigned traces_size()

        pair[vector[unsigned_int], vector[vector[double]]] run_split_merge_cycle[Val](
                  GrammarInfo[Nonterminal]
                , vector[unsigned_int]
                , vector[vector[double]]
                , unsigned_int
                , unsigned_int
                , double
                , unsigned_int
        )
        vector[vector[double]] lift_doubles(vector[double])
        GrammarInfo[NONTERMINAL] grammar_info_id(vector[vector[unsigned_int]])



# choose representation: prob / log-prob
ctypedef LogDouble SemiRing
# ctypedef LogDouble SemiRing

cdef HybridTree[TERMINAL, int]* convert_hybrid_tree(p_tree, terminal_encoding=str) except * :
    # output_helper("convert hybrid tree: " + str(p_tree))
    cdef HybridTree[TERMINAL, int]* c_tree = new HybridTree[TERMINAL, int]()
    assert isinstance(p_tree, gh.HybridTree)
    cdef vector[int] linearization = [-1] * len(p_tree.id_yield())
    c_tree[0].set_entry(0)
    # output_helper(str(p_tree.root))
    (last, _) = insert_nodes_recursive(p_tree, c_tree, p_tree.root, 0, False, 0, 0, linearization, terminal_encoding)
    c_tree[0].set_exit(last)
    # output_helper(str(linearization))
    c_tree[0].set_linearization(linearization)
    return c_tree


cdef pair[int,int] insert_nodes_recursive(p_tree, HybridTree[TERMINAL, int]* c_tree, p_ids, int pred_id, attach_parent, int parent_id, int max_id, vector[int] & linearization, terminal_encoding) except *:
    # output_helper(str(p_ids))
    if p_ids == []:
        return pred_id, max_id
    p_id = p_ids[0]
    cdef c_id = max_id + 1
    max_id += 1
    c_tree[0].add_node(pred_id, terminal_encoding(str(p_tree.node_token(p_id).pos()) + " : " + str(p_tree.node_token(p_id).deprel())), terminal_encoding(str(p_tree.node_token(p_id).pos())), c_id)
    if p_tree.in_ordering(p_id):
        linearization[p_tree.node_index(p_id)] = c_id
    if attach_parent:
        c_tree[0].add_child(parent_id, c_id)
    if p_tree.children(p_id):
        c_tree[0].add_child(c_id, c_id + 1)
        (_, max_id) = insert_nodes_recursive(p_tree, c_tree, p_tree.children(p_id), c_id + 1, True, c_id, c_id + 1, linearization, terminal_encoding)
    return insert_nodes_recursive(p_tree, c_tree, p_ids[1:], c_id, attach_parent, parent_id, max_id, linearization, terminal_encoding)


cdef class Enumerator:
    cdef unsigned counter
    cdef dict obj_to_ind
    cdef dict ind_to_obj
    cdef unsigned first_index

    def __init__(self, first_index=0):
        self.first_index = first_index
        self.counter = first_index
        self.obj_to_ind = {}
        self.ind_to_obj = {}

    def index_object(self, int i):
        """
        :type i: int
        :return:
        """
        return self.ind_to_obj[i]

    cdef int object_index(self, obj):
        if obj in self.obj_to_ind:
            return self.obj_to_ind[obj]
        else:
            self.obj_to_ind[obj] = self.counter
            self.ind_to_obj[self.counter] = obj
            self.counter += 1
            return self.counter - 1


cdef SDCP[NONTERMINAL, TERMINAL] grammar_to_SDCP(grammar, Enumerator rule_map,  nonterminal_encoder, terminal_encoder, lcfrs_conversion=False) except *:
    cdef SDCP[NONTERMINAL, TERMINAL] sdcp
    cdef Rule[NONTERMINAL, TERMINAL]* c_rule
    cdef int arg, mem
    cdef PySTermBuilder py_builder = PySTermBuilder()
    converter = STermConverter(py_builder, terminal_encoder)

    assert isinstance(grammar, gl.LCFRS)

    for rule in grammar.rules():
        converter.set_rule(rule)
        c_rule = new Rule[NONTERMINAL,TERMINAL](nonterminal_encoder(rule.lhs().nont()))
        c_rule[0].set_id(rule_map.object_index(rule))
        for nont in rule.rhs():
            c_rule[0].add_nonterminal(nonterminal_encoder(nont))
        mem = -3
        arg = 0
        for equation in rule.dcp():
            assert isinstance(equation, gd.DCP_rule)
            assert mem < equation.lhs().mem()
            while mem < equation.lhs().mem() - 1:
                c_rule[0].next_inside_attribute()
                mem += 1
                arg = 0
            assert mem == equation.lhs().mem() - 1

            converter.evaluateSequence(equation.rhs())
            py_builder.add_to_rule(c_rule)
            converter.clear()
            arg += 1
        # create remaining empty attributes
        while mem < len(rule.rhs()) - 2:
            c_rule[0].next_inside_attribute()
            mem += 1

        # create LCFRS component
        if (lcfrs_conversion):
            for argument in rule.lhs().args():
                c_rule[0].next_word_function_argument()
                for obj in argument:
                    if isinstance(obj, gl.LCFRS_var):
                        c_rule[0].add_var_to_word_function(obj.mem + 1, obj.arg + 1)
                    else:
                        c_rule[0].add_terminal_to_word_function(terminal_encoder(str(obj)))


        if not sdcp.add_rule(c_rule[0]):
            output_helper(str(rule))
            raise Exception("rule does not satisfy parser restrictions")
        del c_rule

    sdcp.set_initial(nonterminal_encoder(grammar.start()))
    # sdcp.output()
    return sdcp


def print_grammar(grammar):
    cdef Enumerator rule_map = Enumerator()
    cdef Enumerator nonterminal_map = Enumerator()
    cdef Enumerator terminal_map = Enumerator()
    nonterminal_encoder = lambda s: nonterminal_map.object_index(s) if ENCODE_NONTERMINALS else str
    terminal_encoder = lambda s: terminal_map.object_index(s) if ENCODE_TERMINALS else str

    cdef SDCP[NONTERMINAL, TERMINAL] sdcp = grammar_to_SDCP(grammar, rule_map, nonterminal_encoder, terminal_encoder)
    sdcp.output()

def print_grammar_and_parse_tree(grammar, tree):
    cdef Enumerator rule_map = Enumerator()
    cdef Enumerator nonterminal_map = Enumerator()
    cdef Enumerator terminal_map = Enumerator()
    nonterminal_encoder = lambda s: nonterminal_map.object_index(s) if ENCODE_NONTERMINALS else str
    terminal_encoder = lambda s: terminal_map.object_index(s) if ENCODE_TERMINALS else str

    cdef SDCP[NONTERMINAL, TERMINAL] sdcp = grammar_to_SDCP(grammar, rule_map, nonterminal_encoder, terminal_encoder)
    sdcp.output()

    cdef HybridTree[TERMINAL,int]* c_tree = convert_hybrid_tree(tree, str)
    c_tree[0].output()

    cdef SDCPParser[NONTERMINAL,TERMINAL,int] parser
    parser.set_input(c_tree[0])
    parser.set_sDCP(sdcp)
    parser.do_parse()
    parser.set_goal()
    parser.reachability_simplification()
    parser.print_trace()
    del c_tree


cdef class PySTermBuilder:
    cdef STermBuilder[NONTERMINAL, TERMINAL] builder
    cdef STermBuilder[NONTERMINAL, TERMINAL] get_builder(self):
        return self.builder
    def add_terminal(self, TERMINAL term, int position):
        self.builder.add_terminal(term, position)
    def add_var(self, int mem, int arg):
        self.builder.add_var(mem, arg)
    def add_children(self):
        self.builder.add_children()
    def move_up(self):
        self.builder.move_up()
    def clear(self):
        self.builder.clear()
    cdef void add_to_rule(self, Rule[NONTERMINAL, TERMINAL]* rule):
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
        self.builder.add_terminal(self.terminal_encoder(str(pos) + " : " + str(index.dep_label())), i)

    def evaluateString(self, s, id):
        # print s
        self.builder.add_terminal(self.terminal_encoder(s))

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

    def __init__(self, py_builder, terminal_encoder):
        self.builder = py_builder
        self.terminal_encoder = terminal_encoder

    def set_rule(self, rule):
        self.rule = rule

    def get_evaluation(self):
        return self.builder.get_sTerm()

    def  get_pybuilder(self):
        return self.builder

    def clear(self):
        self.builder.clear()


cdef class PyParseItem:
    cdef ParseItem[NONTERMINAL,int] item

    cdef set_item(self, ParseItem[NONTERMINAL,int] item):
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

    cdef ParseItem[NONTERMINAL,int] get_c_item(self):
        return self.item

    def __str__(self):
        return self.nonterminal + " " + str(self.inherited) + " " + str(self.synthesized)

    def serialize(self):
        return self.nonterminal, self.inherited, self.synthesized


cdef class PySDCPParser(object):
    cdef SDCP[NONTERMINAL,TERMINAL] sdcp
    cdef SDCPParser[NONTERMINAL,TERMINAL,int]* parser
    cdef Enumerator rule_map, terminal_map, nonterminal_map
    cdef bint debug

    def __init__(self, lcfrs_parsing=False, debug=False):
        self.debug = debug
        self.parser = new SDCPParser[NONTERMINAL,TERMINAL,int](lcfrs_parsing, debug, True, True)

    cdef void set_sdcp(self, SDCP[NONTERMINAL,TERMINAL] sdcp):
        self.sdcp = sdcp
        self.parser[0].set_sDCP(sdcp)

    cdef void set_rule_map(self, Enumerator rule_map):
        self.rule_map = rule_map

    cdef void set_terminal_map(self, Enumerator terminal_map):
        self.terminal_map = terminal_map

    cdef void set_nonterminal_map(self, Enumerator nonterminal_map):
        self.nonterminal_map = nonterminal_map

    def do_parse(self):
        self.parser[0].do_parse()
        if self.debug:
            output_helper("parsing completed\n")

        self.parser[0].reachability_simplification()

        if self.debug:
            output_helper("reachability simplification completed\n")
            self.parser[0].print_trace()
            output_helper("trace printed\n")

    def recognized(self):
        return self.parser.recognized()

    def set_input(self, tree):
        cdef HybridTree[TERMINAL,int]* c_tree
        if ENCODE_TERMINALS:
            c_tree = convert_hybrid_tree(tree, lambda s: self.terminal_map.object_index(s))
        else:
            c_tree = convert_hybrid_tree(tree)
        self.parser[0].set_input(c_tree[0])
        self.parser[0].set_goal()
        if self.debug:
            c_tree[0].output()

    def query_trace(self, PyParseItem item):
        result = []
        trace_items = self.parser[0].query_trace(item.item)
        for p in trace_items:
            children = []
            for item_ in p.second:
                py_item_ = PyParseItem()
                py_item_.set_item(item_)
                children.append(py_item_)
            result.append((p.first.get_id(), children))
        return result

    def all_derivation_trees(self):
        if self.debug:
            self.parser[0].input.output()
        der = SDCPDerivation(0, self.rule_map)
        goal_py = PyParseItem()
        goal_py.set_item(self.parser[0].goal[0])
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
        self.parser[0].clear()

    def __del__(self):
        del self.parser


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
        cdef Enumerator rule_map = Enumerator()
        cdef Enumerator nonterminal_map = Enumerator()
        cdef Enumerator terminal_map = Enumerator()
        nonterminal_encoder = (lambda s: nonterminal_map.object_index(s)) if ENCODE_NONTERMINALS else lambda s: str(s)
        terminal_encoder = (lambda s: terminal_map.object_index(s)) if ENCODE_TERMINALS else lambda s: str(s)

        cdef SDCP[NONTERMINAL, TERMINAL] sdcp = grammar_to_SDCP(grammar, rule_map, nonterminal_encoder, terminal_encoder)

        parser = PySDCPParser()
        parser.set_sdcp(sdcp)
        parser.set_rule_map(rule_map)
        parser.set_terminal_map(terminal_map)
        parser.set_nonterminal_map(nonterminal_map)
        grammar.sdcp_parser = parser


class LCFRS_sDCP_Parser(PysDCPParser):
    @staticmethod
    def preprocess_grammar(grammar):
        """
        :type grammar: LCFRS
        """
        cdef Enumerator rule_map = Enumerator()
        cdef Enumerator nonterminal_map = Enumerator()
        cdef Enumerator terminal_map = Enumerator()
        nonterminal_encoder = (lambda s: nonterminal_map.object_index(s)) if ENCODE_NONTERMINALS else lambda s: str(s)
        terminal_encoder = (lambda s: terminal_map.object_index(s)) if ENCODE_TERMINALS else lambda s: str(s)

        cdef SDCP[NONTERMINAL, TERMINAL] sdcp = grammar_to_SDCP(grammar, rule_map, nonterminal_encoder, terminal_encoder, lcfrs_conversion=True)

        parser = PySDCPParser(lcfrs_parsing=True, debug=False)
        parser.set_sdcp(sdcp)
        parser.set_rule_map(rule_map)
        parser.set_terminal_map(terminal_map)
        parser.set_nonterminal_map(nonterminal_map)
        grammar.sdcp_parser = parser


cdef class PyTrace:
    cdef TraceManager[NONTERMINAL,TERMINAL,int]* trace_manager
    cdef PySDCPParser parser
    cdef bint debug
    cdef vector[vector[unsigned_int]] cycle_nont_dimensions
    cdef vector[vector[vector[double]]] cycle_i_weights

    def __init__(self, grammar, lcfrs_parsing=True, debug=False):
        """
        :param grammar:
        :type grammar: gl.LCFRS
        :param lcfrs_parsing:
        :type lcfrs_parsing:
        :param debug:
        :type debug:
        """
        output_helper("initializing PyTrace")

        cdef Enumerator rule_map = Enumerator(first_index=0)
        cdef Enumerator nonterminal_map = Enumerator()
        cdef Enumerator terminal_map = Enumerator()
        nonterminal_encoder = (lambda s: nonterminal_map.object_index(s)) if ENCODE_NONTERMINALS else lambda s: str(s)
        terminal_encoder = (lambda s: terminal_map.object_index(s)) if ENCODE_TERMINALS else lambda s: str(s)

        cdef SDCP[NONTERMINAL, TERMINAL] sdcp = grammar_to_SDCP(grammar, rule_map, nonterminal_encoder, terminal_encoder, lcfrs_conversion=lcfrs_parsing)

        self.parser = PySDCPParser(lcfrs_parsing, debug)
        self.trace_manager = new TraceManager[NONTERMINAL,TERMINAL,int](debug)
        self.parser.set_sdcp(sdcp)
        self.parser.set_rule_map(rule_map)
        self.parser.set_terminal_map(terminal_map)
        self.parser.set_nonterminal_map(nonterminal_map)
        self.cycle_i_weights = []
        self.cycle_nont_dimensions = []

    def compute_reducts(self, corpus):
        start_time = time.time()
        for i, tree in enumerate(corpus):
            self.parser.clear()
            self.parser.set_input(tree)
            self.parser.do_parse()
            if self.parser.recognized():
                self.trace_manager[0].add_trace_from_parser(self.parser.parser[0], i)
            if i % 100 == 0:
                output_helper(str(i) + ' ' + str(time.time() - start_time))

    def em_training(self, grammar, n_epochs, init="rfe", tie_breaking=False, sigma=0.005, seed=0):
        random.seed(seed)
        assert isinstance(grammar, gl.LCFRS)
        normalization_groups = []
        rule_to_group = {}
        for nont in grammar.nonts():
            normalization_group = []
            for rule in grammar.lhs_nont_to_rules(nont):
                rule_idx = self.parser.rule_map.object_index(rule)
                normalization_group.append(rule_idx)
                rule_to_group[rule_idx] = self.parser.nonterminal_map.object_index(nont)
            normalization_groups.append(normalization_group)
        initial_weights = [0.0] * self.parser.rule_map.first_index
        for i in xrange(self.parser.rule_map.first_index, self.parser.rule_map.counter):
            if init == "rfe":
                prob = self.parser.rule_map.index_object(i).weight()
            elif init == "equal" or True:
                prob = 1.0 / len(normalization_groups[rule_to_group[i]])

            # this may violates properness
            # but EM makes the grammar proper again
            if tie_breaking:
                prob_new = random.gauss(prob, sigma)
                while prob_new < 0.0:
                    prob_new = random.gauss(prob, sigma)
                prob = prob_new

            initial_weights.append(prob)

        final_weights = self.trace_manager[0].do_em_training[SemiRing](initial_weights, normalization_groups, n_epochs)

        for i in range(self.parser.rule_map.first_index, self.parser.rule_map.counter):
            self.parser.rule_map.index_object(i).set_weight(final_weights[i])

    def split_merge_training(self, grammar, cycles, em_epochs, init="rfe", tie_breaking=True, sigma=0.005, seed=0, merge_threshold=0.5, rule_pruning=exp(-100)):
        random.seed(seed)
        assert isinstance(grammar, gl.LCFRS)
        normalization_groups = []
        rule_to_nonterminals = []
        rule_to_group = {}
        nont_map = self.parser.nonterminal_map
        cdef long i
        for nont in grammar.nonts():
            i = nont_map.object_index(nont)
            normalization_group = []
            for rule in grammar.lhs_nont_to_rules(nont):
                rule_idx = self.parser.rule_map.object_index(rule)
                normalization_group.append(rule_idx)
                rule_to_group[rule_idx] = i
            normalization_groups.append(normalization_group)

        for i in xrange(self.parser.rule_map.first_index, self.parser.rule_map.counter):
            rule = self.parser.rule_map.index_object(i)
            nonts = [nont_map.object_index(rule.lhs().nont())] + [nont_map.object_index(nont) for nont in rule.rhs()]
            rule_to_nonterminals.append(nonts)

        initial_weights = [0.0] * self.parser.rule_map.first_index
        for i in xrange(self.parser.rule_map.first_index, self.parser.rule_map.counter):
            if init == "rfe":
                prob = self.parser.rule_map.index_object(i).weight()
            elif init == "equal" or True:
                prob = 1.0 / len(normalization_groups[rule_to_group[i]])

            # this may violates properness
            # but EM makes the grammar proper again
            if tie_breaking:
                prob_new = random.gauss(prob, sigma)
                while prob_new < 0.0:
                    prob_new = random.gauss(prob, sigma)
                prob = prob_new

            initial_weights.append(prob)

        if not ENCODE_NONTERMINALS or len(self.cycle_nont_dimensions) == 0:
            pre_weights = self.trace_manager[0].do_em_training[SemiRing](initial_weights, normalization_groups, em_epochs)

        output_helper("computing split weights")

        cdef vector[unsigned_int] nont_dimensions
        cdef vector[vector[double]] weights

        cdef unsigned_int n_nonts

        cdef GrammarInfo[NONTERMINAL] grammar_info

        if ENCODE_NONTERMINALS:
            n_nonts = len(grammar.nonts())
            grammar_info = self.trace_manager[0].grammar_info_id(rule_to_nonterminals)
            output_helper("Computed grammar info")
            if len(self.cycle_nont_dimensions) == 0:
                self.cycle_nont_dimensions.push_back([1] * n_nonts)
                self.cycle_i_weights.push_back(self.trace_manager[0].lift_doubles(pre_weights))
            output_helper("Initialized split info")
            for cycle in range(cycles):
                if len(self.cycle_nont_dimensions) <= cycle + 1:

                    self.cycle_nont_dimensions.push_back([])
                    self.cycle_i_weights.push_back([])

                    assert len(self.cycle_i_weights) == len(self.cycle_nont_dimensions)

                    output_helper("Starting " + str(cycle + 1) + ". S/M cycle")
                    # output_helper("Length nont dimensions: " + str(len(self.cycle_nont_dimensions)))
                    # output_helper("Length " + str(cycle) + "entry: " + str(len(self.cycle_nont_dimensions[cycle])))
                    output_helper(str(self.cycle_nont_dimensions[cycle]))
                    the_time = time.time()

                    self.cycle_nont_dimensions[cycle+1], self.cycle_i_weights[cycle+1] \
                        = self.trace_manager[0].run_split_merge_cycle[SemiRing]( grammar_info
                                                , self.cycle_nont_dimensions[cycle]
                                                , self.cycle_i_weights[cycle]
                                                , em_epochs
                                                , n_nonts
                                                , merge_threshold
                                                , cycle)
                    output_helper("Finished "+ str(cycle + 1) + ". S/M cycle in " + str(time.time() - the_time) + " seconds.")
                    output_helper(str(self.cycle_nont_dimensions[cycle+1]))
                    output_helper("Cycle " + str(cycle + 1) + " Rule weights: #" + str(len(self.cycle_i_weights[cycle + 1])))
                else:
                    output_helper("Cycle " + str(cycle + 1) + " Rule weights: #" + str(len(self.cycle_i_weights[cycle + 1])))
                new_grammar = self.build_sm_grammar(grammar, self.cycle_nont_dimensions[cycle + 1], rule_to_nonterminals, self.cycle_i_weights[cycle+1])
                yield new_grammar

            # nont_dimensions, weights = self.trace_manager[0].split_merge_id[SemiRing](pre_weights, rule_to_nonterminals, em_epochs, n_nonts, cycles, merge_threshold)
        else:
            nont_dimensions, weights = self.trace_manager[0].split_merge[SemiRing](pre_weights, rule_to_nonterminals, em_epochs, nont_map.obj_to_ind, cycles, merge_threshold)

            yield self.build_sm_grammar(grammar, nont_dimensions, rule_to_nonterminals, weights)

    def build_sm_grammar(self, grammar, nont_dimensions, rule_to_nonterminals, weights):
        new_grammar = gl.LCFRS(grammar.start() + "[0]")
        for i in xrange(self.parser.rule_map.first_index, self.parser.rule_map.counter):
            rule = self.parser.rule_map.index_object(i)

            rule_dimensions = [nont_dimensions[nont] for nont in rule_to_nonterminals[i]]
            rule_dimensions_exp = itertools.product(*[xrange(dim) for dim in rule_dimensions])

            for la in rule_dimensions_exp:
                index = indexation(list(la), rule_dimensions)
                # output_helper(str(i) + " " + str(rule_dimensions) + " " + str(list(la)) + " " + str(index))
                weight = weights[i][index]
                if weight > exp(-100):
                    lhs_la = gl.LCFRS_lhs(rule.lhs().nont() + "[" + str(la[0]) + "]")
                    for arg in rule.lhs().args():
                        lhs_la.add_arg(arg)
                    nonts = [rhs_nont + "[" + str(la[1 + j]) + "]" for j, rhs_nont in enumerate(rule.rhs())]
                    new_grammar.add_rule(lhs_la, nonts, weight, rule.dcp())

        return new_grammar

    def __del__(self):
        del self.trace_manager

    def serialize_la_state(self):
        return self.cycle_nont_dimensions, self.cycle_i_weights

    def deserialize_la_state(self, nont_dim_list, weights_list):
        self.cycle_nont_dimensions = nont_dim_list
        self.cycle_i_weights = weights_list

    def serialize_trace(self):
        the_trace_list = []
        for i in range(self.trace_manager.traces_size()):
            the_trace_list.append(self.trace_manager.serialize(i))
        return the_trace_list

    def deserialize_trace(self, serialization):
        for entry in serialization:
            self.trace_manager.deserialize(entry, self.parser.sdcp)


def em_training(grammar, corpus, n_epochs, init="rfe", tie_breaking=False, sigma=0.005, seed=0, debug=False):
    output_helper("creating trace")
    trace = PyTrace(grammar, debug=debug)
    output_helper("computing reducts")
    trace.compute_reducts(corpus)
    output_helper("starting actual training")
    trace.em_training(grammar, n_epochs, init, tie_breaking, sigma, seed)

def split_merge_training(grammar, corpus, cycles, em_epochs, init="rfe", tie_breaking=False, sigma=0.005, seed=0, merge_threshold=0.5, debug=False, rule_pruning=exp(-100)):
    output_helper("creating trace")
    trace = PyTrace(grammar, debug=debug)
    output_helper("computing reducts")
    trace.compute_reducts(corpus)

    if debug:
        for i in xrange(trace.parser.rule_map.first_index, trace.parser.rule_map.counter):
            output_helper(str(i) + " " + str(trace.parser.rule_map.index_object(i)))

    output_helper("starting actual split/merge training")
    grammar = trace.split_merge_training(grammar, cycles, em_epochs, init, tie_breaking, sigma, seed, merge_threshold, rule_pruning)

    return grammar

def compute_reducts(grammar, corpus, debug=False):
    output_helper("creating trace")
    trace = PyTrace(grammar, debug=debug)
    output_helper("computing reducts")
    trace.compute_reducts(corpus)
    return trace

def load_reducts(grammar, serialization, debug=False):
    output_helper("creating trace")
    trace = PyTrace(grammar, debug=debug)
    output_helper("restoring reducts")
    trace.deserialize_trace(serialization)
    return trace