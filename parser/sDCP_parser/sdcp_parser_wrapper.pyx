import grammar.lcfrs as gl
from grammar.lcfrs import LCFRS
import grammar.dcp as gd
import hybridtree.general_hybrid_tree as gh
import parser.parser_interface as pi
import parser.derivation_interface as di
from collections import defaultdict

# this needs to be consistent
DEF ENCODE_NONTERMINALS = True
# ctypedef unsigned_int NONTERMINAL
DEF ENCODE_TERMINALS = True
# ctypedef unsigned_int TERMINAL

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


# cdef class Enumerator:
#     # cdef unsigned counter
#     # cdef dict obj_to_ind
#     # cdef dict ind_to_obj
#     # cdef unsigned first_index
#
#     def __init__(self, size_t first_index=0):
#         self.first_index = first_index
#         self.counter = first_index
#         self.obj_to_ind = {}
#         self.ind_to_obj = {}
#
#     def index_object(self, size_t i):
#         """
#         :type i: int
#         :return:
#         """
#         return self.ind_to_obj[i]
#
#     cdef size_t object_index(self, obj):
#         if obj in self.obj_to_ind:
#             return self.obj_to_ind[obj]
#         else:
#             self.obj_to_ind[obj] = self.counter
#             self.ind_to_obj[self.counter] = obj
#             self.counter += 1
#             return self.counter - 1
#
#     # cdef vector[size_t] orderd_objects(self):
#     #     return [self.ind_to_obj[idx] for idx in range(0, self.counter)]



cdef SDCP[NONTERMINAL, TERMINAL] grammar_to_SDCP(grammar, nonterminal_encoder, terminal_encoder, lcfrs_conversion=False) except *:
    cdef SDCP[NONTERMINAL, TERMINAL] sdcp
    cdef Rule[NONTERMINAL, TERMINAL]* c_rule
    cdef int arg, mem
    cdef PySTermBuilder py_builder = PySTermBuilder()
    converter = STermConverter(py_builder, terminal_encoder)

    assert isinstance(grammar, gl.LCFRS)

    for rule in grammar.rules():
        converter.set_rule(rule)
        c_rule = new Rule[NONTERMINAL,TERMINAL](nonterminal_encoder(rule.lhs().nont()))
        c_rule[0].set_id(rule.get_idx()) # rule_map.object_index(rule))
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
    # cdef Enumerator rule_map = Enumerator()
    cdef Enumerator nonterminal_map = Enumerator()
    cdef Enumerator terminal_map = Enumerator()
    nonterminal_encoder = lambda s: nonterminal_map.object_index(s) if ENCODE_NONTERMINALS else str
    terminal_encoder = lambda s: terminal_map.object_index(s) if ENCODE_TERMINALS else str

    cdef SDCP[NONTERMINAL, TERMINAL] sdcp = grammar_to_SDCP(grammar, nonterminal_encoder, terminal_encoder)
    sdcp.output()

def print_grammar_and_parse_tree(grammar, tree):
    # cdef Enumerator rule_map = Enumerator()
    cdef Enumerator nonterminal_map = Enumerator()
    cdef Enumerator terminal_map = Enumerator()
    nonterminal_encoder = lambda s: nonterminal_map.object_index(s) if ENCODE_NONTERMINALS else str
    terminal_encoder = lambda s: terminal_map.object_index(s) if ENCODE_TERMINALS else str

    cdef SDCP[NONTERMINAL, TERMINAL] sdcp = grammar_to_SDCP(grammar, nonterminal_encoder, terminal_encoder)
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
    # cdef SDCP[NONTERMINAL,TERMINAL] sdcp
    # cdef SDCPParser[NONTERMINAL,TERMINAL,int]* parser
    # cdef Enumerator rule_map, terminal_map, nonterminal_map
    # cdef bint debug

    def __init__(self, grammar, lcfrs_parsing=False, debug=False):
        self.debug = debug
        self.parser = new SDCPParser[NONTERMINAL,TERMINAL,int](lcfrs_parsing, debug, True, True)
        # self.__grammar = grammar

    cdef void set_sdcp(self, SDCP[NONTERMINAL,TERMINAL] sdcp):
        self.sdcp = sdcp
        self.parser[0].set_sDCP(sdcp)

    cdef void set_terminal_map(self, Enumerator terminal_map):
        self.terminal_map = terminal_map

    cdef void set_nonterminal_map(self, Enumerator nonterminal_map):
        self.nonterminal_map = nonterminal_map

    cpdef void do_parse(self):
        self.parser[0].do_parse()
        if self.debug:
            output_helper("parsing completed\n")

        self.parser[0].reachability_simplification()

        if self.debug:
            output_helper("reachability simplification completed\n")
            self.parser[0].print_trace()
            output_helper("trace printed\n")

    cpdef bint recognized(self):
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

    def all_derivation_trees(self, grammar):
        if self.debug:
            self.parser[0].input.output()
        der = SDCPDerivation(0, grammar)
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

    cpdef void clear(self):
        self.parser[0].clear()

    def __del__(self):
        del self.parser


class SDCPDerivation(di.AbstractDerivation):
    def __init__(self, max_idx, grammar, idx_to_rule=defaultdict(lambda: None), children=defaultdict(lambda: []), parent=defaultdict(lambda: None)):
        self.max_idx = max_idx
        self.idx_to_rule = idx_to_rule.copy()
        self.children = children.copy()
        self.parent = parent.copy()
        self.grammar = grammar
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
        new_deriv = SDCPDerivation(self.max_idx, self.grammar, self.idx_to_rule, self.children, self.parent)

        new_deriv.idx_to_rule[idx] = self.grammar.rule_index(rule_id)

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
        self.grammar = grammar

    def recognized(self):
        return self.parser.recognized()

    def best_derivation_tree(self):
        pass

    def best(self):
        pass

    def all_derivation_trees(self):
        if self.recognized():
            return self.parser.all_derivation_trees(self.grammar)
        else:
            return []

    @staticmethod
    def preprocess_grammar(grammar):
        """
        :type grammar: LCFRS
        """
        cdef Enumerator nonterminal_map = Enumerator()
        cdef Enumerator terminal_map = Enumerator()
        nonterminal_encoder = (lambda s: nonterminal_map.object_index(s)) if ENCODE_NONTERMINALS else lambda s: str(s)
        terminal_encoder = (lambda s: terminal_map.object_index(s)) if ENCODE_TERMINALS else lambda s: str(s)

        cdef SDCP[NONTERMINAL, TERMINAL] sdcp = grammar_to_SDCP(grammar,  nonterminal_encoder, terminal_encoder)

        parser = PySDCPParser(grammar)
        parser.set_sdcp(sdcp)
        # parser.set_rule_map(rule_map)
        parser.set_terminal_map(terminal_map)
        parser.set_nonterminal_map(nonterminal_map)
        grammar.sdcp_parser = parser


class LCFRS_sDCP_Parser(PysDCPParser):
    @staticmethod
    def preprocess_grammar(grammar):
        """
        :type grammar: LCFRS
        """
        cdef Enumerator nonterminal_map = Enumerator()
        cdef Enumerator terminal_map = Enumerator()
        nonterminal_encoder = (lambda s: nonterminal_map.object_index(s)) if ENCODE_NONTERMINALS else lambda s: str(s)
        terminal_encoder = (lambda s: terminal_map.object_index(s)) if ENCODE_TERMINALS else lambda s: str(s)

        cdef SDCP[NONTERMINAL, TERMINAL] sdcp = grammar_to_SDCP(grammar, nonterminal_encoder, terminal_encoder, lcfrs_conversion=True)

        parser = PySDCPParser(grammar, lcfrs_parsing=True, debug=False)
        parser.set_sdcp(sdcp)
        # parser.set_rule_map(rule_map)
        parser.set_terminal_map(terminal_map)
        parser.set_nonterminal_map(nonterminal_map)
        grammar.sdcp_parser = parser
