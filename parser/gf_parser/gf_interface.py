import pgf
from parser.parser_interface import AbstractParser
from parser.derivation_interface import AbstractDerivation
from parser.gf_parser.gf_export import compile_gf_grammar, export, LANGUAGE, COMPILED_SUFFIX
from grammar.linearization import Enumerator


prefix = '/tmp/'
name = 'gfgrammar'


class GFDerivation(AbstractDerivation):
    def __init__(self, rules, expr):
        """
        :param rules:
        :type rules: Enumerator
        :param expr:
        :type expr: pgf.Expr
        """
        self.rules = rules
        self.nodes = {}

        def populate(expr, gorn):
            self.nodes[tuple(gorn)] = expr
            _, children = expr.unpack()
            for i, child in enumerate(children):
                populate(child, gorn + [i])

        populate(expr, [])

        self._compute_spans()

    def child_ids(self, id):
        exp = self.nodes[id]
        child_count = len(exp.unpack()[1])
        return [tuple(list(id) + [c]) for c in range(child_count)]

    def __str__(self):
        return self.der_to_str_rec(self.root_id(), 0)

    def der_to_str_rec(self, item, indentation):
        s = ' ' * indentation * 2 + str(self.getRule(item)) + '\t(' + str(self.spans[item]) + ')\n'
        for child in self.child_ids(item):
            s += self.der_to_str_rec(child, indentation + 1)
        return s

    def getRule(self, id):
        exp = self.nodes[id]
        rule_id = int(exp.unpack()[0][4:])
        return self.rules.index_object(rule_id)

    def root_id(self):
        return tuple([])

    def terminal_positions(self, id):
        def spanned_positions(id_):
            return [x + 1 for (l,r) in self.spans[id_] for x in range(l, r)]
        own = spanned_positions(id)
        children = [x for cid in self.child_ids(id) for x in spanned_positions(cid)]
        return [x for x in own if not x in children]

    def ids(self):
        return self.nodes.keys()

    def position_relative_to_parent(self, id):
        return id[:-1], id[-1]

    def child_id(self, id, i):
        return tuple(list(id) + [i])


class GFParser(AbstractParser):
    def all_derivation_trees(self):
        pass

    def __init__(self, grammar, input):
        gf_grammar, self.rules = grammar.tmp_gf
        # assert isinstance(gf_grammar, pgf.Concr)
        assert isinstance(self.rules, Enumerator)
        try:
            i = gf_grammar.parse(' '.join(input), n=1)
            self._best, self._goal = i.next()
        except pgf.ParseError:
            self._best = None
            self._goal = None

    def recognized(self):
        return self._goal is not None

    def best(self):
        return self._best

    def best_derivation_tree(self):
        if self._goal is not None:
            return GFDerivation(self.rules, self._goal)
        else:
            return None

    @staticmethod
    def preprocess_grammar(grammar):
        rules, name_ = export(grammar, prefix, name)
        compile_gf_grammar(prefix, name_)
        gf_grammar = pgf.readPGF(prefix + name_ + COMPILED_SUFFIX).languages[name_ + LANGUAGE]
        # print gf_grammar
        grammar.tmp_gf = gf_grammar, rules