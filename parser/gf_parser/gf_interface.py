import pgf
from parser.parser_interface import AbstractParser
from parser.derivation_interface import AbstractDerivation
from parser.gf_parser.gf_export import compile_gf_grammar, export, LANGUAGE, COMPILED_SUFFIX
from grammar.linearization import Enumerator
from collections import defaultdict
from math import exp


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

    def getRule(self, id):
        exp = self.nodes[id]
        rule_id = int(exp.unpack()[0][4:])
        return self.rules.index_object(rule_id)

    def root_id(self):
        return tuple([])

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


class GFParser_k_best(GFParser):
    def recognized(self):
        return self._viterbi is not None

    def __init__(self, grammar, input, k=1):
        gf_grammar, self.rules = grammar.tmp_gf
        # assert isinstance(gf_grammar, pgf.Concr)
        assert isinstance(self.rules, Enumerator)
        self._derivations = []
        try:
            i = gf_grammar.parse(' '.join(input), n=k)
            for obj in i:
                self._derivations.append(obj)
            self._viterbi_weigth, self._viterbi = self._derivations[0]

        except pgf.ParseError:
            self._viterbi_weigth = None
            self._viterbi = None

    def k_best_derivation_trees(self):
        for weight, gf_deriv in self._derivations:
            yield weight, GFDerivation(self.rules, gf_deriv)

    def viterbi_derivation(self):
        if self._viterbi is not None:
            return GFDerivation(self.rules, self._viterbi)
        else:
            return None

    def viterbi_weight(self):
        return exp(-self._viterbi_weigth)

    def best_trees(self, derivation_to_tree):
        weights = defaultdict(lambda: 0.0)
        witnesses = defaultdict(list)
        for i, (weight, der) in enumerate(self.k_best_derivation_trees()):
            tree = derivation_to_tree(der)
            weights[tree] += exp(-weight)
            witnesses[tree] += [i+1]
        the_derivations = weights.items()
        the_derivations.sort(key=lambda x: x[1], reverse=True)
        return [(tree, weight, witnesses[tree]) for tree,weight in the_derivations]