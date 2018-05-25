import pgf
from parser.parser_interface import AbstractParser
from grammar.lcfrs_derivation import LCFRSDerivation
from parser.gf_parser.gf_export import compile_gf_grammar, export, LANGUAGE, COMPILED_SUFFIX
from math import exp
import os.path
from functools import reduce

DEFAULT_PREFIX = '/tmp/'
DEFAULT_NAME = 'gfgrammar'


class GFDerivation(LCFRSDerivation):
    def __init__(self, grammar, expr):
        """
        :param grammar:
        :type grammar: LCFRS
        :param expr:
        :type expr: pgf.Expr
        """
        self.grammar = grammar
        self.nodes = {}
        self.children = {}

        def populate(expr, gorn):
            rule_string, children = expr.unpack()
            self.nodes[tuple(gorn)] = int(rule_string[4:])
            self.children[tuple(gorn)] = len(children)
            for i, child in enumerate(children):
                populate(child, gorn + [i])

        populate(expr, [])

        self._compute_spans()

    def child_ids(self, id):
        child_count = self.children[id]
        return [tuple(list(id) + [c]) for c in range(child_count)]

    def getRule(self, id):
        return self.grammar.rule_index(self.nodes[id])

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

    def __init__(self, grammar, input=None, save_preprocessing=None, load_preprocessing=None, heuristics=-1.0):
        self.grammar = grammar
        self._goal = None
        self._best = None
        self._heuristics = heuristics
        if input is not None:
            if grammar.tmp_gf is not None:
                self.gf_grammar = grammar.tmp_gf
            else:
                self.preprocess_grammar(grammar)
                self.gf_grammar = grammar.tmp_gf
            self.input = input
            self.parse()
        else:
            if load_preprocessing is not None:
                self.gf_grammar = pgf.readPGF(self.resolve_path(load_preprocessing)).languages[load_preprocessing[1] + LANGUAGE]
            else:
                if save_preprocessing is not None:
                    prefix = save_preprocessing[0]
                    name = save_preprocessing[1]
                    override = True
                else:
                    prefix = DEFAULT_PREFIX
                    name = DEFAULT_NAME
                    override = False
                self.gf_grammar = self._preprocess(grammar, prefix=prefix, name=name, override=override)

    @staticmethod
    def resolve_path(path):
        # print path
        # print os.path.join(path[0], path[1] + COMPILED_SUFFIX)
        return os.path.join(path[0], path[1] + COMPILED_SUFFIX)

    def set_input(self, input):
        self.input = input

    def parse(self):
        try:
            i = self.gf_grammar.parse(' '.join(self.input), n=1, heuristics=self._heuristics)
            self._best, self._goal = next(i)
        except pgf.ParseError:
            self._best = None
            self._goal = None

    def clear(self):
        self.input = None
        self._best = None
        self._goal = None

    def recognized(self):
        return self._goal is not None

    def best(self):
        return self._best

    def best_derivation_tree(self):
        if self._goal is not None:
            return GFDerivation(self.grammar, self._goal)
        else:
            return None

    @staticmethod
    def _preprocess(grammar, prefix=DEFAULT_PREFIX, name=DEFAULT_NAME, override=False):
        name_ = export(grammar, prefix, name, override)
        return_code = compile_gf_grammar(prefix, name_)
        if return_code != 0:
            print("Grammar could not be compiled! (return code", return_code, ")")
        gf_grammar = pgf.readPGF(os.path.join(prefix, name_ + COMPILED_SUFFIX)).languages[name_ + LANGUAGE]
        return gf_grammar

    @staticmethod
    def preprocess_grammar(grammar):
        # print gf_grammar
        grammar.tmp_gf = GFParser._preprocess(grammar)


class GFParser_k_best(GFParser):
    def recognized(self):
        return self._viterbi is not None

    def __init__(self, grammar, input=None, save_preprocessing=None, load_preprocessing=None, k=1, heuristics=-1.0):
        self._derivations = []
        self.k = k
        self._viterbi = None
        self._viterbi_weigth = None
        GFParser.__init__(self, grammar, input, save_preprocessing, load_preprocessing, heuristics=heuristics)

    def set_input(self, input):
        self.input = input

    def clear(self):
        self._derivations = []
        self._viterbi = None
        self._viterbi_weigth = None
        self._goal = None
        self.input = None

    def parse(self):
        try:
            i = self.gf_grammar.parse(' '.join(self.input), n=self.k, heuristics=self._heuristics)
            for obj in i:
                self._derivations.append(obj)
            self._viterbi_weigth, self._viterbi = self._derivations[0]
            self._goal = self._viterbi

        except pgf.ParseError:
            self._viterbi_weigth = None
            self._viterbi = None
            self._goal = None

    def k_best_derivation_trees(self):
        for weight, gf_deriv in self._derivations:
            der = GFDerivation(self.grammar, gf_deriv)
            try:
                probability = exp(-weight)
            except OverflowError:
                # print("Recieved invalid derivation weight from GF parser", weight)
                probability = reduce(lambda x, y: x * y, [der.getRule(idx).weight() for idx in der.ids()], 1.0)

            yield probability, GFDerivation(self.grammar, gf_deriv)

    def viterbi_derivation(self):
        if self._viterbi is not None:
            return GFDerivation(self.grammar, self._viterbi)
        else:
            return None

    def viterbi_weight(self):
        return exp(-self._viterbi_weigth)


__all__ = ["GFParser", "GFParser_k_best"]