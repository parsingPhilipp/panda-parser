from __future__ import print_function, unicode_literals
import unittest
from grammar.lcfrs import LCFRS, LCFRS_var, LCFRS_lhs
from pprint import pprint
from parser.discodop_parser.grammar_adapter import transform_grammar
from parser.discodop_parser.parser import DiscodopKbestParser
from discodop.plcfrs import parse
from discodop.containers import Grammar
from discodop.kbest import lazykbest
from parser.supervised_trainer.trainer import PyDerivationManager
import tempfile
from parser.coarse_to_fine_parser.trace_weight_projection import py_edge_weight_projection
from parser.trace_manager.sm_trainer import build_PyLatentAnnotation_initial
from parser.trace_manager.sm_trainer_util import PyGrammarInfo, PyStorageManager


class DiscodopAdapterTest(unittest.TestCase):
    def build_grammar(self):
        grammar = LCFRS("START")
        # rule 0
        lhs = LCFRS_lhs("START")
        lhs.add_arg([LCFRS_var(0, 0)])
        grammar.add_rule(lhs, ["S"])

        # rule 1
        lhs = LCFRS_lhs("S")
        lhs.add_arg([LCFRS_var(0, 0), LCFRS_var(1, 0)])
        grammar.add_rule(lhs, ["S", "S"])

        # rule 1.5
        lhs = LCFRS_lhs("S")
        lhs.add_arg([LCFRS_var(0, 0), LCFRS_var(1, 0)])
        grammar.add_rule(lhs, ["S", "S"], dcp=["1.5"])

        # rule 2
        lhs = LCFRS_lhs("S")
        lhs.add_arg(["a"])
        grammar.add_rule(lhs, [])

        # rule 3
        lhs = LCFRS_lhs("S")
        lhs.add_arg(["b"])
        grammar.add_rule(lhs, [], weight=2.0)

        # rule 4
        lhs = LCFRS_lhs("S")
        lhs.add_arg(["b"])
        grammar.add_rule(lhs, [], dcp=["4"])

        # rule 5
        lhs = LCFRS_lhs("A")
        lhs.add_arg(["a"])
        grammar.add_rule(lhs, [])

        grammar.make_proper()
        return grammar

    def build_nm_grammar(self):
        grammar = LCFRS("START")
        # rule 0
        lhs = LCFRS_lhs("START")
        lhs.add_arg([LCFRS_var(0, 0)])
        grammar.add_rule(lhs, ["S"])

        # rule 1
        lhs = LCFRS_lhs("S")
        lhs.add_arg([LCFRS_var(0, 0), LCFRS_var(1, 0), LCFRS_var(0, 1), LCFRS_var(1, 1)])
        grammar.add_rule(lhs, ["N", "M"])

        for nont, term in [("A", "a"), ("B", "b"), ("C", "c"), ("D", "d")]:
            # rule 2
            lhs = LCFRS_lhs(nont)
            lhs.add_arg([term])
            grammar.add_rule(lhs, [])

        for nont, nont_, c1, c2 in [("N", "N'", "A", "C"), ("M", "M'", "B", "D")]:
            # rule 3
            lhs = LCFRS_lhs(nont)
            lhs.add_arg([LCFRS_var(0, 0)])
            lhs.add_arg([LCFRS_var(1, 0)])
            grammar.add_rule(lhs, [c1, c2])

            # rule 4
            lhs = LCFRS_lhs(nont)
            lhs.add_arg([LCFRS_var(0, 0), LCFRS_var(1, 0)])
            lhs.add_arg([LCFRS_var(0,1)])
            grammar.add_rule(lhs, [nont_, c1])

            # rule 5
            lhs = LCFRS_lhs(nont_)
            lhs.add_arg([LCFRS_var(0, 0)])
            lhs.add_arg([LCFRS_var(0, 1), LCFRS_var(1, 0)])
            grammar.add_rule(lhs, [nont, c2])

        grammar.make_proper()
        return grammar

    def test_discodop_kbest_parser(self):
        grammar = self.build_grammar()
        parser = DiscodopKbestParser(grammar)
        inp = ["a"] * 5
        parser.set_input(inp)
        parser.parse()
        self.assertTrue(parser.recognized())
        counter = 0
        for weight, der in parser.k_best_derivation_trees():
            # print(weight, der)
            self.assertTrue(der.check_integrity_recursive(der.root_id(), grammar.start()))
            counter += 1
        self.assertEqual(50, counter)

    def test_copy_grammar(self):
        grammar = self.build_nm_grammar()
        parser = DiscodopKbestParser(grammar)
        n = 2
        m = 3
        inp = ["a"] * n + ["b"] * m + ["c"] * n + ["d"] * m
        parser.set_input(inp)
        parser.parse()
        self.assertTrue(parser.recognized())
        counter = 0
        for weight, der in parser.k_best_derivation_trees():
            print(weight, der)
            self.assertTrue(der.check_integrity_recursive(der.root_id(), grammar.start()))
            self.assertEqual(inp, der.compute_yield())
            counter += 1
        self.assertEqual(1, counter)

    def test_something(self):
        grammar = self.build_grammar()

        for r in transform_grammar(grammar):
            pprint(r)

        rule_list = list(transform_grammar(grammar))
        pprint(rule_list)
        disco_grammar = Grammar(rule_list, start=grammar.start())
        print(disco_grammar)

        inp = ["a"] * 3
        chart, msg = parse(inp, disco_grammar)
        print(chart)
        print(msg)
        chart.filter()
        print("filtered chart")
        print(disco_grammar.nonterminals)
        print(type(disco_grammar.nonterminals))

        print(chart)
        # print(help(chart))

        root = chart.root()
        print("root", root, type(root))
        print(chart.indices(root))
        print(chart.itemstr(root))
        print(chart.stats())
        print("root label", chart.label(root))
        print(root, chart.itemid1(chart.label(root), chart.indices(root)))
        for i in range(1, chart.numitems()):
            # print(i, chart.label(i), chart.indices(i), chart.numedges(i))
            if True or len(chart.indices(i)) > 1:
                for edge_num in range(chart.numedges(i)):
                    edge = chart.getEdgeForItem(i, edge_num)
                    if isinstance(edge, tuple):
                        print("\t", disco_grammar.nonterminalstr(chart.label(i)) + "[" + str(i) + "]", "->", ' '.join([disco_grammar.nonterminalstr(chart.label(j)) + "[" + str(j) + "]" for j in [edge[1], edge[2]] if j != 0]))
                    else:
                        print("\t", disco_grammar.nonterminalstr(chart.label(i)) + "[" + str(i) + "]", "->", inp[edge])
        print(chart.getEdgeForItem(root, 0))
        # print(lazykbest(chart, 5))


        manager = PyDerivationManager(grammar)
        manager.convert_chart_to_hypergraph(chart, disco_grammar, debug=True)


        file = tempfile.mktemp()
        print(file)
        manager.serialize(bytes(file, encoding="utf-8"))

        gi = PyGrammarInfo(grammar, manager.get_nonterminal_map())
        sm = PyStorageManager()
        la = build_PyLatentAnnotation_initial(grammar, gi, sm)

        vec = py_edge_weight_projection(la, manager, variational=True)
        print(vec)
        self.assertEqual([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 1.0], vec)

        vec = py_edge_weight_projection(la, manager, variational=False)
        print(vec)
        self.assertEqual([1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.0], vec)

        der = manager.viterbi_derivation(0, vec, grammar)
        print(der)

        # print(disco_grammar.rulenos)
        # print(disco_grammar.numrules)
        # print(disco_grammar.lexicalbylhs)
        # print(disco_grammar.lexicalbyword)
        # print(disco_grammar.lexicalbynum)
        # print(disco_grammar.origrules, type(disco_grammar.origrules))
        # print(disco_grammar.numbinary)
        # print(disco_grammar.numunary)
        # print(disco_grammar.toid)
        # print(disco_grammar.tolabel)
        # print(disco_grammar.bitpar)
        # striplabelre = re.compile(r'-\d+$')
        # msg = disco_grammar.getmapping(None, None)
        # disco_grammar.getrulemapping(disco_grammar, striplabelre)
        # mapping = disco_grammar.rulemapping
        # print(mapping)
        # for idx, group in enumerate(mapping):
        #     print("Index", idx)
        #     for elem in group:
        #         print(grammar.rule_index(elem))

        # for _, item in zip(range(20), chart.parseforest):
        #     edge = chart.parseforest[item]
        #     print(item, item.binrepr(), item.__repr__(), item.lexidx())
        #     print(type(edge))


if __name__ == '__main__':
    unittest.main()
