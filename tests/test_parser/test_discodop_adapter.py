from __future__ import print_function, unicode_literals
import unittest
from grammar.lcfrs import LCFRS, LCFRS_var, LCFRS_lhs
from pprint import pprint
from parser.discodop_parser.grammar_adapter import transform_grammar
from discodop.plcfrs import parse
from discodop.containers import Grammar


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

        grammar.make_proper()
        return grammar

    def test_something(self):
        grammar = self.build_grammar()

        for r in transform_grammar(grammar):
            pprint(r)

        rule_list = list(transform_grammar(grammar))
        pprint(rule_list)
        disco_grammar = Grammar(rule_list, start=grammar.start())
        print(disco_grammar)

        inp = ["a"] * 3
        chart, msg = parse(["a"] * 3, disco_grammar)
        chart.filter()
        print(msg)
        print(disco_grammar.nonterminals)
        print(type(disco_grammar.nonterminals))

        print(chart)
        print(help(chart))

        root = chart.root()
        print("root", root, type(root))
        print(chart.indices(root))
        print(chart.itemstr(root))
        print(chart.stats())
        print("root label", chart.label(root))
        print(root, chart.itemid1(chart.label(root), chart.indices(root)))
        for i in range(1, chart.numitems()):
            print(i, chart.label(i), chart.indices(i), chart.numedges(i))
            if True or len(chart.indices(i)) > 1:
                for edge_num in range(chart.numedges(i)):
                    edge = chart.getEdgeForItem(i, edge_num)
                    if isinstance(edge, tuple):
                        print("\t", disco_grammar.nonterminalstr(chart.label(i)), "->", ' '.join([disco_grammar.nonterminalstr(chart.label(j)) + "[" + str(j) + "]" for j in [edge[1], edge[2]] if j != 0]))
                    else:
                        print("\t", disco_grammar.nonterminalstr(chart.label(i)), "->", inp[edge])
        print(chart.getEdgeForItem(root, 0))

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
