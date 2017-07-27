from __future__ import print_function
import unittest
from graphs.dog import *
from graphs.graph_decomposition import *
from corpora.tiger_parse import sentence_name_to_deep_syntax_graph, sentence_names_to_deep_syntax_graphs
from hybridtree.monadic_tokens import ConstituentTerminal
from parser.naive.parsing import LCFRS_parser
from parser.cpp_cfg_parser.parser_wrapper import CFGParser
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory, fanout_limited_partitioning
from grammar.induction.terminal_labeling import PosTerminals


class MyTestCase(unittest.TestCase):
    def test_acyclic_dog(self):
        for dog in [build_acyclic_dog(), dog_s1(), dog_s2(), dog_s3(), dog_s11(), dog_s12(), dog_s13(), dog_s131()]:
            self.assertFalse(dog.cyclic())
            self.assertTrue(dog.output_connected())

    def test_cyclic_dog(self):
        dog = dog_se()
        self.assertTrue(dog.cyclic())
        self.assertTrue(dog.output_connected())

    def test_substitution(self):
        dog_13 = dog_s13()
        dog_13.replace_by(0, dog_s131())
        dog_13.replace_by(1, dog_s132())
        dog_13.replace_by(2, dog_s133())

        dog_1 = dog_s1()
        dog_1.replace_by(0, dog_s11())
        dog_1.replace_by(1, dog_s12())

        dog_3 = dog_s3()
        dog_3.replace_by(0, dog_s32())
        dog_3.replace_by(1, dog_13)

        dog_host = dog_se()
        dog_host.replace_by(0, dog_1)
        dog_host.replace_by(1, dog_s2())
        dog_host.replace_by(2, dog_3)

        self.assertEqual(build_acyclic_dog(), dog_host)

    def test_eq(self):
        self.assertEqual(build_acyclic_dog(), build_acyclic_dog())
        self.assertEqual(dog_s132(), dog_s132())
        self.assertNotEqual(build_acyclic_dog(), build_acyclic_dog_permuted())
        self.assertNotEqual(dog_s1(), dog_s3())
        self.assertEqual(dog_s1() == dog_s13(), dog_s13() == dog_s1())
        self.assertNotEqual(dog_s13(), dog_s2())
        self.assertNotEqual(dog_s1(), dog_s2())
        self.assertNotEqual(dog_s131(), dog_s132())

    def test_order(self):
        self.assertListEqual(build_acyclic_dog().ordered_nodes(), [0, 1, 4, 5, 6, 8, 9, 10, 2, 3, 7])
        self.assertListEqual(build_acyclic_dog_permuted().ordered_nodes(), [0, 1, 4, 5, 6, 8, 9, 10, 2, 3, 7])
        self.assertListEqual(dog_se().ordered_nodes(), [0, 1, 5, 4, 2, 3])
        dog = dog_se()
        dog.compress_node_names()
        self.assertListEqual(dog.ordered_nodes(), [i for i in range(6)])

    def test_top_bottom(self):
        dog = build_acyclic_dog()

        nodes = [1, 4, 5]
        self.assertListEqual(dog.top(nodes), [1, 4])
        self.assertListEqual(dog.bottom(nodes), [6])

        nodes = [3, 7, 6, 8, 9, 10]
        self.assertListEqual(dog.top(nodes), [6, 3])
        self.assertListEqual(dog.bottom(nodes), [4])

        nodes = [i for i in range(11)]
        self.assertListEqual(dog.top(nodes), [0])
        self.assertListEqual(dog.bottom(nodes), [])

        nodes = [2, 4, 5, 7, 8, 9, 10]
        for node in nodes:
            self.assertListEqual(dog.top([node]), [node])
            self.assertListEqual(dog.bottom([node]), [])

    def test_extraction(self):
        dog = build_acyclic_dog()

        self.assertEqual(dog_se(), dog.extract_dog([i for i in range(11)], [[1, 4, 5], [2], [3, 6, 7, 8, 9, 10]]))

        for (lab, i) in [
            ('und', 2), ('Sie', 4), ('entwickelt', 5), ('druckt', 7),
            ('Verpackungen', 8), ('und', 9), ('Etiketten', 10)
        ]:
            self.assertEqual(build_terminal_dog(lab), dog.extract_dog([i], []))

        self.assertEqual(dog_s1(), dog.extract_dog([1, 4, 5], [[4], [5]]))
        self.assertEqual(dog_s3(), dog.extract_dog([3, 7, 6, 8, 9, 10], [[7], [6, 8, 9, 10]]))

    def test_copy_and_substitution_order_invariance(self):
        dog = dog_s13()
        dog.replace_by(0, dog_s131())
        dog_2 = deepcopy(dog)
        self.assertEqual(dog, dog_2)
        dog.replace_by(1, dog_s132())
        self.assertNotEqual(dog, dog_2)
        dog_2.replace_by(2, dog_s133())
        dog_2.replace_by(1, dog_s132())
        self.assertNotEqual(dog, dog_2)
        dog.replace_by(2, dog_s133())
        self.assertEqual(dog, dog_2)

    def test_primary(self):
        dog = build_acyclic_dog()
        self.assertTrue(dog.primary_is_tree())
        dog2 = build_acyclic_dog_permuted()
        self.assertFalse(dog2.primary_is_tree())

    def test_upward_closure(self):
        dog = build_acyclic_dog()
        for (lab, i) in [
            ('und', 2), ('Sie', 4), ('entwickelt', 5), ('druckt', 7),
            ('Verpackungen', 8), ('und', 9), ('Etiketten', 10)
        ]:
            self.assertListEqual(upward_closure(dog, [i]), [i])

        self.assertSetEqual(set(upward_closure(dog, [4, 5])), {1, 4, 5})
        self.assertSetEqual(set(upward_closure(dog, [8, 9, 10])), {6, 8, 9, 10})
        self.assertSetEqual(set(upward_closure(dog, [4, 5, 2, 7, 8, 9, 10])), set([i for i in range(11)]))

    def test_dsg(self):
        dsg = build_dsg()
        rec_part = dsg.recursive_partitioning()
        self.assertEqual(rec_part, ({0, 1, 2, 3, 4, 5, 6}, [({0, 1}, [({0}, []), ({1}, [])]), ({2}, []), ({3, 4, 5, 6}, [({3}, []), ({4, 5, 6}, [({4}, []), ({5}, []), ({6}, [])])])]))
        dcmp = compute_decomposition(dsg, rec_part)
        self.assertEqual(dcmp, ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [([1, 4, 5], [([4], []), ([5], [])]), ([2], []), ([3, 6, 7, 8, 9, 10], [([7], []), ([6, 8, 9, 10], [([8], []), ([9], []), ([10], [])])])]))
        self.__structurally_equal(rec_part, dcmp)

        grammar = induce_grammar_from(dsg, rec_part, dcmp, terminal_labeling=str)
        # print(grammar)

        for nont, label in zip(["[4]", "[5]", "[2]", "[7]", "[8]", "[9]", "[10]"],
                ["Sie", "entwickelt", "und", "druckt", "Verpackungen", "und", "Etiketten"]):
            for rule in grammar.lhs_nont_to_rules(nont):
                self.assertEqual(rule.dcp()[0], build_terminal_dog(label))

        for nont, graph in zip(["[1, 4, 5]", "[6, 8, 9, 10]", "[3, 6, 7, 8, 9, 10]", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"],
                                [dog_s1(), dog_s13(), dog_s3(), dog_se()]):
            for rule in grammar.lhs_nont_to_rules(nont):
                self.assertEqual(rule.dcp()[0], graph)

        parser = LCFRS_parser(grammar)
        parser.set_input(dsg.sentence)  # ["Sie", "entwickelt", "und", "druckt", "Verpackungen", "und", "Etiketten"]
        parser.parse()
        self.assertTrue(parser.recognized())

        derivation = parser.best_derivation_tree()
        self.assertNotEqual(derivation, None)

        dog, sync = dog_evaluation(derivation)
        self.assertEqual(dog, dsg.dog)

        sync_list = [(key, sync[key]) for key in sync]
        self.assertEqual(len(sync_list), len(dsg.sentence))
        sync_list.sort(lambda x, y: x[0] < y[0])
        sync_list = map(lambda x: x[1], sync_list)
        # print(dog)
        # print(sync)
        # print(sync_list)

        morphism, _ = dsg.dog.compute_isomorphism(dog)

        for i in range(len(dsg.sentence)):
            self.assertListEqual(map(lambda x: morphism[x], dsg.get_graph_position(i)), sync_list[i])

    def test_induction_with_labeling_strategies(self):
        dsg = build_dsg()
        rec_part_strategy = the_recursive_partitioning_factory().getPartitioning('right-branching')[0]
        rec_part = rec_part_strategy(dsg)
        dcmp = compute_decomposition(dsg, rec_part)

        grammar = induce_grammar_from(dsg, rec_part, dcmp, labeling=simple_labeling, terminal_labeling=str)
        print(grammar)

        parser = LCFRS_parser(grammar)
        parser.set_input(dsg.sentence)  # ["Sie", "entwickelt", "und", "druckt", "Verpackungen", "und", "Etiketten"]
        parser.parse()
        self.assertTrue(parser.recognized())

        derivation = parser.best_derivation_tree()
        self.assertNotEqual(derivation, None)

        dog, sync = dog_evaluation(derivation)
        self.assertEqual(dog, dsg.dog)

        sync_list = [(key, sync[key]) for key in sync]
        self.assertEqual(len(sync_list), len(dsg.sentence))
        sync_list.sort(lambda x, y: x[0] < y[0])
        sync_list = map(lambda x: x[1], sync_list)
        # print(dog)
        # print(sync)
        # print(sync_list)

        morphism, _ = dsg.dog.compute_isomorphism(dog)

        for i in range(len(dsg.sentence)):
            self.assertListEqual(map(lambda x: morphism[x], dsg.get_graph_position(i)), sync_list[i])

    def __structurally_equal(self, rec_part, decomp):
        self.assertEqual(len(rec_part[1]), len(decomp[1]))
        for rec_part_child, decomp_child in zip(rec_part[1], decomp[1]):
            self.__structurally_equal(rec_part_child, decomp_child)

    def test_tiger_parse_to_dsg(self):
        dsg = sentence_name_to_deep_syntax_graph("s26954", "res/tiger/tiger_s26954.xml")

        f = lambda token: token.form() if isinstance(token, ConstituentTerminal) else token
        dsg.dog.project_labels(f)

        print(map(str, dsg.sentence))
        print(dsg.label)
        print(dsg.dog)
        print([dsg.get_graph_position(i) for i in range(len(dsg.sentence))])

        # strip "VROOT"
        sub_dog = dsg.dog.extract_dog([i for i in range(11)], [])

        self.assertEqual(sub_dog, build_acyclic_dog())

    def test_induction_from_corpus_tree(self):
        dsg = sentence_name_to_deep_syntax_graph("s26954", "res/tiger/tiger_s26954.xml")

        def label_edge(edge):
            if isinstance(edge.label, ConstituentTerminal):
                return edge.label.pos()
            else:
                return edge.label
        labeling = lambda nodes, dsg: simple_labeling(nodes, dsg, label_edge)

        term_labeling_token = PosTerminals()

        def term_labeling(token):
            if isinstance(token, ConstituentTerminal):
                return term_labeling_token.token_label(token)
            else:
                return token

        rec_part_strategy = the_recursive_partitioning_factory().getPartitioning('cfg')[0]
        rec_part = rec_part_strategy(dsg)
        dcmp = compute_decomposition(dsg, rec_part)

        grammar = induce_grammar_from(dsg, rec_part, dcmp, labeling=labeling, terminal_labeling=term_labeling)

        print(grammar)

        parser = LCFRS_parser(grammar)
        parser.set_input(term_labeling_token.prepare_parser_input(dsg.sentence))
        parser.parse()
        self.assertTrue(parser.recognized())

        derivation = parser.best_derivation_tree()
        self.assertNotEqual(derivation, None)

    def test_induction_on_a_corpus(self):
        start = 1
        stop = 50
        path = "res/tiger/tiger_release_aug07.corrected.16012013.utf8.xml"
        # path = "res/tiger/tiger_8000.xml"
        exclude = []
        dsgs = sentence_names_to_deep_syntax_graphs(
            ['s' + str(i) for i in range(start, stop + 1) if i not in exclude]
            , path
            , hold=False)

        rec_part_strategy = the_recursive_partitioning_factory().getPartitioning('cfg')[0]

        def label_edge(edge):
            if isinstance(edge.label, ConstituentTerminal):
                return edge.label.pos()
            else:
                return edge.label
        nonterminal_labeling = lambda nodes, dsg: simple_labeling(nodes, dsg, label_edge)

        term_labeling_token = PosTerminals()
        def term_labeling(token):
            if isinstance(token, ConstituentTerminal):
                return term_labeling_token.token_label(token)
            else:
                return token

        grammar = induction_on_a_corpus(dsgs, rec_part_strategy, nonterminal_labeling, term_labeling)
        grammar.make_proper()

        parser = CFGParser(grammar)

        for dsg in dsgs:
            parser.set_input(term_labeling_token.prepare_parser_input(dsg.sentence))
            parser.parse()
            self.assertTrue(parser.recognized())
            derivation = parser.best_derivation_tree()
            dog, sync = dog_evaluation(derivation)
            dsg2 = DeepSyntaxGraph(dsg.sentence, dog, sync)

            f = lambda token: token.pos() if isinstance(token, ConstituentTerminal) else token
            dsg.dog.project_labels(f)
            parser.clear()

            # print('dsg: ', dsg.dog, '\n', [dsg.get_graph_position(i) for i in range(len(dsg.sentence))], '\n\n parsed: ', dsg2.dog, '\n', [dsg2.get_graph_position(i+1) for i in range(len(dsg2.sentence))])
            # print()

if __name__ == '__main__':
    unittest.main()

def build_acyclic_dog():
    dog = DirectedOrderedGraph()
    for i in range(11):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([(1, 'p'), (2, 'p'), (3, 'p')], 'CS', 0)\
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    dog.add_terminal_edge([(4, 'p'), (5, 'p'), 6], 'S', 1)\
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_terminal_edge([4, (7, 'p'), (6, 'p')], 'S', 3) \
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_terminal_edge([(8, 'p'), (9, 'p'), (10, 'p')], 'CNP', 6) \
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    for (lab, i) in [
        ('und', 2), ('Sie', 4), ('entwickelt', 5), ('druckt', 7),
        ('Verpackungen', 8), ('und', 9), ('Etiketten', 10)
    ]:
        dog.add_terminal_edge([], lab, i)

    return dog

def build_acyclic_dog_permuted():
    dog = DirectedOrderedGraph()
    for i in range(11):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([1, 2, 3], 'CS', 0)\
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    dog.add_terminal_edge([4, 5, 6], 'S', 1) \
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_terminal_edge([4, 7, 6], 'S', 3) \
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_terminal_edge([8, 9, 10], 'CNP', 6)\
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    for (lab, i) in [
        ('und', 2), ('Sie', 4), ('entwickelt', 7), ('druckt', 5),
        ('Verpackungen', 8), ('und', 9), ('Etiketten', 10)
    ]:
        dog.add_terminal_edge([], lab, i)

    return dog

def build_terminal_dog(terminal):
    dog = DirectedOrderedGraph()
    dog.add_node(0)
    dog.add_to_outputs(0)
    dog.add_terminal_edge([], terminal, 0)
    return dog

def dog_se():
    dog = DirectedOrderedGraph()
    for i in range(6):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([1, 2, 3], 'CS', 0)\
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    dog.add_nonterminal_edge([5], [1, 4])
    dog.add_nonterminal_edge([], [2])
    dog.add_nonterminal_edge([4], [5, 3])
    return dog


def dog_s1():
    dog = DirectedOrderedGraph()
    for i in range(4):
        dog.add_node(i)
    dog.add_to_outputs(0)
    dog.add_to_outputs(1)
    dog.add_to_inputs(3)

    dog.add_terminal_edge([1, 2, 3], 'S', 0)\
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_nonterminal_edge([], [1])
    dog.add_nonterminal_edge([], [2])

    return dog

def dog_s2():
    return build_terminal_dog('und')

def dog_s3():
    dog = DirectedOrderedGraph()
    for i in range(4):
        dog.add_node(i)
    dog.add_to_outputs(3)
    dog.add_to_outputs(0)
    dog.add_to_inputs(1)

    dog.add_terminal_edge([1, 2, 3], 'S', 0)\
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_nonterminal_edge([], [2])
    dog.add_nonterminal_edge([], [3])

    return dog

def dog_s11():
    return build_terminal_dog('Sie')

def dog_s12():
    return build_terminal_dog('entwickelt')

def dog_s32():
    return build_terminal_dog('druckt')

def dog_s13():
    dog = DirectedOrderedGraph()
    for i in range(4):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([1, 2, 3], 'CNP', 0)\
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    for i in range(1, 4):
        dog.add_nonterminal_edge([], [i])
    return dog

def dog_s131():
    return build_terminal_dog('Verpackungen')

def dog_s132():
    return build_terminal_dog('und')

def dog_s133():
    return build_terminal_dog('Etiketten')

def build_dsg():
    dog = build_acyclic_dog()
    sentence = ["Sie", "entwickelt", "und", "druckt", "Verpackungen", "und", "Etiketten"]
    synchronization = [[4], [5], [2], [7], [8], [9], [10]]
    return DeepSyntaxGraph(sentence, dog, synchronization)
