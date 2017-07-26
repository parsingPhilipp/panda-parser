from __future__ import print_function
import unittest
from graphs.dog import *
from graphs.graph_decomposition import *


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

        self.assertSetEqual(set(upward_closure(dog, [4, 5])), set([1, 4, 5]))
        self.assertSetEqual(set(upward_closure(dog, [8, 9, 10])), set([6, 8, 9, 10]))
        self.assertSetEqual(set(upward_closure(dog, [4, 5, 2, 7, 8, 9, 10])), set([i for i in range(11)]))

    def test_dsg(self):
        dsg = build_dsg()
        rec_part = dsg.extract_recursive_partitioning()
        self.assertEqual(rec_part, ([0, 1, 2, 3, 4, 5, 6], [([0, 1], [([0], []), ([1], [])]), ([2], []), ([3, 4, 5, 6], [([3], []), ([4, 5, 6], [([4], []), ([5], []), ([6], [])])])]))
        dcmp = compute_decomposition(dsg, rec_part)
        self.assertEqual(dcmp, ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [([1, 4, 5], [([4], []), ([5], [])]), ([2], []), ([3, 6, 7, 8, 9, 10], [([7], []), ([6, 8, 9, 10], [([8], []), ([9], []), ([10], [])])])]))
        self.__structurally_equal(rec_part, dcmp)

        grammar = induce_grammar_from(dsg, rec_part, dcmp, labeling=str)
        print(grammar)

        for nont, label in zip(["[4]", "[5]", "[2]", "[7]", "[8]", "[9]", "[10]"],
                ["Sie", "entwickelt", "und", "druckt", "Verpackungen", "und", "Etiketten"]):
            for rule in grammar.lhs_nont_to_rules(nont):
                self.assertEqual(rule.dcp()[0], build_terminal_dog(label))

        for nont, graph in zip(["[1, 4, 5]", "[6, 8, 9, 10]", "[3, 6, 7, 8, 9, 10]", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"],
                                [dog_s1(), dog_s13(), dog_s3(), dog_se()]):
            for rule in grammar.lhs_nont_to_rules(nont):
                self.assertEqual(rule.dcp()[0], graph)

    def __structurally_equal(self, rec_part, decomp):
        self.assertEqual(len(rec_part[1]), len(decomp[1]))
        for rec_part_child, decomp_child in zip(rec_part[1], decomp[1]):
            self.__structurally_equal(rec_part_child, decomp_child)


if __name__ == '__main__':
    unittest.main()

def build_acyclic_dog():
    dog = DirectedOrderedGraph()
    for i in range(11):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([(1, 'p'), (2, 'p'), (3, 'p')], 'CS', 0)
    dog.add_terminal_edge([(4, 'p'), (5, 'p'), 6], 'S', 1)
    dog.add_terminal_edge([4, (7, 'p'), (6, 'p')], 'S', 3)
    dog.add_terminal_edge([(8, 'p'), (9, 'p'), (10, 'p')], 'CNP', 6)
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

    dog.add_terminal_edge([1, 2, 3], 'CS', 0)
    dog.add_terminal_edge([4, 5, 6], 'S', 1)
    dog.add_terminal_edge([4, 7, 6], 'S', 3)
    dog.add_terminal_edge([8, 9, 10], 'CNP', 6)
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

    dog.add_terminal_edge([1, 2, 3], 'CS', 0)
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

    dog.add_terminal_edge([1, 2, 3], 'S', 0)
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

    dog.add_terminal_edge([1, 2, 3], 'S', 0)
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

    dog.add_terminal_edge([1, 2, 3], 'CNP', 0)
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