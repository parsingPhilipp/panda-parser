import unittest
from graphs.dog import *


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


if __name__ == '__main__':
    unittest.main()

def build_acyclic_dog():
    dog = DirectedOrderedGraph()
    for i in range(11):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([1, 2, 3], 'CS', 0)
    dog.add_terminal_edge([4, 5, 6], 'S', 1)
    dog.add_terminal_edge([4, 7, 6], 'S', 3)
    dog.add_terminal_edge([8, 9, 10], 'CNP', 6)
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
    dog.add_nonterminal_edge([4], [3, 5])
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
    dog.add_to_outputs(0)
    dog.add_to_outputs(3)
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