from __future__ import print_function
import unittest
from parser.trace_manager.sm_trainer import build_PyLatentAnnotation
from parser.trace_manager.sm_trainer_util import PyStorageManager, PyGrammarInfo
from grammar.lcfrs import LCFRS, LCFRS_lhs, LCFRS_var
from util.enumerator import Enumerator
from parser.naive.parsing import LCFRS_parser
from random import random


class TestProjection(unittest.TestCase):
    def test_asymmetric(self):
        v_1 = [[0.25, 0.25, 0.25, 0.25], [0.0, 0.5], [1.0, 0.5]]
        pr_1 = [1.0, 0.25, 0.75]
        v_2 = [[0.0, 0.5, 0.5, 0.0], [0.2, 0.7], [0.8, 0.3]]
        pr_2 = [1.0, 0.45, 0.55]
        v_3 = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.7], [0.0, 0.3]]
        pr_3 = [1.0, 1.0, 0.0]
        for merge_method in [False, True]:
            for v, pr in zip([v_1, v_2, v_3], [pr_1, pr_2, pr_3]):
                pr_auto = self.__projections(v)
                # print(v, pr, pr_auto)
                self.assertEqual(len(pr_auto), len(pr))
                for x, y in zip(pr, pr_auto):
                    self.assertAlmostEqual(x, y)
                self.__test_projection(v, pr_auto, merge_method)

            for i in range(25):
                vec = self.__random_vector()
                print(vec)
                pr_auto = self.__projections(vec)
                self.__test_projection(vec, pr_auto, merge_method)

    def __random_vector(self):
        def randvec(num):
            while True:
                v = [random() for _ in range(num)]
                norm = sum(v)
                if norm > 0.0:
                    break
            return list(map(lambda x: x / norm, v))

        pa1 = list(randvec(2))
        pa2 = list(randvec(2))
        return [list(randvec(4)), pa1[0:1] + pa2[0:1], pa1[1:2] + pa2[1:2]]

    def __insides(self, vec):
        a = [vec[1][0] + vec[2][0], vec[1][1] + vec[2][1]]
        s = [vec[0][0] * a[0] * a[0] + vec[0][1] * a[0] * a[1] + vec[0][2] * a[1] * a[0] + vec[0][3] * a[1] * a[1]]
        return s, a

    def __outsides(self, vec):
        i_s, i_a = self.__insides(vec)
        o_s = [1.0]
        o_a = [o_s[0] * (vec[0][0] * i_a[0] * 2 + (vec[0][1] + vec[0][2]) * i_a[1]),
               o_s[0] * (vec[0][3] * i_a[1] * 2 + (vec[0][1] + vec[0][2]) * i_a[0])]
        return o_s, o_a

    def __projections(self, vec):
        def mult(x, y):
            return x * y

        i_s, i_a = self.__insides(vec)
        o_s, o_a = self.__outsides(vec)
        f_s = sum(map(mult, i_s, o_s))
        f_a = sum(map(mult, i_a, o_a))
        # print(i_s, i_a)
        # print(o_s, o_a)
        # print(f_s, f_a)

        p_0 = (o_s[0] * vec[0][0] * i_a[0] * i_a[0]
               + o_s[0] * (vec[0][1] + vec[0][2]) * i_a[0] * i_a[1]
               + o_s[0] * vec[0][3] * i_a[1] * i_a[1]) / f_s
        p_1 = sum(map(mult, o_a, vec[1])) / f_a
        p_2 = sum(map(mult, o_a, vec[2])) / f_a
        return [p_0, p_1, p_2]

    def __test_projection(self, split_weights, goal_weights, merge_method=False):
        grammar = LCFRS("S")
        # rule 0
        lhs = LCFRS_lhs("S")
        lhs.add_arg([LCFRS_var(0, 0), LCFRS_var(1, 0)])
        grammar.add_rule(lhs, ["A", "A"])

        # rule 1
        lhs = LCFRS_lhs("A")
        lhs.add_arg(["a"])
        grammar.add_rule(lhs, [])

        lhs = LCFRS_lhs("A")
        lhs.add_arg(["b"])
        grammar.add_rule(lhs, [], weight=2.0)


        grammar.make_proper()
        # print(grammar)

        nonterminal_map = Enumerator()
        grammarInfo = PyGrammarInfo(grammar, nonterminal_map)
        storageManager = PyStorageManager()

        la = build_PyLatentAnnotation([1, 2], [1.0], split_weights, grammarInfo, storageManager)

        # parser = LCFRS_parser(grammar)
        # parser.set_input(["a", "b"])
        # parser.parse()
        # der = parser.best_derivation_tree()

        # print(la.serialize())
        if merge_method:
            la.project_weights(grammar, grammarInfo)
        else:
            splits, _, _ = la.serialize()
            merge_sources = [[[split for split in range(0, splits[nont_idx])]]
                             for nont_idx in range(0, nonterminal_map.get_counter())]

            # print("Projecting to fine grammar LA", file=self.logger)
            coarse_la = la.project_annotation_by_merging(grammarInfo, merge_sources, debug=False)
            coarse_la.project_weights(grammar, grammarInfo)

        # print(grammar)
        for i in range(3):
            self.assertAlmostEqual(grammar.rule_index(i).weight(), goal_weights[i])


if __name__ == '__main__':
    unittest.main()
