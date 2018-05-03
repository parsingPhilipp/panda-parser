import unittest
from corpora.conll_parse import parse_conll_corpus
from hybridtree.dependency_tree import disconnect_punctuation


class TestDependencyTree(unittest.TestCase):
    def test_wsj(self):
        corpus = "res/wsj_dependency/24.conll"
        trees = parse_conll_corpus(corpus, False, 5000)
        trees = disconnect_punctuation(trees)

        x = 0
        for tree in trees:
            x += 1

        self.assertEqual(1346, x)

    def test_tiger(self):
        corpus = "res/dependency_conll/german/tiger/test/german_tiger_test.conll"
        trees = parse_conll_corpus(corpus, False, 5000)
        trees = disconnect_punctuation(trees)

        x = 0
        for tree in trees:
            x += 1

        self.assertEqual(357, x)


if __name__ == '__main__':
    unittest.main()
