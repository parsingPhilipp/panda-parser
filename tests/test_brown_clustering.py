import unittest
from grammar.induction.brown_clustering import BrownClustering


class BrownClusteringTest(unittest.TestCase):

    def test_large_corpus(self):
        corpus = [["ich", "habe", "heute", "geburtstag"], ["heute", "gab", "es", "grüne", "tomaten"], ["heute", "habe", "ich", "grüne", "tomaten", "gegessen"]]
        bc = BrownClustering(corpus, 3, "test")

    def test_min_corpus(self):
        corpus = [["0", "1", "1", "2", "3", "4"]]
        bc = BrownClustering(corpus, 3, "min")
