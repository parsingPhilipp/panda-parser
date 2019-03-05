import unittest
from grammar.induction.brownclustering import BrownClustering
class BrownClusteringTest(unittest.TestCase):
    def test_large_corpus(self):
        corpus = [["ich", "habe", "heute", "geburtstag"], ["heute", "gab", "es", "grüne", "tomaten"],
         ["heute", "habe", "ich", "grüne", "tomaten", "gegessen"]]
        bc = BrownClustering(corpus, 3, "test")
        #assert(abs(bc.avg_mut_info)<0.000000000000001)
    def test_min_corpus(self):
        corpus = [["0", "1", "1", "2","3","4"]]
        bc = BrownClustering(corpus, 3, "min")
        #assert (abs(bc.avg_mut_info) < 0.000000000000001)