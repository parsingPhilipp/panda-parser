import unittest
import corpora.negra_parse as np
from grammar.induction.brown_clustering import BrownClustering
import pstats, cProfile
import pyximport

pyximport.install()


class BrownClusteringTest(unittest.TestCase):

    def test_large_corpus(self):

        bc = BrownClustering("large_corpus_test", 3, "test",max_vocab_size=5, optimization=True)
        print(bc.avg_mut_info)

    def test_min_corpus(self):
        bc = BrownClustering("min_corpus_test", 3, "min")
    def test_full_corpus(self):
        bc = BrownClustering("tigertext",30,"test_moving", optimization=True)

