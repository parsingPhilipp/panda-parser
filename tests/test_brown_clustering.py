import unittest
import corpora.negra_parse as np
from grammar.induction.brown_clustering import BrownClustering
import pstats, cProfile
import pyximport

pyximport.install()
class BrownClusteringTest(unittest.TestCase):

    def test_large_corpus(self):
        corpus = [["ich", "habe", "heute", "geburtstag"], ["heute", "gab", "es", "grüne", "tomaten"], ["heute", "habe", "ich", "grüne", "tomaten", "gegessen"]]
        bc = BrownClustering(corpus, 3, "test",max_vocab_size=5, optimization=True)
        print(bc.avg_mut_info)

    def test_min_corpus(self):
        corpus = [["0", "1", "1", "2", "3", "4"]]
        bc = BrownClustering(corpus, 3, "min")
    def test_full_corpus(self):
        file = "/home/mango/Dokumente/Parsing/panda-parser/res/TIGER/tiger21/tigertext.raw"
        textfile = open(file)
        lines = textfile.read().splitlines()
        textfile.close()
        corpus = []
        numsentences = 800
        for x in range(numsentences):
            line = str.split(lines[x])
            lower_line = [x.lower() for x in line]
            corpus.append(lower_line)
        bc = BrownClustering(corpus,30,"tiger",optimization=True)

