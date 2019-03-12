import unittest
import corpora.negra_parse as np
from grammar.induction.brown_clustering import BrownClustering


class BrownClusteringTest(unittest.TestCase):

    def test_large_corpus(self):
        corpus = [["ich", "habe", "heute", "geburtstag"], ["heute", "gab", "es", "grüne", "tomaten"], ["heute", "habe", "ich", "grüne", "tomaten", "gegessen"]]
        bc = BrownClustering(corpus, 3, "test")

    def test_min_corpus(self):
        corpus = [["0", "1", "1", "2", "3", "4"]]
        bc = BrownClustering(corpus, 3, "min")
    def test_full_corpus(self):
        file = "/home/mango/Dokumente/Parsing/panda-parser/res/TIGER/tiger21/tigertraindev_root_attach.export"
        trees = np.sentence_names_to_hybridtrees([str(x) for x in range(1000) if x % 10 > 1], file,
                                                  disconnect_punctuation=False)
        corpus = list()
        for tree in trees:
            sentence = []
            for token in tree.token_yield():
                sentence.append(token.form().lower())
            corpus.append(sentence)
        bc = BrownClustering(corpus,30,"tiger")