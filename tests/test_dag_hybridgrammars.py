import unittest
import corpora.negra_parse as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        pass
        names = list(map(str, [26952]))
        corpus = np.sentence_names_to_hybridtrees(names, "res/tiger/tiger_full_with_sec_edges.export")
        dag = corpus[0]
        print(dag)

if __name__ == '__main__':
    unittest.main()
