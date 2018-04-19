import unittest
import corpora.negra_parse as np
from hybridtree.general_hybrid_tree import HybridTree


class MyTestCase(unittest.TestCase):
    def test_negra_to_dag_parsing(self):
        pass
        names = list(map(str, [26954]))
        corpus = np.sentence_names_to_hybridtrees(names, "res/tiger/tiger_s26954.export", secedge=True)
        corpus2 = np.sentence_names_to_hybridtrees(names, "res/tiger/tiger_s26954_bin.export", secedge=True)
        dag = corpus[0]
        print(dag)
        assert isinstance(dag, HybridTree)
        self.assertEqual(8, len(dag.token_yield()))
        for token in dag.token_yield():
            print(token.form() + '/' + token.pos(), end=' ')
        print()

        dag_bin = corpus2[0]
        print(dag_bin)

        for token in dag_bin.token_yield():
            print(token.form() + '/' + token.pos(), end=' ')
        print()
        self.assertEqual(8, len(dag_bin.token_yield()))



if __name__ == '__main__':
    unittest.main()
