# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
from discodop.tree import DiscTree, Tree, DrawTree, ParentedTree
from discodop.eval import TreePairResult
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.monadic_tokens import ConstituentCategory
from corpora.tiger_parse import sentence_names_to_hybridtrees
from constituent.discodop_adapter import TreeComparator, convert_tree, build_param
import sys

# FIELDS = tuple(range(6))
# WORD, LEMMA, TAG, MORPH, FUNC, PARENT = FIELDS

if sys.version_info < (3,):
    reload(sys)
    sys.setdefaultencoding('utf8')


def main():
    # train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train/train.German.gold.xml'
    # corpus = sentence_names_to_hybridtrees(["s" + str(i) for i in range(1, 10)], file_name=train_path, hold=False)

    train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'
    names = ["s" + str(i) for i in range(40675, 40700)]
    names = ['s40564']
    corpus = sentence_names_to_hybridtrees(names, file_name=train_path, hold=False)

    cp = TreeComparator()

    tree_sys = ConstituentTree()
    tree_sys.add_node('0', ConstituentCategory('PN'))
    tree_sys.add_node('1', corpus[0].token_yield()[0], order=True)
    tree_sys.add_punct("3", '$.', '.')
    tree_sys.add_to_root('0')
    tree_sys.add_child('0', '1')

    param = build_param()

    for i, hybridtree in enumerate(corpus):
        print(i)

        # discotree = convert_tree(hybridtree)
        tree, sent = convert_tree(hybridtree)
        tree2, sent2 = convert_tree(tree_sys)

        if i == 11:
            pass

        # print(discotree)

        # print(discotree.draw())

        # print(DrawTree(discotree, discotree.sent))
        print(DrawTree(tree, sent))

        print(' '.join(map(lambda x: x.form(), hybridtree.full_token_yield())))

        print(DrawTree(tree2, sent2))

        print(tree[::-1])

        print('POS', tree.pos())

        result = TreePairResult(i, tree, sent, tree2, sent2, param)
        print(result.scores())

        print("Comparator: ", cp.compare_hybridtrees(hybridtree, hybridtree))



if __name__ == "__main__":
    main()