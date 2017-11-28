from __future__ import print_function
from hybridtree.constituent_tree import ConstituentTree


def check_single_child_label(tree, label="HD"):
    """
    :type tree: ConstituentTree
    :rtype: Bool

    Checks, if every node in the tree has at most one child with a particular edge label.
    For instance, one can check, that there is at most one child labeled "HD" or "SB".
    """

    def check_single_head_rec(node):
        heads = 0
        for child in tree.children(node):
            if tree.node_token(child).edge() == label:
                heads += 1
        if heads > 1:
            return False
        else:
            return all([check_single_head_rec(child) for child in tree.children(node)])

    return check_single_head_rec(tree.root[0])
