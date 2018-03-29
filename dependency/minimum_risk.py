from collections import defaultdict
import copy
from hybridtree.dependency_tree import HybridTree


def compute_minimum_risk_tree(trees, probabilities):
    """
    :type trees: list[HybridTree] 
    :type probabilities: list[float]
    :rtype: HybridTree
    """
    heads = defaultdict(lambda: defaultdict(lambda : 0.0))
    for tree, probability in zip(trees, probabilities):
        assert isinstance(tree, HybridTree)
        for position, id in enumerate(tree.id_yield()):
            parent_id = tree.parent(id)
            if parent_id is None:
                assert id in tree.root
                head = 0
            else:
                head = tree.id_yield().index(parent_id) + 1
            heads[position + 1][head, tree.node_token(id).deprel()] += probability

    cleaned_tokens = copy.deepcopy(trees[0].token_yield())
    min_risk_tree = HybridTree()
    n = len(cleaned_tokens)
    for position, token in enumerate(cleaned_tokens):
        min_risk_tree.add_node(position + 1, token, order=True)

    for position in range(1, n + 1):
        best_head = 0, '_'
        best_prob = 0.0
        for head in heads[position]:
            # print position, head, heads[position][head]
            if heads[position][head] > best_prob:
                best_head = head
                best_prob = heads[position][head]
        min_risk_tree.node_token(position).set_edge_label(best_head[1])
        if best_head[0] == 0:
            min_risk_tree.add_to_root(position)
        else:
            assert best_head[0] in min_risk_tree.nodes()
            min_risk_tree.add_child(best_head[0], position)

    return min_risk_tree


__all__ = ["compute_minimum_risk_tree"]
