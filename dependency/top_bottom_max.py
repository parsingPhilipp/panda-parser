def top_max(tree, id_set):
    """
    :rtype: [[str]]
    :param tree: HybridTree
    :param id_set: list of string
    :return: list of list of string
    Compute list of node ids that delimit id_set from the top
    and group maximal subsets of neighbouring nodes together.
    """
    return maximize(tree, top(tree, id_set))


def bottom_max(tree, id_set):
    """
    :rtype: [[str]]
    :param tree: HybridTree
    :param id_set: list of string
    :return: list of list of string
    Compute list of node ids that delimit id_set from the bottom.
    and group maximal subsets of neighbouring nodes together.
    """
    return maximize(tree, bottom(tree, id_set))


def top(tree, id_set):
    """
    :rtype: [[str]]
    :param tree: HybridTree
    :param id_set: list of string  (node ids)
    :return: list of string  (node ids)
    Compute list of node ids that delimit id_set from the top.
    """
    top_nodes = [id for id in id_set if tree.parent(id) not in id_set]
    return top_nodes


def bottom(tree, id_set):
    """
    :rtype: [[str]]
    :param tree: list of node ids that delimit id_set from the bottom.
    :param id_set: list of string  (node ids)
    :return: list of string  (node ids)
    Compute list of node ids that delimit id_set from the bottom.
    """
    bottom_nodes = [id for id in tree.nodes()
                    if tree.parent(id) in id_set and id not in id_set]
    return bottom_nodes


def maximize(tree, id_set):
    """
    :param tree: HybridTree
    :param id_set: list of string
    :return: list of list of string
    Group maximal subsets of neighbouring nodes together.
    """
    nodes = id_set[:]
    max_list = []
    while len(nodes) > 0:
        id = nodes[0]

        # Assume that the following two lists contain tree nodes, that are
        # siblings, ordered from left to right.
        all_siblings = tree.siblings(id)
        present_siblings = [id for id in all_siblings if id in nodes]

        nodes = [id for id in nodes if id not in present_siblings]

        while len(present_siblings) > 0:
            i = all_siblings.index(present_siblings[0])
            j = 1
            while j < len(present_siblings) and all_siblings[i + j] == present_siblings[j]:
                j += 1
            max_list += [present_siblings[:j]]
            present_siblings = present_siblings[j:]

    return max_list


__all__ = ["top", "bottom", "top_max", "bottom_max"]
