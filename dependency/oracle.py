def compute_oracle_tree(trees, gold_tree):
    """
    :param trees: 
    :type trees: list[HybridTree]
    :type gold_tree: HybridTree
    :return: 
    :rtype: 
    """

    gold_labels = {}
    gold_heads = {}

    for position, id in enumerate(gold_tree.id_yield()):
        parent_id = gold_tree.parent(id)
        gold_labels[position] = gold_tree.node_token(id).deprel()
        if parent_id is None:
            assert id in gold_tree.root
            gold_heads[position] = 0
        else:
            gold_heads[position] = gold_tree.id_yield().index(parent_id) + 1

    best_hypothesis = None
    correct_labeled_attachments = -1
    correct_unlabeled_attachments = -1
    correct_labels = -1

    for tree in trees:
        las, uas, lac = 0, 0, 0
        for position, id in enumerate(tree.id_yield()):
            parent_id = tree.parent(id)
            if parent_id is None:
                assert id in tree.root
                head = 0
            else:
                head = tree.id_yield().index(parent_id) + 1
            label = tree.node_token(id).deprel()

            if gold_heads[position] == head:
                uas += 1
            if gold_labels[position] == label:
                lac += 1
            if gold_heads[position] == head and gold_labels[position] == label:
                las += 1
        # python uses lexicographic order on tuples !
        if (las, uas, las) > (correct_labeled_attachments, correct_unlabeled_attachments, correct_labels):
            best_hypothesis = tree
            correct_labeled_attachments = las
            correct_unlabeled_attachments = uas
            correct_labels = lac

    return best_hypothesis
