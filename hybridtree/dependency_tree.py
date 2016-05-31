from corpora.conll_parse import is_punctuation
from hybridtree.general_hybrid_tree import HybridTree


def disconnect_punctuation(trees):
    """
    :param trees: corpus of hybrid trees
    :type trees: __generator[HybridTree]
    :return: corpus of hybrid trees
    :rtype: __generator[GeneralHybridTree]
    lazily disconnect punctuation from each hybrid tree in a corpus of hybrid trees
    """
    for tree in trees:
        tree2 = HybridTree(tree.sent_label())
        for root_id in tree.root:
            if not is_punctuation(tree.node_token(root_id).form()):
                tree2.add_to_root(root_id)
        for id in tree.full_yield():
            token = tree.node_token(id)
            if not is_punctuation(token.form()):
                parent = tree.parent(id)
                while parent and parent not in tree.root and is_punctuation(tree.node_token(parent).form()):
                    parent = tree.parent(parent)
                if parent and is_punctuation(tree.node_token(parent).form()):
                    tree2.add_to_root(id)
                else:
                    tree2.add_child(parent, id)
                tree2.add_node(id, token, True, True)
            else:
                tree2.add_node(id, token, True, False)

        if tree2:
            # basic sanity checks
            if not tree2.root \
                    and len(tree2.id_yield()) == 0 \
                    and len(tree2.nodes()) == len(tree2.full_yield()):
                # Tree consists only of punctuation
                continue
            elif not tree2.root \
                    or tree2.n_nodes() != len(tree2.id_yield()) \
                    or len(tree2.nodes()) != len(tree2.full_yield()):
                print tree

                print tree2
                print tree2.sent_label()
                print "Root:", tree2.root
                print "Nodes: ", tree2.n_nodes()
                print "Id_yield:", len(tree2.id_yield()), tree2.id_yield()
                print "Nodes: ", len(tree2.nodes())
                print "full yield: ", len(tree2.full_yield())
                raise Exception()
            yield tree2
