from __future__ import print_function

from discodop.tree import ParentedTree, DrawTree
from discodop.eval import readparam, TreePairResult
from discodop.treebank import handlefunctions

import math


def convert_tree(htree, vroot="VROOT"):
    """
    :param htree:
    :type htree: ConstituentTree
    :type vroot: str
    :return:
    :rtype: Tuple[ParentedTree, List]
    """
    nodes = {}
    for idx in htree.nodes():
        token = htree.node_token(idx)
        if token.type() == "CONSTITUENT-CATEGORY":
            nodes[idx] = ParentedTree(token.category(), [])
            nodes[idx].source = (token.category(), '--', '--', '--', token.edge(), '--')
        elif token.type() == "CONSTITUENT-TERMINAL":
            # nodes[idx] = Tree(htree.full_yield().index(idx), [])
            nodes[idx] = ParentedTree(token.pos(), [htree.full_yield().index(idx)])
            nodes[idx].source = (token.form(), '--', token.pos(), token.morph_feats(), token.edge(), '--')

    if True or len(htree.root) > 1 :
        tree = ParentedTree(vroot, [nodes[r] for r in htree.root])
    else:
        tree = nodes[htree.root[0]]

    for idx in htree.nodes():
        for c_idx in htree.children(idx):
            nodes[idx].append(nodes[c_idx])
        if htree.disconnected(idx):
            tree.append(nodes[idx])

    # handlefunctions(action='between', tree=tree)

    sent = [token.form() for token in htree.full_token_yield()]

    return tree, sent


def build_param(path='../util/proper.prm'):
    param = readparam(path if path else None)
    # param['CUTOFF_LEN'] = int(opts.get('--cutofflen', param['CUTOFF_LEN']))
    # param['DISC_ONLY'] = '--disconly' in opts
    # param['DEBUG'] = max(param['DEBUG'],
    #       '--verbose' in opts, 2 * ('--debug' in opts))
    # param['TED'] |= '--ted' in opts
    # param['LA'] |= '--la' in opts
    # param['DEP'] = '--headrules' in opts
    return param


class TreeComparator:
    def __init__(self, param_path='../util/proper.prm'):
        self.param = build_param(param_path)

    def compare_hybridtrees(self, gold, system):
        """
        :type gold: ConstituentTree
        :type system: ConstituentTree
        :return:
        :rtype:
        """
        gtree, gsent = convert_tree(gold)
        stree, ssent = convert_tree(system)
        try:
            result = TreePairResult(0, gtree, gsent, stree, ssent, self.param).scores()
            f1 = float(result['LF'])
            if math.isnan(f1):
                return 0.0
            else:
                return f1

        except (KeyError, IndexError, ValueError):
            gtree, gsent = convert_tree(gold)
            stree, ssent = convert_tree(system)
            print('gold tree:')
            print(DrawTree(gtree, gsent))
            print(gold)
            print(gold.root)

            print('system tree')
            print(DrawTree(stree, ssent))
            print(system)
            print(system.root)
            result = TreePairResult(0, gtree, gsent, stree, ssent, self.param).scores()
            assert False
