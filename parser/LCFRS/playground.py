from dependency.induction import induce_grammar, the_terminal_labeling_factory, cfg
from grammar.lcfrs import LCFRS
from hybridtree.general_hybrid_tree import HybridTree
from dependency.labeling import the_labeling_factory
from hybridtree.monadic_tokens import CoNLLToken, construct_conll_token
from parser.LCFRS.LCFRS_conversion import parse_LCFRS
from LCFRS_conversion import parse_LCFRS
from sys import stderr


def play_with_parser():
    tree = hybrid_tree_1()
    tree2 = hybrid_tree_2()
    terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

    (_, grammar) = induce_grammar([tree, tree2],
                                  the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                  terminal_labeling.token_label, [cfg], 'START')

    print map((lambda x: x.pos()), tree.full_token_yield())

    print parse_LCFRS(grammar, map((lambda x: x.pos()), tree.full_token_yield()))
    return





def hybrid_tree_1():
    tree = HybridTree()
    tree.add_node('v1', CoNLLToken('Piet', '_', 'NP', 'NP', '_', 'SBJ'), True)
    tree.add_node('v21', CoNLLToken('Marie', '_', 'N', 'N', '_', 'OBJ'), True)
    tree.add_node('v', CoNLLToken('helpen', '_', 'V', 'V', '_', 'ROOT'), True)
    tree.add_node('v2', CoNLLToken('lezen', '_', 'V', 'V', '_', 'VBI'), True)
    tree.add_child('v', 'v2')
    tree.add_child('v', 'v1')
    tree.add_child('v2', 'v21')
    tree.add_to_root('v')
    tree.reorder()
    return tree


def hybrid_tree_2():
    tree2 = HybridTree()
    tree2.add_node('v1', CoNLLToken('Piet', '_', 'NP', 'NP', '_', 'SBJ'), True)
    tree2.add_node('v211', CoNLLToken('Marie', '_', 'N', 'N', '_', 'OBJ'), True)
    tree2.add_node('v', CoNLLToken('helpen', '_', 'V', 'V', '_', 'ROOT'), True)
    tree2.add_node('v2', CoNLLToken('leren', '_', 'V', 'V', '_', 'VBI'), True)
    tree2.add_node('v21', CoNLLToken('lezen', '_', 'V', 'V', '_', 'VFIN'), True)
    tree2.add_child('v', 'v2')
    tree2.add_child('v', 'v1')
    tree2.add_child('v2', 'v21')
    tree2.add_child('v21', 'v211')
    tree2.add_to_root('v')
    tree2.reorder()
    return tree2



play_with_parser()