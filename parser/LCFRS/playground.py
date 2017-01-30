import Queue
from collections import deque

from dependency.induction import induce_grammar, the_terminal_labeling_factory, cfg
from grammar.lcfrs import LCFRS
from hybridtree.general_hybrid_tree import HybridTree
from dependency.labeling import the_labeling_factory
from hybridtree.monadic_tokens import CoNLLToken, construct_conll_token
from sys import stderr
from parser.LCFRS.LCFRS_Parser_Wrapper import PyLCFRSFactory
from corpora.conll_parse import parse_conll_corpus


def play_with_parser():
    """
    :t grammar LCFRS
    :return:
    """
    tree = hybrid_tree_1()
    tree2 = hybrid_tree_2()
    terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')

    (_, grammar) = induce_grammar([tree, tree2],
                                  the_labeling_factory().create_simple_labeling_strategy('empty', 'pos'),
                                  terminal_labeling.token_label, [cfg], 'START')

    print map((lambda x: x.pos()), tree.full_token_yield())

    # print '\n\n'
    # print grammar

    factory = PyLCFRSFactory(grammar.start())

    factory.import_grammar(grammar)
    word = map((lambda x: x.pos()), tree.full_token_yield())

    factory.do_parse(word)

    print 'Passive Items:'
    passiveItems = factory.get_passive_items_map()
    for (i, pItem) in passiveItems.iteritems():
        print str(i) + ': ' + str(pItem[0]) + str(pItem[1])

    print '\n\n Trace:'
    trace = factory.convert_trace()
    for (i, pItem) in trace.iteritems():
        print str(i) + ': ' + str(pItem)

    print "No of parses: " + str(count_parses(passiveItems, trace, grammar.start(), len(word)))




def play_with_corpus():
    train = '../../res/dependency_conll/german/tiger/train/german_tiger_train.conll'
    limit_train = 100
    test = train
    limit_test = 10
    # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
    trees = parse_conll_corpus(train, False, limit_train)

    test_trees = parse_conll_corpus(test, False, limit_train)

    primary_labelling = the_labeling_factory().create_simple_labeling_strategy("child", "deprel")
    term_labelling = the_terminal_labeling_factory().get_strategy('pos')
    start = 'START'
    recursive_partitioning = [cfg]

    (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                             recursive_partitioning, start)

    factory = PyLCFRSFactory(grammar_prim.start())
    factory.import_grammar(grammar_prim)

    for i in range(0, limit_test):
        tree = test_trees.next()
        word = map((lambda x: x.pos()), tree.full_token_yield())
        factory.do_parse(word)
        print str(word) + "    " + str(len(word))
        passiveItems = factory.get_passive_items_map()
        trace = factory.convert_trace()
        print "No of parses: " , count_parses(passiveItems, trace, grammar_prim.start(), len(word))




def count_parses(passiveItems, trace, initial_nont, wordlen):

    initial = passiveItems.keys()[passiveItems.values().index((initial_nont, [(0L, wordlen)]))]
    (result, _) = parses_per_pitem(trace, initial, {})
    return result


def parses_per_pitem(trace, pItemNo, resultMap):
    if pItemNo in resultMap:
        return resultMap[pItemNo], resultMap
    result = 0
    for pItemList in trace[pItemNo]:
        noPerTrace = 1
        for pItem in pItemList[1]:
            (count, resultMap) = parses_per_pitem(trace, pItem, resultMap)
            noPerTrace *= count
        result += noPerTrace
    resultMap[pItemNo] = result
    return result, resultMap







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



# play_with_parser()
play_with_corpus()