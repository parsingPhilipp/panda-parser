from __future__ import print_function
import timeit

from dependency.induction import induce_grammar
from grammar.induction.recursive_partitioning import direct_extraction, cfg
from grammar.induction.terminal_labeling import the_terminal_labeling_factory
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

    print(map((lambda x: x.pos()), tree.full_token_yield()))

    # print '\n\n'
    # print grammar

    factory = PyLCFRSFactory(grammar.start())

    factory.import_grammar(grammar)
    word = map((lambda x: x.pos()), tree.full_token_yield())

    parser = factory.build_parser()

    parser.do_parse(word)

    print('Passive Items:')
    passiveItems = parser.get_passive_items_map()
    for (i, pItem) in passiveItems.iteritems():
        print(i, ': ', pItem[0], pItem[1])

    print('\n\n Trace:')
    trace = parser.convert_trace()
    for (i, pItem) in trace.iteritems():
        print(i, ': ', pItem)

    print("No of parses: ", count_parses(passiveItems, trace, parser.get_initial_passive_item()))




def play_with_corpus():
    train = '../../res/dependency_conll/german/tiger/train/german_tiger_train.conll'
    limit_train = 100
    test = train
    limit_test = 50
    # test = '../../res/dependency_conll/german/tiger/test/german_tiger_test.conll'
    trees = parse_conll_corpus(train, False, limit_train)

    test_trees = parse_conll_corpus(test, False, limit_train)

    primary_labelling = the_labeling_factory().create_simple_labeling_strategy("child", "deprel")
    term_labelling = the_terminal_labeling_factory().get_strategy('pos')
    start = 'START'
    recursive_partitioning = [direct_extraction]

    (n_trees, grammar_prim) = induce_grammar(trees, primary_labelling, term_labelling.token_label,
                                             recursive_partitioning, start)

    factory = PyLCFRSFactory(grammar_prim.start())
    factory.import_grammar(grammar_prim)
    parser = factory.build_parser()

    for i in range(0, limit_test):
        tree = next(test_trees)
        if len(tree.token_yield()) > 15:
            continue
        word = list(map((lambda x: x.pos()), tree.token_yield()))
        parser.do_parse(word)
        passiveItems = parser.get_passive_items_map()
        trace = parser.convert_trace()
        print("Word length: ", len(word),
              " - #passive Items: ", len(passiveItems),
              " - #parses: ", count_parses(passiveItems, trace, parser.get_initial_passive_item()))



def play_with_manual_grammar():
    factory = PyLCFRSFactory("S")
    factory.new_rule("S")
    factory.add_variable(0,0)
    factory.add_variable(1,0)
    factory.complete_argument()
    factory.add_rule_to_grammar("AB", 0)

    factory.new_rule("S")
    factory.add_variable(0, 0)
    factory.add_variable(1, 0)
    factory.complete_argument()
    factory.add_rule_to_grammar("AC", 1)

    factory.new_rule("A")
    factory.add_terminal("a")
    factory.complete_argument()
    factory.add_rule_to_grammar("", 2)

    factory.new_rule("A")
    factory.add_terminal("a")
    factory.complete_argument()
    factory.add_rule_to_grammar("", 3)

    factory.new_rule("B")
    factory.add_terminal("b")
    factory.complete_argument()
    factory.add_rule_to_grammar("", 4)

    factory.new_rule("C")
    factory.add_terminal("b")
    factory.complete_argument()
    factory.add_rule_to_grammar("", 5)

    factory.new_rule("C")
    factory.add_terminal("b")
    factory.complete_argument()
    factory.add_rule_to_grammar("", 6)

    factory.new_rule("C")
    factory.add_terminal("b")
    factory.complete_argument()
    factory.add_rule_to_grammar("", 7)


    word = "ab"

    parser = factory.build_parser()
    parser.do_parse(word)

    print('Passive Items:')
    passiveItems = parser.get_passive_items_map()
    for (i, pItem) in passiveItems.iteritems():
        print(i, ': ', pItem[0], pItem[1])

    print('\n\n Trace:')
    trace = parser.convert_trace()
    for (i, pItem) in trace.iteritems():
        print(i, ': ', pItem)

    print("No of parses: ", count_parses(passiveItems, trace, parser.get_initial_passive_item()))




def count_parses(passiveItems, trace, initial_passive_item):
    # initial = passiveItems.keys()[passiveItems.values().index(initial_passive_item)]
    for key in passiveItems:
        if passiveItems[key] == initial_passive_item:
            initial = key
            break
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


seconds = 0
# play_with_parser()
# play_with_manual_grammar()
seconds = timeit.timeit(play_with_corpus, number=1)

print('Finished in ', seconds, ' seconds')
