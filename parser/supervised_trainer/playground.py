from __future__ import print_function

from sys import stderr

from corpora.conll_parse import parse_conll_corpus
from dependency.induction import induce_grammar
from grammar.induction.recursive_partitioning import cfg
from grammar.induction.terminal_labeling import the_terminal_labeling_factory
from dependency.labeling import the_labeling_factory
from hybridtree.monadic_tokens import construct_constituent_token
from parser.derivation_interface import derivation_to_hybrid_tree
from parser.gf_parser.gf_interface import GFParser_k_best, GFParser
from parser.supervised_trainer.trainer import PyDerivationManager
from parser.trace_manager.sm_trainer_util import PyGrammarInfo, PyStorageManager
from parser.trace_manager.sm_trainer import PySplitMergeTrainerBuilder, build_PyLatentAnnotation_initial


limit_train = 20
limit_test = 10
train = '../../res/dependency_conll/german/tiger/train/german_tiger_train.conll'
test = train
max_cycles = 2
parsing = True


def obtain_derivations(grammar, term_labelling):
    # build parser
    tree_yield = term_labelling.prepare_parser_input
    parser = GFParser_k_best(grammar, k=50)

    # parse sentences
    trees = parse_conll_corpus(test, False, limit_test)
    for i, tree in enumerate(trees):
        print("Parsing sentence ", i, file=stderr, )

        parser.set_input(tree_yield(tree.token_yield()))
        parser.parse()

        derivations = [der for der in parser.k_best_derivation_trees()]

        print("# derivations: ", len(derivations), file=stderr)
        parser.clear()

        for der in derivations:
            yield der[1]


def main():
    # induce grammar from a corpus
    trees = parse_conll_corpus(train, False, limit_train)
    nonterminal_labelling = the_labeling_factory().create_simple_labeling_strategy("childtop", "deprel")
    term_labelling = the_terminal_labeling_factory().get_strategy('pos')
    start = 'START'
    recursive_partitioning = [cfg]
    _, grammar = induce_grammar(trees, nonterminal_labelling, term_labelling.token_label, recursive_partitioning, start)

    # compute some derivations
    derivations = obtain_derivations(grammar, term_labelling)

    # create derivation manager and add derivations
    manager = PyDerivationManager(grammar)
    manager.convert_derivations_to_hypergraphs(derivations)
    manager.serialize(b"/tmp/derivations.txt")

    # build and configure split/merge trainer and supplementary objects

    rule_to_nonterminals = []
    for i in range(0, len(grammar.rule_index())):
        rule = grammar.rule_index(i)
        nonts = [manager.get_nonterminal_map().object_index(rule.lhs().nont())] + [manager.get_nonterminal_map().object_index(nont) for nont in rule.rhs()]
        rule_to_nonterminals.append(nonts)

    grammarInfo = PyGrammarInfo(grammar, manager.get_nonterminal_map())
    storageManager = PyStorageManager()
    builder = PySplitMergeTrainerBuilder(manager, grammarInfo)
    builder.set_em_epochs(20)
    builder.set_percent_merger(60.0)

    splitMergeTrainer = builder.build()

    latentAnnotation = [build_PyLatentAnnotation_initial(grammar, grammarInfo, storageManager)]

    for i in range(max_cycles + 1):
        latentAnnotation.append(splitMergeTrainer.split_merge_cycle(latentAnnotation[-1]))
        # pickle.dump(map(lambda la: la.serialize(), latentAnnotation), open(sm_info_path, 'wb'))
        smGrammar = latentAnnotation[i].build_sm_grammar( grammar
                                                         , grammarInfo
                                                         , rule_pruning=0.0001
                                                         , rule_smoothing=0.01)
        print("Cycle: ", i, "Rules: ", len(smGrammar.rules()))

        if parsing:
            parser = GFParser(smGrammar)

            trees = parse_conll_corpus(test, False, limit_test)
            for tree in trees:
                parser.set_input(term_labelling.prepare_parser_input(tree.token_yield()))
                parser.parse()
                if parser.recognized():
                    print(derivation_to_hybrid_tree(
                            parser.best_derivation_tree()
                            , [token.pos() for token in tree.token_yield()]
                            , [token.form() for token in tree.token_yield()]
                            , construct_constituent_token
                            )
                         )


if __name__ == '__main__':
    main()
