from __future__ import print_function
from corpora.tiger_parse import sentence_names_to_deep_syntax_graphs
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory
from grammar.induction.terminal_labeling import PosTerminals
from hybridtree.monadic_tokens import ConstituentTerminal
from graphs.dog import DeepSyntaxGraph
from graphs.graph_decomposition import simple_labeling, top_bot_labeling, missing_child_labeling, induction_on_a_corpus, dog_evaluation
from graphs.parse_accuracy import PredicateArgumentScoring
from graphs.util import render_and_view_dog
from parser.cpp_cfg_parser.parser_wrapper import CFGParser


def run_experiment():
    interactive = True  # False
    start = 1
    stop = 7000

    test_start = 7001
    test_stop = 7200

    # path = "res/tiger/tiger_release_aug07.corrected.16012013.utf8.xml"
    path = "res/tiger/tiger_8000.xml"
    exclude = []
    train_dsgs = sentence_names_to_deep_syntax_graphs(
        ['s' + str(i) for i in range(start, stop + 1) if i not in exclude]
        , path
        , hold=False)
    test_dsgs = sentence_names_to_deep_syntax_graphs(
        ['s' + str(i) for i in range(test_start, test_stop + 1) if i not in exclude]
        , path
        , hold=False)

    rec_part_strategy = the_recursive_partitioning_factory().getPartitioning('cfg')[0]

    def label_edge(edge):
        if isinstance(edge.label, ConstituentTerminal):
            return edge.label.pos()
        else:
            return edge.label

    def stupid_edge(edge):
        return "X"

    def label_child(edge, j):
        return edge.get_function(j)

    # nonterminal_labeling = lambda nodes, dsg: simple_labeling(nodes, dsg, label_edge)
    nonterminal_labeling = lambda nodes, dsg: top_bot_labeling(nodes, dsg, label_edge, stupid_edge)
    # nonterminal_labeling = lambda nodes, dsg: missing_child_labeling(nodes, dsg, label_edge, label_child)


    term_labeling_token = PosTerminals()

    def term_labeling(token):
        if isinstance(token, ConstituentTerminal):
            return term_labeling_token.token_label(token)
        else:
            return token

    grammar = induction_on_a_corpus(train_dsgs, rec_part_strategy, nonterminal_labeling, term_labeling)
    grammar.make_proper()

    parser = CFGParser(grammar)

    scorer = PredicateArgumentScoring()

    not_output_connected = 0

    for dsg in test_dsgs:
        parser.set_input(term_labeling_token.prepare_parser_input(dsg.sentence))
        parser.parse()

        f = lambda token: token.pos() if isinstance(token, ConstituentTerminal) else token
        dsg.dog.project_labels(f)

        if parser.recognized():
            derivation = parser.best_derivation_tree()
            dog, sync = dog_evaluation(derivation)

            if not dog.output_connected():
                not_output_connected += 1
                if interactive:
                    z2 = render_and_view_dog(dog, "parsed_" + dsg.label)
                    z2.communicate()

            dsg2 = DeepSyntaxGraph(dsg.sentence, dog, sync)

            scorer.add_accuracy_frames(
                dsg.labeled_frames(guard=lambda x: len(x[1]) > 0),
                dsg2.labeled_frames(guard=lambda x: len(x[1]) > 0)
            )

            # print('dsg: ', dsg.dog, '\n', [dsg.get_graph_position(i) for i in range(len(dsg.sentence))], '\n\n parsed: ', dsg2.dog, '\n', [dsg2.get_graph_position(i+1) for i in range(len(dsg2.sentence))])
            # print()
            if False and interactive:
                if dsg.label == 's50':
                    pass
                if dsg.dog != dog:
                    z1 = render_and_view_dog(dsg.dog, "corpus_" + dsg.label)
                    z2 = render_and_view_dog(dog, "parsed_" + dsg.label)
                    z1.communicate()
                    z2.communicate()
        else:

            scorer.add_failure(dsg.labeled_frames(guard=lambda x: len(x[1]) > 0))

        parser.clear()
    print("Parse failures:", scorer.labeled_frame_scorer.n_failures())
    print("Not output connected", not_output_connected)
    print("Labeled frames:")
    print("P", scorer.labeled_frame_scorer.precision(), "R", scorer.labeled_frame_scorer.recall(),
          "F1", scorer.labeled_frame_scorer.fmeasure(), "EM", scorer.labeled_frame_scorer.exact_match())
    print("Unlabeled frames:")
    print("P", scorer.unlabeled_frame_scorer.precision(), "R", scorer.unlabeled_frame_scorer.recall(),
          "F1", scorer.unlabeled_frame_scorer.fmeasure(), "EM", scorer.unlabeled_frame_scorer.exact_match())
    print("Labeled dependencies:")
    print("P", scorer.labeled_dependency_scorer.precision(), "R", scorer.labeled_dependency_scorer.recall(),
          "F1", scorer.labeled_dependency_scorer.fmeasure(), "EM", scorer.labeled_dependency_scorer.exact_match())
    print("Unlabeled dependencies:")
    print("P", scorer.unlabeled_dependency_scorer.precision(), "R", scorer.unlabeled_dependency_scorer.recall(),
          "F1", scorer.unlabeled_dependency_scorer.fmeasure(), "EM", scorer.unlabeled_dependency_scorer.exact_match())


if __name__ == "__main__":
    run_experiment()
