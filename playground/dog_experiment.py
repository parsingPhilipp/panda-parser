from __future__ import print_function
import json
import os
from os import path
import shutil
import subprocess
import sys
# structure representation and corpus tools
from graphs.dog import DeepSyntaxGraph
from hybridtree.monadic_tokens import ConstituentTerminal
from corpora.tiger_parse import sentence_names_to_deep_syntax_graphs
# grammar induction
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory
from grammar.induction.terminal_labeling import PosTerminals
from graphs.graph_decomposition import simple_labeling, top_bot_labeling, missing_child_labeling, induction_on_a_corpus, dog_evaluation
# reduct computation
from graphs.graph_bimorphism_json_export import export_corpus_to_json, export_dog_grammar_to_json
from graphs.schick_parser_rtg_import import read_rtg
from util.enumerator import Enumerator
# EM/SM training
from parser.supervised_trainer.trainer import PyDerivationManager
from parser.trace_manager.sm_trainer_util import PyGrammarInfo, PyStorageManager
from parser.trace_manager.sm_trainer import PySplitMergeTrainerBuilder, build_PyLatentAnnotation_initial
# parsing / testing
from parser.cpp_cfg_parser.parser_wrapper import CFGParser
from parser.gf_parser.gf_interface import GFParser_k_best, GFParser
from parser.coarse_to_fine_parser.coarse_to_fine import Coarse_to_fine_parser
from graphs.parse_accuracy import PredicateArgumentScoring
from graphs.util import render_and_view_dog


schick_executable = 'HypergraphReduct-1.0-SNAPSHOT.jar'
threads = 1


def run_experiment():
    start = 1
    stop = 50

    test_start = 7001
    test_stop = 7200

    # path = "res/tiger/tiger_release_aug07.corrected.16012013.utf8.xml"
    corpus_path = "res/tiger/tiger_8000.xml"
    exclude = []
    train_dsgs = sentence_names_to_deep_syntax_graphs(
        ['s' + str(i) for i in range(start, stop + 1) if i not in exclude]
        , corpus_path
        , hold=False)
    test_dsgs = sentence_names_to_deep_syntax_graphs(
        ['s' + str(i) for i in range(test_start, test_stop + 1) if i not in exclude]
        , corpus_path
        , hold=False)

    # Grammar induction
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

    # Compute reducts, i.e., intersect grammar with each training dsg
    terminal_map = Enumerator()
    basedir = '/tmp/dog_experiments'
    if not os.path.isdir(basedir):
        os.makedirs(basedir)
    data = export_dog_grammar_to_json(grammar, terminal_map)
    grammar_path = path.join(basedir, 'grammar.json')
    with open(grammar_path, 'w') as file:
        json.dump(data, file)

    corpus_path = path.join(basedir, 'corpus.json')
    with open(corpus_path, 'w') as file:
        json.dump(export_corpus_to_json(train_dsgs, terminal_map, terminal_labeling=term_labeling), file)

    with open(path.join(basedir, 'enumerator.enum'), 'w') as file:
        terminal_map.print_index(file)

    reduct_dir = path.join(basedir, 'reduct_grammars')
    if os.path.isdir(reduct_dir):
        shutil.rmtree(reduct_dir)
    os.makedirs(reduct_dir)
    p = subprocess.Popen([' '.join(
        ["java", "-jar", os.path.join("util", schick_executable), 'dog-reduct', '-g', grammar_path, '-t', corpus_path,
         "-o", reduct_dir])], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        nextline = p.stdout.readline()
        if nextline == '' and p.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()

    p.wait()
    rtgs = []
    for i in range(1, len(train_dsgs) + 1):
        rtgs.append(read_rtg(path.join(reduct_dir, str(i) + '.gra')))

    derivation_manager = PyDerivationManager(grammar)
    derivation_manager.convert_rtgs_to_hypergraphs(rtgs)
    derivation_manager.serialize(path.join(basedir, 'reduct_manager.trace'))

    # Training
    ## prepare EM training
    em_epochs = 20
    seed = 0
    smoothing_factor = 0.01
    split_randomization = 0.01
    sm_cycles = 2
    merge_percentage = 50.0
    grammarInfo = PyGrammarInfo(grammar, derivation_manager.get_nonterminal_map())
    storageManager = PyStorageManager()

    em_builder = PySplitMergeTrainerBuilder(derivation_manager, grammarInfo)
    em_builder.set_em_epochs(em_epochs)
    em_builder.set_simple_expector(threads=threads)
    emTrainer = em_builder.build()

    # randomize initial weights and do em training
    la_no_splits = build_PyLatentAnnotation_initial(grammar, grammarInfo, storageManager)
    la_no_splits.add_random_noise(grammarInfo, seed=seed)
    emTrainer.em_train(la_no_splits)
    la_no_splits.project_weights(grammar, grammarInfo)

    ## prepare SM training
    builder = PySplitMergeTrainerBuilder(derivation_manager, grammarInfo)
    builder.set_em_epochs(em_epochs)
    builder.set_split_randomization(1.0, seed + 1)
    builder.set_simple_expector(threads=threads)
    builder.set_smoothing_factor(smoothingFactor=smoothing_factor)
    builder.set_split_randomization(percent=split_randomization)
    # builder.set_scc_merger(-0.2)
    builder.set_percent_merger(merge_percentage)
    splitMergeTrainer = builder.build()

    # splitMergeTrainer.setMaxDrops(validationDropIterations, mode="smoothing")
    splitMergeTrainer.setEMepochs(em_epochs, mode="smoothing")

    # set initial latent annotation
    latentAnnotation = [la_no_splits]

    # carry out split/merge training and do parsing
    parsing_method = "filter-ctf"
    # parsing_method = "single-best-annotation"
    k_best = 50
    for i in range(1, sm_cycles + 1):
        splitMergeTrainer.reset_random_seed(seed + i + 1)
        latentAnnotation.append(splitMergeTrainer.split_merge_cycle(latentAnnotation[-1]))
        print("Cycle: ", i)
        if parsing_method == "single-best-annotation":
            smGrammar = latentAnnotation[i].build_sm_grammar(grammar
                                                         , grammarInfo
                                                         , rule_pruning=0.0001
                                                         , rule_smoothing=0.1)
            print("Rules in smoothed grammar: ", len(smGrammar.rules()))
            parser = GFParser(smGrammar)
        elif parsing_method == "filter-ctf":
            latentAnnotation[-1].project_weights(grammar, grammarInfo)
            parser = Coarse_to_fine_parser(grammar, GFParser_k_best, latentAnnotation[-1], grammarInfo, derivation_manager.get_nonterminal_map(), k=k_best)
        else:
            raise(Exception())
        do_parsing(parser, test_dsgs, term_labeling_token)
        del parser


def do_parsing(parser, test_dsgs, term_labeling_token):
    interactive = True  # False

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
