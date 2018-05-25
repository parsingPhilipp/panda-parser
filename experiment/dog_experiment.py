from __future__ import print_function
import json
import os
from os import path
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from itertools import product
# structure representation and corpus tools
from graphs.dog import DeepSyntaxGraph
from hybridtree.monadic_tokens import ConstituentTerminal, ConstituentCategory
from corpora.tiger_parse import sentence_names_to_deep_syntax_graphs
# grammar induction
from grammar.induction.recursive_partitioning import fanout_limited_partitioning, fanout_limited_partitioning_left_to_right
from grammar.induction.terminal_labeling import PosTerminals
from graphs.graph_decomposition import simple_labeling, top_bot_labeling, missing_child_labeling, induction_on_a_corpus, dog_evaluation, compute_decomposition, induce_grammar_from
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
from experiment.base_experiment import ScoringExperiment
from experiment.resources import TRAINING, TESTING, RESULT, CorpusFile, ScorerResource
from sys import stdout


SCHICK_PARSER_JAR = 'HypergraphReduct-1.0-SNAPSHOT.jar'
THREADS = 1


class DOGScorerResource(ScorerResource):
    def __init__(self, path=None, start=None, end=None):
        super(self.__class__, self).__init__(path, start, end)
        self.scorer = PredicateArgumentScoring()

    def score(self, system, gold, secondaries=None):
        found = self.get_labeled_frames(system)
        correct = self.get_labeled_frames(gold)
        self.scorer.add_accuracy_frames(found, correct)

    def get_labeled_frames(self, obj):
        return obj.labeled_frames(guard=lambda x: len(x[1]) > 0)

    def failure(self, gold):
        self.scorer.add_failure(self.get_labeled_frames(gold))


class InductionSettings:
    def __init__(self):
        self.rec_part_strategy = None
        self.nonterminal_labeling = None
        self.terminal_labeling = None
        self.terminal_labeling_token = None
        self.start = None
        self.normalize = True  # normalize node indices in graph fragments (True) xor keep names of origin (False)
        # No normalization will cause rules to be syntactically different, when added to a grammar, resulting in
        # spurious duplicates.
        self.reorder_children = False  # reorder children alphabetically
        self.binarize = True
        self.direction = None
        self.subgrouping = False
        self.fanout = 1

    @staticmethod
    def modify_token(token):
        if isinstance(token, ConstituentCategory):
            token_new = deepcopy(token)
            token_new.set_category(token.category() + '-BAR')
            return token_new
        elif isinstance(token, str):
            return token + '-BAR'
        else:
            assert False

    @staticmethod
    def is_bin(token):
        if isinstance(token, ConstituentCategory):
            if token.category().endswith('-BAR'):
                return True
        elif isinstance(token, str):
            if token.endswith('-BAR'):
                return True
        return False

    def __str__(self):
        attributes = [('recursive partitioning', self.rec_part_strategy),
                      ('nonterminal labeling', self.nonterminal_labeling),
                      ('terminal labeling', self.terminal_labeling),
                      ('terminal labeling token', self.terminal_labeling_token),
                      ('start symbol', self.start),
                      ('normalize graph vertices', self.normalize),
                      ('recorder children alphabetically', self.reorder_children),
                      ('binarize DOG before induction', self.binarize),
                      ('direction', self.direction),
                      ('subgrouping', self.subgrouping),
                      ('fanout', self.fanout)
                      ]
        return '\n'.join(['\t' + x[0] + ' : ' + str(x[1]) for x in attributes])


class Statistics:
    def __init__(self):
        self.not_output_connected = 0


class DOGExperiment(ScoringExperiment):
    def __init__(self, induction_settings):
        super(self.__class__, self).__init__()
        self.induction_settings = induction_settings
        self.statistics = Statistics()
        self.interactive = False
        self.resources[RESULT] = DOGScorerResource()
        self.k_best = 50
        self.max_score = 1.0

    def induce_from(self, dsg):
        rec_part = self.induction_settings.rec_part_strategy(dsg)
        # if calc_fanout(rec_part) > 1 or calc_rank(rec_part) > 2:
        #     rec_part = rec_part_strategy(dsg)
        #     assert False
        decomposition = compute_decomposition(dsg, rec_part)
        dsg_grammar = induce_grammar_from(dsg, rec_part, decomposition, self.induction_settings.nonterminal_labeling,
                                          self.induction_settings.terminal_labeling, self.induction_settings.terminal_labeling, self.induction_settings.start,
                                          self.induction_settings.normalize)
        return dsg_grammar, None

    def postprocess_grammar(self, grammar):
        if self.purge_rule_freq is not None:
            grammar.purge_rules(self.purge_rule_freq)

        max_node = 0
        for rule in grammar.rules():
            dog = rule.dcp()[0]
            sync = rule.dcp()[1]
            translation = {node: max_node + 1 + j for j, node in enumerate(dog.nodes)}
            if not set(translation.keys()).isdisjoint(translation.values()):
                tmp_max = max(dog.nodes + [max_node + len(dog.nodes)])
                t1 = {node: tmp_max + 1 + j for j, node in enumerate(dog.nodes)}
                t2 = {tmp_max + 1 + j: max_node + 1 + j for j in range(len(dog.nodes))}
                dog.rename_nodes(t1)
                dog.rename_nodes(t2)
            else:
                dog.rename_nodes(translation)
            for sync_sub in sync:
                dog.replace_inplace_many(sync_sub, translation)
            max_node += len(dog.nodes)

        grammar.make_proper()

    def read_corpus(self, resource):
        return sentence_names_to_deep_syntax_graphs(
            ['s' + str(i) for i in range(resource.start, resource.end + 1) if i not in resource.exclude]
            , resource.path
            , hold=False
            , reorder_children=self.induction_settings.reorder_children)

    def preprocess_before_induction(self, dsg):
        if self.induction_settings.binarize:
            return dsg.binarize(bin_modifier=self.induction_settings.modify_token)
        else:
            return dsg

    def parsing_postprocess(self, sentence, derivation, label=None):
        dog, sync = dog_evaluation(derivation, compress=False)

        if self.induction_settings.binarize:
            dog = dog.debinarize(is_bin=self.induction_settings.is_bin)

        if not dog.output_connected():
            self.statistics.not_output_connected += 1
            if self.interactive:
                z2 = render_and_view_dog(dog, "parsed_" + str(label))
                # z2.communicate()

        return DeepSyntaxGraph(sentence, dog, sync)

    def obtain_label(self, obj):
        return obj.label

    def obtain_sentence(self, obj):
        return obj.sentence

    def parsing_preprocess(self, obj):
        return self.induction_settings.terminal_labeling_token.prepare_parser_input(self.obtain_sentence(obj))

    def initialize_parser(self):
        self.parser = GFParser_k_best(self.base_grammar, k=self.k_best)

    def evaluate(self, result_resource, gold_resource):
        scorer = result_resource.scorer
        print("Parse failures:", scorer.labeled_frame_scorer.n_failures())
        print("Not output connected", self.statistics.not_output_connected)
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
              "F1", scorer.unlabeled_dependency_scorer.fmeasure(), "EM",
              scorer.unlabeled_dependency_scorer.exact_match())

    def score_object(self, obj, gold):
        relevant = gold.labeled_frames(guard=lambda x: len(x[1]) > 0)
        retrieved = obj.labeled_frames(guard=lambda x: len(x[1]) > 0)
        _, _, f1 = self.precision_recall_f1(relevant, retrieved)
        return f1

    def print_config(self, file=stdout):
        super(DOGExperiment, self).print_config(file)
        print("k-best", self.k_best)
        print("Induction Settings {", file=file)
        print(self.induction_settings, "\n}", file=file)


def run_experiment(rec_part_strategy, nonterminal_labeling, exp, reorder_children, binarize=True):
    start = 1
    stop = 7000

    test_start = 7001
    test_stop = 7200

    # path = "res/tiger/tiger_release_aug07.corrected.16012013.utf8.xml"
    corpus_path = "res/tiger/tiger_8000.xml"
    exclude = []
    train_dsgs = sentence_names_to_deep_syntax_graphs(
        ['s' + str(i) for i in range(start, stop + 1) if i not in exclude]
        , corpus_path
        , hold=False
        , reorder_children=reorder_children)
    test_dsgs = sentence_names_to_deep_syntax_graphs(
        ['s' + str(i) for i in range(test_start, test_stop + 1) if i not in exclude]
        , corpus_path
        , hold=False
        , reorder_children=reorder_children)

    # Grammar induction
    term_labeling_token = PosTerminals()

    def term_labeling(token):
        if isinstance(token, ConstituentTerminal):
            return term_labeling_token.token_label(token)
        else:
            return token

    if binarize:
        def modify_token(token):
            if isinstance(token, ConstituentCategory):
                token_new = deepcopy(token)
                token_new.set_category(token.category() + '-BAR')
                return token_new
            elif isinstance(token, str):
                return token + '-BAR'
            else:
                assert False
        train_dsgs = [dsg.binarize(bin_modifier=modify_token) for dsg in train_dsgs]

        def is_bin(token):
            if isinstance(token, ConstituentCategory):
                if token.category().endswith('-BAR'):
                    return True
            elif isinstance(token, str):
                if token.endswith('-BAR'):
                    return True
            return False

        def debinarize(dsg):
            return dsg.debinarize(is_bin=is_bin)

    else:
        debinarize = id

    grammar = induction_on_a_corpus(train_dsgs, rec_part_strategy, nonterminal_labeling, term_labeling)
    grammar.make_proper()

    print("Nonterminals", len(grammar.nonts()), "Rules", len(grammar.rules()))

    parser = GFParser_k_best(grammar, k=500)
    return do_parsing(parser, test_dsgs, term_labeling_token, oracle=True, debinarize=debinarize)


    # Compute reducts, i.e., intersect grammar with each training dsg
    basedir = path.join('/tmp/dog_experiments', 'exp' + str(exp))
    reduct_dir = path.join(basedir, 'reduct_grammars')

    terminal_map = Enumerator()
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

    if os.path.isdir(reduct_dir):
        shutil.rmtree(reduct_dir)
    os.makedirs(reduct_dir)
    p = subprocess.Popen([' '.join(
        ["java", "-jar", os.path.join("util", SCHICK_PARSER_JAR), 'dog-reduct', '-g', grammar_path, '-t', corpus_path,
         "-o", reduct_dir])], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        nextline = p.stdout.readline()
        if nextline == '' and p.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()

    p.wait()
    p.stdout.close()

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
    em_builder.set_simple_expector(threads=THREADS)
    emTrainer = em_builder.build()

    # randomize initial weights and do em training
    la_no_splits = build_PyLatentAnnotation_initial(grammar, grammarInfo, storageManager)
    la_no_splits.add_random_noise(seed=seed)
    emTrainer.em_train(la_no_splits)
    la_no_splits.project_weights(grammar, grammarInfo)

    do_parsing(CFGParser(grammar), test_dsgs, term_labeling_token)
    return
    ## prepare SM training
    builder = PySplitMergeTrainerBuilder(derivation_manager, grammarInfo)
    builder.set_em_epochs(em_epochs)
    builder.set_split_randomization(1.0, seed + 1)
    builder.set_simple_expector(threads=THREADS)
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
            parser = Coarse_to_fine_parser(grammar, latentAnnotation[-1], grammarInfo, derivation_manager.get_nonterminal_map(), base_parser_type=GFParser_k_best, k=k_best)
        else:
            raise(Exception())
        do_parsing(parser, test_dsgs, term_labeling_token)
        del parser


def do_parsing(parser, test_dsgs, term_labeling_token, oracle=False, debinarize=id):
    interactive = True  # False

    scorer = PredicateArgumentScoring()

    not_output_connected = 0

    start = time.time()
    for dsg in test_dsgs:
        parser.set_input(term_labeling_token.prepare_parser_input(dsg.sentence))
        parser.parse()

        f = lambda token: token.pos() if isinstance(token, ConstituentTerminal) else token
        dsg.dog.project_labels(f)

        if parser.recognized():
            if oracle:
                derivation = compute_oracle_derivation(parser, dsg, debinarize)
            else:
                derivation = parser.best_derivation_tree()
            dog, sync = dog_evaluation(derivation)

            if not dog.output_connected():
                not_output_connected += 1
                if interactive:
                    z2 = render_and_view_dog(dog, "parsed_" + dsg.label)
                    # z2.communicate()

            dsg2 = DeepSyntaxGraph(dsg.sentence, debinarize(dog), sync)

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
    print("Completed parsing in", time.time() - start, "seconds.")
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
    return scorer


def compute_oracle_derivation(parser, dsg, mapping=id):
    validationMethod = "F1"
    best_der = None
    best_f1 = -1
    best_prec = -1
    best_rec = -1

    relevant = dsg.labeled_frames(guard=lambda x: x[1] > 0)
    for _, derivation in parser.k_best_derivation_trees():
        dog, sync = dog_evaluation(derivation)
        dsg2 = DeepSyntaxGraph(dsg.sentence, mapping(dog), sync)
        retrieved = dsg2.labeled_frames(guard=lambda x: x[1] > 0)

        inters = retrieved & relevant

        # in case of parse failure there are two options here:
        #   - parse failure -> no spans at all, thus precision = 1
        #   - parse failure -> a dummy tree with all spans wrong, thus precision = 0

        precision = 1.0 * len(inters) / len(retrieved) \
            if len(retrieved) > 0 else 0
        recall = 1.0 * len(inters) / len(relevant) \
            if len(relevant) > 0 else 0
        fmeasure = 2.0 * precision * recall / (precision + recall) \
            if precision + recall > 0 else 0

        if (validationMethod == "F1" and fmeasure > best_f1)\
                or (validationMethod == "Precision" and precision > best_prec)\
                or (validationMethod == "Recall"and recall > best_rec):
            best_der, best_f1, best_prec, best_rec = derivation, fmeasure, precision, recall

    return best_der


def main():
    directions = ["left-to-right", "right-to-left"]

    def rec_part_strategy(direction, subgrouping, fanout, binarize):
        if direction == "right-to-left":
            return lambda dsg: fanout_limited_partitioning(dsg.recursive_partitioning(subgrouping, weak=binarize), fanout)
        else:
            return lambda dsg: fanout_limited_partitioning_left_to_right(dsg.recursive_partitioning(subgrouping
                                                                                                    , weak=binarize),
                                                                         fanout)

    subgroupings = [True, False]
    fanouts = [1]
    reorder_children = [True, False]

    def label_edge(edge):
        if isinstance(edge.label, ConstituentTerminal):
            return edge.label.pos()
        else:
            return edge.label

    def stupid_edge(edge):
        return "X"

    def label_child(edge, j):
        return edge.get_function(j)

    def simple_nonterminal_labeling(nodes, dsg):
        return simple_labeling(nodes, dsg, label_edge)

    def bot_stupid_nonterminal_labeling(nodes, dsg):
        return top_bot_labeling(nodes, dsg, label_edge, stupid_edge)

    def missing_child_nonterminal_labeling(nodes, dsg):
        return missing_child_labeling(nodes, dsg, label_edge, label_child)

    nonterminal_labelings = [simple_nonterminal_labeling, bot_stupid_nonterminal_labeling,
                             missing_child_nonterminal_labeling]

    start_exp = 0
    exp = start_exp
    scorers = []
    binarize = True
    for direction, subgrouping, fanout, nonterminal_labeling, reorder in \
            product(directions, subgroupings, fanouts, nonterminal_labelings, reorder_children):

        print()
        print("Experiment", exp, "direction", direction, "fanout", fanout, "subgrouping", subgrouping, "nonterminals"
              , nonterminal_labeling.__name__, "reorder children", reorder)
        print()

        scorer = run_experiment(rec_part_strategy(direction, subgrouping, fanout, binarize), nonterminal_labeling, exp=exp
                                , reorder_children=reorder, binarize=binarize)
        scorers.append(scorer)

        exp += 1

    best_scorer = max(scorers, key=lambda s: s.labeled_frame_scorer.fmeasure())
    print()
    print("Best labeled frame F1 of", best_scorer.labeled_frame_scorer.fmeasure()
          , "in experiment", scorers.index(best_scorer) + start_exp)


def main2():
    induction_settings = InductionSettings()

    # terminal labeling
    induction_settings.terminal_labeling_token = PosTerminals()
    def term_labeling(token):
        if isinstance(token, ConstituentTerminal):
            return induction_settings.terminal_labeling_token.token_label(token)
        else:
            return token

    induction_settings.terminal_labeling = term_labeling

    # recursive partitioning
    def rec_part_strategy(direction, subgrouping, fanout, binarize):
        if direction == "right-to-left":
            return lambda dsg: fanout_limited_partitioning(dsg.recursive_partitioning(subgrouping, weak=binarize), fanout)
        else:
            return lambda dsg: fanout_limited_partitioning_left_to_right(dsg.recursive_partitioning(subgrouping
                                                                                                    , weak=binarize),
                                                                         fanout)

    induction_settings.binarize = True
    induction_settings.direction = "left-to-right"
    induction_settings.subgrouping = False
    induction_settings.fanout = 1
    induction_settings.rec_part_strategy = rec_part_strategy(induction_settings.direction
                                                             , induction_settings.subgrouping
                                                             , induction_settings.fanout
                                                             , induction_settings.binarize)

    # Nonterminal Labeling
    induction_settings.start = "START"
    def label_edge(edge):
        if isinstance(edge.label, ConstituentTerminal):
            return edge.label.pos()
        else:
            return edge.label

    def stupid_edge(edge):
        return "X"

    def label_child(edge, j):
        return edge.get_function(j)

    def simple_nonterminal_labeling(nodes, dsg):
        return simple_labeling(nodes, dsg, label_edge)

    def bot_stupid_nonterminal_labeling(nodes, dsg):
        return top_bot_labeling(nodes, dsg, label_edge, stupid_edge)

    def missing_child_nonterminal_labeling(nodes, dsg):
        return missing_child_labeling(nodes, dsg, label_edge, label_child)

    induction_settings.nonterminal_labeling = simple_nonterminal_labeling
    induction_settings.normalize = True

    experiment = DOGExperiment(induction_settings)

    # Corpora
    start = 1
    stop = 2000

    test_start = 7001
    test_stop = 7200

    # path = "res/tiger/tiger_release_aug07.corrected.16012013.utf8.xml"
    corpus_path = "res/tiger/tiger_8000.xml"
    exclude = []

    experiment.resources[TRAINING] = CorpusFile(corpus_path, start, stop)
    experiment.resources[TESTING] = CorpusFile(corpus_path, test_start, test_stop)
    experiment.oracle_parsing = True
    experiment.purge_rule_freq = None  # 1.0
    experiment.k_best = 100
    experiment.run_experiment()


if __name__ == "__main__":
    main2()
