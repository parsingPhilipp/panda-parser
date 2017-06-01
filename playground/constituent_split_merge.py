from __future__ import print_function
from corpora.tiger_parse import sentence_names_to_hybridtrees
from grammar.induction.terminal_labeling import FormPosTerminalsUnk
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory
from grammar.lcfrs import LCFRS
from constituent.induction import fringe_extract_lcfrs
from constituent.parse_accuracy import ParseAccuracyPenalizeFailures
from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
from parser.coarse_to_fine_parser.coarse_to_fine import Coarse_to_fine_parser
import time
import copy
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.monadic_tokens import construct_constituent_token
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts
from parser.supervised_trainer.trainer import PyDerivationManager
from parser.trace_manager.sm_trainer_util import PyStorageManager, PyGrammarInfo
from parser.trace_manager.sm_trainer import PyEMTrainer, PySplitMergeTrainerBuilder, build_PyLatentAnnotation_initial
from parser.trace_manager.score_validator import PyCandidateScoreValidator
from parser.sDCPevaluation.evaluator import The_DCP_evaluator, dcp_to_hybridtree

def build_corpus(path, start, stop, exclude):
    return sentence_names_to_hybridtrees(
        ['s' + str(i) for i in range(start, stop + 1) if i not in exclude]
        , path
        , hold=False)

train_limit = 2000
train_exclude = []
train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train5k/train5k.German.gold.xml'
train_corpus = build_corpus(train_path, 1, train_limit, train_exclude)
validation_corpus = build_corpus(train_path, train_limit + 1, train_limit + 200, [])

test_start = 40475
test_limit = test_start + 500
test_exclude = []
test_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'
test_corpus = build_corpus(test_path, test_start, test_limit, test_exclude)

terminal_labeling = FormPosTerminalsUnk(train_corpus, 20)
recursive_partitioning = the_recursive_partitioning_factory().getPartitioning('fanout-1')[0]

max_length = 20
em_epochs = 20
seed = 0
merge_percentage = 65.0
sm_cycles = 4

validationMethod = "F1"
validationDropIterations = 3

k_best = 50

# parsing_method = "single-best-annotation"
parsing_method = "filter-ctf"

def do_parsing(parser):
    accuracy = ParseAccuracyPenalizeFailures()

    start_at = time.time()

    n = 0

    for tree in test_corpus:

        if not tree.complete() \
                or tree.empty_fringe() \
                or not 2 <= len(tree.word_yield()) <= max_length:
            continue

        parser.set_input(terminal_labeling.prepare_parser_input(tree.token_yield()))
        parser.parse()
        if not parser.recognized():
            relevant = tree.labelled_spans()
            accuracy.add_failure(relevant)
            # print('failure', tree.sent_label()) # for testing
        else:
            n += 1
            dcp_tree = ConstituentTree()
            dcp_tree = parser.dcp_hybrid_tree_best_derivation(dcp_tree, tree.token_yield(), False,
                                                              construct_constituent_token)
            retrieved = dcp_tree.labelled_spans()
            relevant = tree.labelled_spans()
            accuracy.add_accuracy(retrieved, relevant)
        parser.clear()

    end_at = time.time()

    print('Parsed:', n)
    if accuracy.n() > 0:
        print('Recall:', accuracy.recall())
        print('Precision:', accuracy.precision())
        print('F-measure:', accuracy.fmeasure())
        print('Parse failures:', accuracy.n_failures())
    else:
        print('No successful parsing')
    print('time:', end_at - start_at)
    print('')


def build_score_validator(baseline_grammar, grammarInfo, nont_map, storageManager, term_labelling, parser, corpus_validation, validationMethod):
    validator = PyCandidateScoreValidator(grammarInfo, storageManager, validationMethod)

    # parser = GFParser(baseline_grammar)
    tree_count = 0
    der_count = 0
    for gold_tree in corpus_validation:
        tree_count += 1
        parser.set_input(term_labelling.prepare_parser_input(gold_tree.token_yield()))
        parser.parse()
        derivations = [der for _, der in parser.k_best_derivation_trees()]
        manager = PyDerivationManager(baseline_grammar, nont_map)
        manager.convert_hypergraphs(derivations)
        scores = []

        relevant = set([tuple(t) for t in gold_tree.labelled_spans()])

        for der in derivations:
            der_count += 1

            h_tree = ConstituentTree()
            cleaned_tokens = copy.deepcopy(gold_tree.full_token_yield())
            dcp = The_DCP_evaluator(der).getEvaluation()
            dcp_to_hybridtree(h_tree, dcp, cleaned_tokens, False, construct_constituent_token)

            retrieved = set([tuple(t) for t in h_tree.labelled_spans()])
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

            if validationMethod == "F1":
                scores.append(fmeasure)
            elif validationMethod == "Precision":
                scores.append(precision)
            elif validationMethod == "Recall":
                scores.append(recall)
            else:
                raise()

        validator.add_scored_candidates(manager, scores, 1.0 if len(relevant) > 0 else 0.0)
        print(tree_count, scores)
        parser.clear()

    print("trees used for validation ", tree_count, "with", der_count * 1.0 / tree_count, "derivations on average")

    return validator


def main():
    grammar = LCFRS('START')
    for tree in train_corpus:
        part = recursive_partitioning(tree)
        tree_grammar = fringe_extract_lcfrs(tree, part, naming='child', term_labeling=terminal_labeling)
        grammar.add_gram(tree_grammar)
    grammar.make_proper()

    # do_parsing(grammar)

    # compute reducts
    trace = compute_reducts(grammar, train_corpus, terminal_labeling)

    # do EM training
    emTrainer = PyEMTrainer(trace)
    emTrainer.em_training(grammar, n_epochs=em_epochs, init="rfe", tie_breaking=True, seed=seed)

    baseline_parser = GFParser_k_best(grammar, k=k_best)

    # prepare SM training
    grammarInfo = PyGrammarInfo(grammar, trace.get_nonterminal_map())
    storageManager = PyStorageManager()

    builder = PySplitMergeTrainerBuilder(trace, grammarInfo)
    builder.set_em_epochs(em_epochs)
    builder.set_split_randomization(1.0, seed + 1)
    builder.set_simple_expector(threads=4)


    validator = build_score_validator(grammar, grammarInfo, trace.get_nonterminal_map(), storageManager,
                                      terminal_labeling, baseline_parser, validation_corpus, validationMethod)
    builder.set_score_validator(validator, validationDropIterations)

    splitMergeTrainer = builder.set_percent_merger(merge_percentage).build()
    splitMergeTrainer.setMaxDrops(validationDropIterations, mode="smoothing")
    splitMergeTrainer.setEMepochs(em_epochs, mode="smoothing")

    # set initial latent annotation
    latentAnnotation = [build_PyLatentAnnotation_initial(grammar, grammarInfo, storageManager)]

    # do SM training
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
            parser = Coarse_to_fine_parser(grammar, GFParser_k_best, latentAnnotation[-1], grammarInfo, trace.get_nonterminal_map(), k=k_best)
        else:
            raise()
        do_parsing(parser)

if __name__ == '__main__':
    main()