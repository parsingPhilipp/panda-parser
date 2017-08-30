from __future__ import print_function
from corpora.tiger_parse import sentence_names_to_hybridtrees
from corpora.negra_parse import hybridtrees_to_sentence_names
from grammar.induction.terminal_labeling import FormPosTerminalsUnk
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory
from grammar.lcfrs import LCFRS
from constituent.induction import fringe_extract_lcfrs
from constituent.parse_accuracy import ParseAccuracyPenalizeFailures
from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
from parser.coarse_to_fine_parser.coarse_to_fine import Coarse_to_fine_parser
import time
import copy
import os
import pickle
import heapq
import random
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.monadic_tokens import construct_constituent_token, ConstituentCategory
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
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

grammar_path = '/tmp/constituent_grammar.pkl'
reduct_path = '/tmp/constituent_grammar_reduct.pkl'
terminal_labeling_path = '/tmp/constituent_labeling.pkl'
# train_limit = 5000
# train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train5k/train5k.German.gold.xml'
train_limit = 200
train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train/train.German.gold.xml'
train_exclude = [7561,17632,46234,50224]
train_corpus = None

def get_train_corpus():
    global train_corpus
    if train_corpus is None:
        train_corpus = build_corpus(train_path, 1, train_limit, train_exclude)
    return train_corpus
validation_start = 40475
validation_size = validation_start + 100
validation_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'
validation_corpus = build_corpus(validation_path, validation_start, validation_size, train_exclude)

test_start = 40475
test_limit = test_start + 100
test_exclude = train_exclude
test_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'
test_corpus = build_corpus(test_path, test_start, test_limit, test_exclude)

if not os.path.isfile(terminal_labeling_path):
    terminal_labeling = FormPosTerminalsUnk(get_train_corpus(), 20)
    pickle.dump(terminal_labeling, open(terminal_labeling_path, "wb"))
else:
    terminal_labeling = pickle.load(open(terminal_labeling_path, "rb"))
recursive_partitioning = the_recursive_partitioning_factory().getPartitioning('fanout-1-left-to-right')[0]

max_length = 2000
em_epochs = 5
seed = 0
merge_percentage = 50.0
sm_cycles = 2
threads = 10
smoothing_factor = 0.05
split_randomization = 5.0

scc_merger_threshold = -0.2


genetic_initial = 2
genetic_population = 3
genetic_cycles = 2

validationMethod = "F1"
validationDropIterations = 6

k_best = 200

# parsing_method = "single-best-annotation"
parsing_method = "filter-ctf"
parse_results_prefix = "/tmp"
parse_results = "results"
parse_results_suffix = ".export"


def dummy_constituent_tree(token_yield, full_yield, dummy_label, dummy_root):
    """
    generates a dummy tree for a given yield using 'S' as inner node symbol
    :param token_yield: connected yield of a parse tree
    :type: list of ConstituentTerminal
    :param full_yield: full yield of the parse tree
    :type: list of ConstituentTerminal
    :return: parse tree
    :rtype: ConstituentTree
    """
    tree = ConstituentTree()


    # create all leaves and punctuation
    for token in full_yield:
        if token not in token_yield:
            tree.add_punct(full_yield.index(token), token.pos(), token.form())
        else:
            tree.add_leaf(full_yield.index(token), token.pos(), token.form())

    # generate root node
    root_id = 'n0'
    tree.add_node(root_id, ConstituentCategory(dummy_root))
    tree.add_to_root(root_id)

    parent = root_id

    if len(token_yield) > 1:
        i = 1
        # generate inner nodes of branching tree
        for token in token_yield[:-2]:
            node = ConstituentCategory(str(dummy_label))
            tree.add_node('n' + str(i), node)
            tree.add_child(parent, 'n' + str(i))
            tree.add_child(parent, full_yield.index(token))
            parent = 'n' + str(i)
            i+=1

        token = token_yield[len(token_yield) - 2]
        tree.add_child(parent, full_yield.index(token))
        token = token_yield[len(token_yield) - 1]
        tree.add_child(parent, full_yield.index(token))
    elif len(token_yield) == 1:
        tree.add_child(parent, full_yield.index(token_yield[0]))

    return tree


def do_parsing(parser):
    accuracy = ParseAccuracyPenalizeFailures()
    system_trees = []

    start_at = time.time()

    n = 0

    for tree in test_corpus:

        if not tree.complete() \
                or tree.empty_fringe() \
                or not 0 < len(tree.word_yield()) <= max_length:
            continue

        parser.set_input(terminal_labeling.prepare_parser_input(tree.token_yield()))
        parser.parse()
        if not parser.recognized():
            relevant = tree.labelled_spans()
            accuracy.add_failure(relevant)

            system_trees.append(dummy_constituent_tree(tree.token_yield(), tree.full_token_yield(), "NP", "S"))
            # print('failure', tree.sent_label()) # for testing
        else:
            n += 1
            dcp_tree = ConstituentTree()
            punctuation_positions = [i + 1 for i, idx in enumerate(tree.full_yield()) if idx not in tree.id_yield()]
            dcp_tree = parser.dcp_hybrid_tree_best_derivation(
                dcp_tree
                , tree.full_token_yield()
                , False
                , construct_constituent_token
                , punctuation_positions=punctuation_positions)

            retrieved = dcp_tree.labelled_spans()
            relevant = tree.labelled_spans()
            accuracy.add_accuracy(retrieved, relevant)

            system_trees.append(dcp_tree)

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

    name = parse_results
    # do not overwrite existing result files
    i = 1
    while os.path.isfile(os.path.join(parse_results_prefix, name + parse_results_suffix)):
        i += 1
        name = parse_results + '_' + str(i)

    path = os.path.join(parse_results_prefix, name + parse_results_suffix)

    with open(path, 'w') as result_file:
        print('Exporting parse trees of length <=', max_length, 'to', str(path))
        map(lambda x: x.strip_vroot(), system_trees)
        result_file.writelines(hybridtrees_to_sentence_names(system_trees, test_start, max_length))

    return accuracy

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
        # print(tree_count, scores)
        parser.clear()

    print("trees used for validation ", tree_count, "with", der_count * 1.0 / tree_count, "derivations on average")

    return validator

def main():
    # induce or load grammar
    if not os.path.isfile(grammar_path):
        grammar = LCFRS('START')
        for tree in get_train_corpus():
            if not tree.complete() or tree.empty_fringe():
                continue
            part = recursive_partitioning(tree)
            tree_grammar = fringe_extract_lcfrs(tree, part, naming='child', term_labeling=terminal_labeling)
            grammar.add_gram(tree_grammar)
        grammar.make_proper()
        pickle.dump(grammar, open(grammar_path, 'wb'))
    else:
        grammar = pickle.load(open(grammar_path, 'rb'))

    # compute or load reducts
    if not os.path.isfile(reduct_path):
        trace = compute_reducts(grammar, get_train_corpus(), terminal_labeling)
        trace.serialize(reduct_path)
    else:
        trace = PySDCPTraceManager(grammar, terminal_labeling)
        trace.load_traces_from_file(reduct_path)

    global train_corpus
    if train_corpus is not None:
        del train_corpus
    # prepare EM training
    grammarInfo = PyGrammarInfo(grammar, trace.get_nonterminal_map())
    storageManager = PyStorageManager()

    em_builder = PySplitMergeTrainerBuilder(trace, grammarInfo)
    em_builder.set_em_epochs(em_epochs)
    em_builder.set_simple_expector(threads=threads)
    emTrainer = em_builder.build()

    # randomize initial weights and do em training
    la_no_splits = build_PyLatentAnnotation_initial(grammar, grammarInfo, storageManager)
    la_no_splits.add_random_noise(grammarInfo, seed=seed)
    emTrainer.em_train(la_no_splits)
    la_no_splits.project_weights(grammar, grammarInfo)

    # emTrainerOld = PyEMTrainer(trace)
    # emTrainerOld.em_training(grammar, 30, "rfe", tie_breaking=True)

    global validation_corpus
    # compute parses for validation set
    baseline_parser = GFParser_k_best(grammar, k=k_best)
    validator = build_score_validator(grammar, grammarInfo, trace.get_nonterminal_map(), storageManager,
                                       terminal_labeling, baseline_parser, validation_corpus, validationMethod)
    del baseline_parser
    del validation_corpus

    # prepare SM training
    builder = PySplitMergeTrainerBuilder(trace, grammarInfo)
    builder.set_em_epochs(em_epochs)
    builder.set_split_randomization(1.0, seed + 1)
    builder.set_simple_expector(threads=threads)
    builder.set_score_validator(validator, validationDropIterations)
    builder.set_smoothing_factor(smoothingFactor=smoothing_factor)
    builder.set_split_randomization(percent=split_randomization)
    splitMergeTrainer = builder.set_scc_merger(threshold=scc_merger_threshold, threads=threads).build()

    splitMergeTrainer.setMaxDrops(validationDropIterations, mode="smoothing")
    splitMergeTrainer.setEMepochs(em_epochs, mode="smoothing")

    # set initial latent annotation
    latentAnnotations = []
    for i in range(0, genetic_initial):
        splitMergeTrainer.reset_random_seed(seed + i + 1)
        la = splitMergeTrainer.split_merge_cycle(la_no_splits)
        if not la.check_for_validity():
            print('[Genetic] Initial LA', i, 'is not consistent! (See details before)')
        if not la.is_proper(grammarInfo):
            print('[Genetic] Initial LA', i, 'is not proper!')
        heapq.heappush(latentAnnotations, (evaluate_la(grammar, grammarInfo, la, trace), la))

    print("[Genetic] Started with an Annotation of F-Score of ", min(latentAnnotations)[0])

    random.seed(seed)
    for round in range(1, genetic_cycles + 1):
        print("[Genetic] Starting Recombination Round ", round)
        # newpopulation = list(latentAnnotations)
        newpopulation = []
        # Cross all candidates!
        for leftIndex in range(0, len(latentAnnotations)):
            for rightIndex in range(leftIndex+1, len(latentAnnotations)):
                left = latentAnnotations[leftIndex][1]
                right = latentAnnotations[rightIndex][1]
                # TODO: How to determine NTs to keep?
                keepFromOne = []
                for i in range(0, len(grammar.nonts())):
                    keepFromOne.append(random.choice([True, False]))

                la = left.genetic_recombination(right, grammarInfo, keepFromOne, 0.00000001, 300)
                if not la.check_for_validity():
                    print('[Genetic] LA is not consistent! (See details before)')
                if not la.is_proper(grammarInfo):
                    print('[Genetic] LA is not proper!')

                # do SM-Training on recombined LAs
                la = splitMergeTrainer.split_merge_cycle(la)
                if not la.check_for_validity():
                    print('[Genetic] Split/Merge introduced inconsistencys into LA! (See details before)')
                if not la.is_proper(grammarInfo):
                    print('[Genetic] Split/Merge introduced problems with properness!')

                fscore = evaluate_la(grammar, grammarInfo, la, trace)
                print("[Genetic] Genetic combination yields (F-score): ", fscore)
                heapq.heappush(newpopulation, (fscore, la))
        heapq.heapify(newpopulation)
        latentAnnotations = heapq.nsmallest(genetic_population, heapq.merge(latentAnnotations, newpopulation))
        heapq.heapify(latentAnnotations)
        print("[Genetic] Best annotation has F-Score of ", min(latentAnnotations)[0])


def evaluate_la(grammar, grammarInfo, la, trace):
    parser = Coarse_to_fine_parser(grammar, GFParser_k_best, la, grammarInfo,
                                   trace.get_nonterminal_map(), k=k_best)
    accuracy = do_parsing(parser)
    return - accuracy.fmeasure()


def main2():
     name = parse_results
     # do not overwrite existing result files
     i = 1
     while os.path.isfile(os.path.join(parse_results_prefix, name + parse_results_suffix)):
         i += 1
         name = parse_results + '_' + str(i)

     path = os.path.join(parse_results_prefix, name + parse_results_suffix)

     corpus = copy.deepcopy(test_corpus)
     map(lambda x: x.strip_vroot(), corpus)

     with open(path, 'w') as result_file:
         print('Exporting parse trees of length <=', max_length, 'to', str(path))
         result_file.writelines(hybridtrees_to_sentence_names(corpus, test_start, max_length))


if __name__ == '__main__':
    main()
