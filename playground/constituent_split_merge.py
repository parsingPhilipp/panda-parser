from __future__ import print_function
from corpora.tiger_parse import sentence_names_to_hybridtrees
from corpora.negra_parse import hybridtrees_to_sentence_names
from grammar.induction.terminal_labeling import FormPosTerminalsUnk, FormTerminalsUnk, FormTerminalsPOS, PosTerminals, TerminalLabeling, FeatureTerminals, FrequencyBiasedTerminalLabeling
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory
from constituent.induction import fringe_extract_lcfrs, token_to_features
from constituent.construct_morph_annotation import build_nont_splits_dict, pos_cat_feats, pos_cat_and_lex_in_unary
from constituent.parse_accuracy import ParseAccuracyPenalizeFailures
from constituent.dummy_tree import dummy_constituent_tree
from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
import copy
import os
import subprocess
from sys import stdout
from collections import defaultdict
from hybridtree.constituent_tree import ConstituentTree
from hybridtree.monadic_tokens import construct_constituent_token, ConstituentCategory
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
from parser.sDCPevaluation.evaluator import The_DCP_evaluator, dcp_to_hybridtree
from parser.trace_manager.sm_trainer import build_PyLatentAnnotation
from experiment_helpers import ScoringExperiment, CorpusFile, ScorerResource, RESULT, TRAINING, TESTING, VALIDATION, \
    SplitMergeExperiment
from constituent.discodop_adapter import TreeComparator as DiscoDopScorer
import tempfile
import sys
from functools32 import lru_cache
from ast import literal_eval
if sys.version_info < (3,):
    reload(sys)
    sys.setdefaultencoding('utf8')
# import codecs
# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
# sys.stderr = codecs.getwriter('utf8')(sys.stderr)

grammar_path = '/tmp/constituent_grammar.pkl'
reduct_path = '/tmp/constituent_grammar_reduct.pkl'
terminal_labeling_path = '/tmp/constituent_labeling.pkl'
train_limit = 1000  # 2000
train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train5k/train5k.German.gold.xml'
# train_limit = 40474
# train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train/train.German.gold.xml'
train_exclude = [7561, 17632, 46234, 50224]
train_corpus = None


validation_start = 40475
validation_size = validation_start + 200 #4999
validation_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'

test_start = validation_size # 40475
test_limit = test_start + 200 # 4999
test_exclude = train_exclude
test_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'

# if not os.path.isfile(terminal_labeling_path):
#     terminal_labeling = FormPosTerminalsUnk(get_train_corpus(), 10)
#     pickle.dump(terminal_labeling, open(terminal_labeling_path, "wb"))
# else:
#     terminal_labeling = pickle.load(open(terminal_labeling_path, "rb"))

# terminal_labeling = PosTerminals()

# terminal_labeling = FeatureTerminals(token_to_features, feature_filter=lambda x: pos_cat_and_lex_in_unary(x, no_function=True))

fine_terminal_labeling = FeatureTerminals(token_to_features, feature_filter=lambda x: pos_cat_and_lex_in_unary(x, no_function=True))
fallback_terminal_labeling = PosTerminals()


def terminal_labeling(corpus):
    return FrequencyBiasedTerminalLabeling(fine_terminal_labeling, fallback_terminal_labeling, corpus, 5.0)

fanout = 2
recursive_partitioning = the_recursive_partitioning_factory().getPartitioning('fanout-' + str(fanout) + '-left-to-right')[0]

max_length = 5000
em_epochs = 6
em_epochs_sm = 20
seed = 1
merge_percentage = 50.0
sm_cycles = 3
threads = 1  # 0
smoothing_factor = 0.01
split_randomization = 2.0

validationMethod = "F1"
validationDropIterations = 6

k_best = 500

# parsing_method = "single-best-annotation"
parsing_method = "filter-ctf"
parse_results_prefix = "/tmp"
parse_results = "results"
parse_results_suffix = ".export"
NEGRA = "NEGRA"


class InductionSettings:
    def __init__(self):
        self.recursive_partitioning = None
        self.terminal_labeling = None
        self.isolate_pos = False
        self.naming_scheme = 'child'
        self.disconnect_punctuation = True
        self.normalize = False
        self.feature_la = False
        self.feat_function = lambda x: pos_cat_and_lex_in_unary(x, hmarkov=1)

    def __str__(self):
        attributes = [("recursive partitioning", self.recursive_partitioning.__name__)
                      , ("terminal labeling", self.terminal_labeling)
                      , ("isolate POS", self.isolate_pos)
                      , ("naming scheme", self.naming_scheme)
                      , ("disconnect punctuation", self.disconnect_punctuation)
                      , ("normalize corpus", self.normalize)
                      , ("feat_function", self.feat_function)]
        return '\n'.join([a[0] + ' : ' + str(a[1]) for a in attributes])


class ConstituentScorer(ScorerResource):
    def __init__(self):
        super(ConstituentScorer, self).__init__()
        self.scorer = ParseAccuracyPenalizeFailures()

    def score(self, system, gold):
        self.scorer.add_accuracy(system.labelled_spans(), gold.labelled_spans())

    def failure(self, gold):
        self.scorer.add_failure(gold.labelled_spans())


class ScorerAndWriter(ConstituentScorer, CorpusFile):
    def __init__(self, experiment, path=None):
        ConstituentScorer.__init__(self)
        path = tempfile.mktemp() if path is None else path
        CorpusFile.__init__(self, path=path)
        self.experiment = experiment
        self.reference = CorpusFile()

    def init(self):
        CorpusFile.init(self)
        self.reference.init()

    def finalize(self):
        CorpusFile.finalize(self)
        self.reference.finalize()
        print('Wrote results to', self.path)
        print('Wrote reference to', self.reference.path)

    def score(self, system, gold):
        ConstituentScorer.score(self, system, gold)
        self.file.writelines(self.experiment.serialize(system))
        self.reference.file.writelines(self.experiment.serialize(gold))

    def failure(self, gold):
        ConstituentScorer.failure(self, gold)
        sentence = self.experiment.obtain_sentence(gold)
        label = self.experiment.obtain_label(gold)
        fallback = self.experiment.compute_fallback(sentence, label)
        self.file.writelines(self.experiment.serialize(fallback))
        self.reference.file.writelines(self.experiment.serialize(gold))

    def __str__(self):
        return CorpusFile.__str__(self)


class ConstituentExperiment(ScoringExperiment, SplitMergeExperiment):
    def __init__(self, induction_settings):
        """
        :type induction_settings: InductionSettings
        """
        ScoringExperiment.__init__(self)
        SplitMergeExperiment.__init__(self)
        self.induction_settings = induction_settings
        self.resources[RESULT] = ScorerAndWriter(self)
        self.k_best = 50
        self.serialization_type = NEGRA
        self.use_output_counter = True
        self.output_counter = 0
        self.strip_vroot = True
        self.terminal_labeling = induction_settings.terminal_labeling

        self.discodop_scorer = DiscoDopScorer()
        self.max_score = 100.0
        self.rule_smooth_list = None
        if self.induction_settings.feature_la:
            self.feature_log = defaultdict(lambda: 0)

    def induce_from(self, tree):
        if not tree.complete() or tree.empty_fringe():
            return None, None
        part = self.induction_settings.recursive_partitioning(tree)
        if self.induction_settings.feature_la:
            features = defaultdict(lambda: 0)
        else:
            features = None
        tree_grammar = fringe_extract_lcfrs(tree, part, naming=self.induction_settings.naming_scheme,
                                            term_labeling=self.induction_settings.terminal_labeling,
                                            isolate_pos=self.induction_settings.isolate_pos,
                                            feature_logging=features)
        return tree_grammar, features

    def parsing_postprocess(self, sentence, derivation, label=None):
        full_yield, id_yield, full_token_yield, token_yield = sentence

        dcp_tree = ConstituentTree(label)
        punctuation_positions = [i + 1 for i, idx in enumerate(full_yield)
                                 if idx not in id_yield]

        cleaned_tokens = copy.deepcopy(full_token_yield)
        dcp = The_DCP_evaluator(derivation).getEvaluation()
        dcp_to_hybridtree(dcp_tree, dcp, cleaned_tokens, False, construct_constituent_token,
                          punct_positions=punctuation_positions)

        if self.strip_vroot:
            dcp_tree.strip_vroot()

        return dcp_tree

    def obtain_sentence(self, hybrid_tree):
        sentence = hybrid_tree.full_yield(), hybrid_tree.id_yield(), \
                   hybrid_tree.full_token_yield(), hybrid_tree.token_yield()
        return sentence

    def obtain_label(self, hybrid_tree):
        return hybrid_tree.sent_label()

    def compute_fallback(self, sentence, label=None):
        full_yield, id_yield, full_token_yield, token_yield = sentence
        return dummy_constituent_tree(token_yield, full_token_yield, 'NP', 'S', label)

    def read_corpus(self, resource):
        path = resource.path
        prefix = 's'
        if self.induction_settings.normalize:
            path = self.normalize_corpus(path, src='tigerxml', dest='tigerxml', renumber=False)
            prefix = ''

        return sentence_names_to_hybridtrees(
            [prefix + str(i) for i in range(resource.start, resource.end + 1) if i not in resource.exclude]
            , path
            , hold=False
            , disconnect_punctuation=self.induction_settings.disconnect_punctuation)

    def initialize_parser(self):
        self.parser = GFParser_k_best(grammar=self.base_grammar, k=self.k_best)

    def parsing_preprocess(self, hybrid_tree):
        if self.strip_vroot:
            hybrid_tree.strip_vroot()
        return self.terminal_labeling.prepare_parser_input(hybrid_tree.token_yield())

    @lru_cache(maxsize=500)
    def normalize_corpus(self, path, src='export', dest='export', renumber=True):
        first_stage = tempfile.mktemp(suffix=".export")
        subprocess.call(["treetools", "transform", path, first_stage, "--trans", "root_attach",
                         "--src-format", src, "--dest-format", "export"])
        second_stage = tempfile.mktemp(suffix=".export")
        second_call = ["discodop", "treetransforms"]
        if renumber:
            second_call.append("--renumber")
        subprocess.call(second_call + ["--punct=move", first_stage, second_stage,
                         "--inputfmt=export", "--outputfmt=export"])
        if dest == 'export':
            return second_stage
        elif dest == 'tigerxml':
            third_stage = tempfile.mktemp(suffix=".xml")
            subprocess.call(["treetools", "transform", second_stage, third_stage,
                             "--src-format", "export", "--dest-format", dest])
            return third_stage

    def evaluate(self, result_resource, gold_resource):
        accuracy = result_resource.scorer
        print('')
        # print('Parsed:', n)
        if accuracy.n() > 0:
            print('Recall:   ', accuracy.recall())
            print('Precision:', accuracy.precision())
            print('F-measure:', accuracy.fmeasure())
            print('Parse failures:', accuracy.n_failures())
        else:
            print('No successful parsing')
        # print('time:', end_at - start_at)
        print('')

        print('normalize results with treetools and discodop')

        ref_rn = self.normalize_corpus(result_resource.reference.path)
        sys_rn = self.normalize_corpus(result_resource.path)
        prm = "../util/proper.prm"

        print('running discodop evaluation on gold:', ref_rn, ' and sys:', sys_rn, "with proper.prm")
        subprocess.call(["discodop", "eval", ref_rn, sys_rn, prm])

    @staticmethod
    def __obtain_labelled_spans(obj):
        spans = obj.labelled_spans()
        spans = map(tuple, spans)
        spans = set(spans)
        return spans

    def score_object(self, obj, gold):
        # _, _, f1 = self.precision_recall_f1(self.__obtain_labelled_spans(gold), self.__obtain_labelled_spans(obj))
        f1 = self.discodop_scorer.compare_hybridtrees(gold, obj)
        return f1

    def serialize(self, obj):
        if self.serialization_type == NEGRA:
            if self.use_output_counter:
                self.output_counter += 1
                number = self.output_counter
            else:
                number = int(self.obtain_label(obj)[1:])
            return hybridtrees_to_sentence_names([obj], number, max_length)
        else:
            assert False

    def print_config(self, file=stdout):
        ScoringExperiment.print_config(self, file=file)
        SplitMergeExperiment.print_config(self, file=file)
        print("Induction Settings {", file=file)
        print(self.induction_settings, "\n}", file=file)
        print("k-best", self.k_best)
        print("Serialization type", self.serialization_type)
        print("Output counter", self.use_output_counter, "start", self.output_counter)
        print("VROOT stripping", self.strip_vroot)

    def compute_reducts(self, resource):
        training_corpus = self.read_corpus(resource)
        parser = self.organizer.training_reducts.get_parser() if self.organizer.training_reducts is not None else None
        nonterminal_map = self.organizer.nonterminal_map
        return compute_reducts(self.base_grammar, training_corpus, self.induction_settings.terminal_labeling,
                               parser=parser, nont_map=nonterminal_map)

    def create_initial_la(self):
        if self.induction_settings.feature_la:
            print("building initial LA from features")
            nonterminal_splits, rootWeights, ruleWeights, split_id = build_nont_splits_dict(self.base_grammar, self.feature_log, self.organizer.nonterminal_map, feat_function=self.induction_settings.feat_function)
            print("number of nonterminals:", len(nonterminal_splits))
            print("total splits", sum(nonterminal_splits))
            max_splits = max(nonterminal_splits)
            max_splits_index = nonterminal_splits.index(max_splits)
            max_splits_nont = self.organizer.nonterminal_map.index_object(max_splits_index)
            print("max. nonterminal splits", max_splits, "at index ", max_splits_index, "i.e.,", max_splits_nont)
            for key in split_id[max_splits_nont]:
                print(key)
            print("number of rules", len(ruleWeights))
            print("total split rules", sum(map(len, ruleWeights)))
            print("number of split rules with 0 prob.", sum(map(sum, map(lambda xs: map(lambda x: 1 if x == 0.0 else 0, xs), ruleWeights))))

            la = build_PyLatentAnnotation(nonterminal_splits, rootWeights, ruleWeights, self.organizer.grammarInfo,
                                          self.organizer.storageManager)
            la.add_random_noise(self.organizer.grammarInfo, seed=self.organizer.seed)
            self.split_id = split_id
            return la
        else:
            return super(ConstituentExperiment, self).create_initial_la()

    def do_em_training(self):
        super(ConstituentExperiment, self).do_em_training()
        if self.induction_settings.feature_la:
            self.patch_initial_grammar()

    def custom_sm_options(self, builder):
        if self.rule_smooth_list is not None:
            builder.set_count_smoothing(self.rule_smooth_list, 0.5)

    def patch_initial_grammar(self):
        print("Merging feature splits with SCC merger.")
        mergedLa = self.organizer.emTrainer.merge(self.organizer.latent_annotations[0])
        if False:
            self.organizer.latent_annotations[0] = mergedLa
            self.organizer.merge_sources[0] = self.organizer.emTrainer.get_current_merge_sources()
            print(self.organizer.merge_sources[0])

        else:
            splits, _, _ = mergedLa.serialize()
            merge_sources = self.organizer.emTrainer.get_current_merge_sources()

            lookup = self.print_funky_listing(merge_sources)

            fine_grammar_merge_sources = []
            for nont_idx in range(0, self.organizer.nonterminal_map.get_counter()):
                nont = self.organizer.nonterminal_map.index_object(nont_idx)
                if any([rule.rhs() == [] for rule in self.base_grammar.lhs_nont_to_rules(nont)]):
                    fine_grammar_merge_sources.append([[split] for split in range(0, splits[nont_idx])])
                else:
                    fine_grammar_merge_sources.append([[split for split in range(0, splits[nont_idx])]])

            print("Projecting to fine grammar LA")
            fine_grammar_LA = mergedLa.project_annotation_by_merging(self.organizer.grammarInfo, fine_grammar_merge_sources)

            def arg_transform(arg, la):
                arg_mod = []
                for elem in arg:
                    if isinstance(elem, str):
                        arg_mod.append(elem + "-group-" + str(la[0]))
                    else:
                        arg_mod.append(elem)
                return arg_mod

            def smooth_transform(arg):
                arg_mod = []
                for elem in arg:
                    if isinstance(elem, str):
                        try:
                            term = literal_eval(elem)
                            if isinstance(term, tuple):
                                pos = dict(term[0]).get("pos", "UNK")
                                arg_mod.append(pos)
                                # print(term, pos)
                            else:
                                arg_mod.append(elem)
                        except ValueError:
                            arg_mod.append(elem)
                    else:
                        arg_mod.append(elem)
                return arg_mod

            def id_arg(arg, la):
                return arg

            print("Constructing fine grammar")
            grammar_fine, grammar_fine_LA_full, grammar_fine_info, grammar_fine_nonterminal_map, nont_translation, smooth_rules \
                = fine_grammar_LA.construct_fine_grammar(self.base_grammar, self.organizer.grammarInfo, id_arg,
                                                         mergedLa, smooth_transform=smooth_transform)

            self.rule_smooth_list = smooth_rules

            grammar_fine.make_proper()
            grammar_fine_LA_full.make_proper(grammar_fine_info)
            print(grammar_fine_LA_full.is_proper(grammar_fine_info))
            nonterminal_splits, rootWeights, ruleWeights = grammar_fine_LA_full.serialize()

            # for rule in grammar_fine.rules():
            #     print(rule, ruleWeights[rule.get_idx()])
            print("number of nonterminals:", len(nonterminal_splits))
            print("total splits", sum(nonterminal_splits))
            print("number of rules", len(ruleWeights))
            print("total split rules", sum(map(len, ruleWeights)))
            print("number of split rules with 0 prob.",
                  sum(map(sum, map(lambda xs: map(lambda x: 1 if x == 0.0 else 0, xs), ruleWeights))))
            self.base_grammar_backup = self.base_grammar
            self.base_grammar = grammar_fine

            self.organizer.grammarInfo = grammar_fine_info
            self.organizer.nonterminal_map = grammar_fine_nonterminal_map

            self.organizer.last_sm_cycle = 0
            if True:
                self.organizer.latent_annotations[0] = grammar_fine_LA_full
            else:
                self.organizer.latent_annotations[0] = super(ConstituentExperiment, self).create_initial_la()
            self.organizer.training_reducts = None

            print("Recomputing reducts")
            self.organizer.training_reducts = self.compute_reducts(self.resources[TRAINING])
            # self.initialize_training_environment()
            # self.organizer.last_sm_cycle = 0
            # self.organizer.latent_annotations[0] = super(ConstituentExperiment, self).create_initial_la()

            # raise Exception("No text")

    def print_funky_listing(self, merge_sources):
        lookup = {}

        for nont_idx in range(0, self.organizer.nonterminal_map.get_counter()):
            nont = self.organizer.nonterminal_map.index_object(nont_idx)
            term = None
            if any([rule.rhs() == [] for rule in self.base_grammar.lhs_nont_to_rules(nont)]):
                print(nont)
                for rule in self.base_grammar.lhs_nont_to_rules(nont):
                    print(rule)
                    assert len(rule.lhs().args()) == 1 and len(rule.lhs().args()[0]) == 1
                    # rule_term = rule.lhs().args()[0][0]
                    # assert rule_term is not None
                    # if term is None:
                    #     term = rule_term
                    # else:
                    #     if term != rule_term:
                    #         print(term, rule_term)
                    #     assert term == rule_term
                    if rule.rhs() != []:
                        raise Exception("this is bad!")
                lookup[nont] = {}
                # print(merge_sources[nont_idx])
                # print(self.split_id[nont])
                for group, sources in enumerate(merge_sources[nont_idx]):
                    print("group", group)
                    for source in sources:
                        for key in self.split_id[nont]:
                            if self.split_id[nont][key] - 1 == source:
                                print("\t", key)
                                lookup[nont][frozenset(key[0])] = group
                                continue
                # print("lookup")
                # for key in lookup[nont]:
                #     print(key, lookup[nont][key])
        return lookup

    def patch_terminal_labeling(self, lookup):
        this_class = self

        class PatchedTerminalLabeling(TerminalLabeling):
            def __init__(self, other, lookup):
                self.other = other
                self.lookup = lookup

            def token_label(self, token):
                other_label = self.other.token_label(token)
                feat_list = token_to_features(token)
                features = this_class.induction_settings.feat_function([feat_list])
                feature_set = frozenset(features[0])
                group_idx = self.lookup.get(feature_set, 0)

                return other_label + "-group-" + str(group_idx)

        class PatchedTerminalLabeling2(TerminalLabeling):
            def __init__(self, other, lookup):
                self.other = other
                self.lookup = lookup

            def token_label(self, token):
                other_label = self.other.token_label(token)
                feat_list = token_to_features(token)
                features = this_class.induction_settings.feat_function([feat_list])
                feature_set = frozenset(features[0])
                if token.pos() + "/1" not in self.lookup:
                    return token.pos()
                if feature_set in self.lookup[token.pos() + "/1"]:
                    return other_label
                else:
                    return token.pos()

        self.terminal_labeling = PatchedTerminalLabeling2(self.induction_settings.terminal_labeling, lookup)






def main3():
    induction_settings = InductionSettings()
    induction_settings.recursive_partitioning = recursive_partitioning
    induction_settings.normalize = False
    induction_settings.disconnect_punctuation = True
    induction_settings.naming_scheme = 'child'
    induction_settings.isolate_pos = True
    induction_settings.feature_la = True
    experiment = ConstituentExperiment(induction_settings)
    experiment.organizer.seed = 2
    experiment.organizer.em_epochs = em_epochs
    experiment.organizer.em_epochs_sm = em_epochs_sm
    experiment.organizer.validator_type = "SIMPLE"
    experiment.organizer.max_sm_cycles = sm_cycles
    experiment.organizer.refresh_score_validator = True
    experiment.organizer.project_weights_before_parsing = False
    experiment.organizer.disable_split_merge = False
    experiment.organizer.disable_em = False
    experiment.organizer.merge_percentage = 60.0
    experiment.organizer.merge_type = "PERCENT"
    experiment.organizer.merge_threshold = -3.0
    experiment.resources[TRAINING] = CorpusFile(path=train_path, start=1, end=train_limit, exclude=train_exclude)
    experiment.resources[VALIDATION] = CorpusFile(path=validation_path, start=validation_start, end=validation_size
                                                  , exclude=train_exclude)
    experiment.resources[TESTING] = CorpusFile(path=test_path, start=test_start,
                                               end=test_limit, exclude=train_exclude)
    experiment.oracle_parsing = False
    experiment.k_best = k_best
    experiment.purge_rule_freq = None
    induction_settings.terminal_labeling = terminal_labeling(experiment.read_corpus(experiment.resources[TRAINING]))
    experiment.terminal_labeling = induction_settings.terminal_labeling
    experiment.run_experiment()

if __name__ == '__main__':
    main3()
