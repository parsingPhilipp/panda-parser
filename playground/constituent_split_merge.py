from __future__ import print_function
from corpora.tiger_parse import sentence_names_to_hybridtrees
from corpora.negra_parse import hybridtrees_to_sentence_names
from grammar.induction.terminal_labeling import FormPosTerminalsUnk, FormTerminalsUnk, FormTerminalsPOS, PosTerminals
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory
from constituent.induction import fringe_extract_lcfrs
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
if sys.version_info < (3,):
    reload(sys)
    sys.setdefaultencoding('utf8')
# import codecs
# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
# sys.stderr = codecs.getwriter('utf8')(sys.stderr)

grammar_path = '/tmp/constituent_grammar.pkl'
reduct_path = '/tmp/constituent_grammar_reduct.pkl'
terminal_labeling_path = '/tmp/constituent_labeling.pkl'
train_limit = 5000  # 2000
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
terminal_labeling = PosTerminals()
fanout = 1
recursive_partitioning = the_recursive_partitioning_factory().getPartitioning('fanout-' + str(fanout) + '-left-to-right')[0]

max_length = 2000
em_epochs = 30
seed = 0
merge_percentage = 50.0
sm_cycles = 4
threads = 10
smoothing_factor = 0.05
split_randomization = 5.0

validationMethod = "F1"
validationDropIterations = 6

k_best = 200

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
        self.feat_function = pos_cat_and_lex_in_unary

    def __str__(self):
        attributes = [("recursive partitioning", self.recursive_partitioning.__name__)
                      , ("terminal labeling", self.terminal_labeling)
                      , ("isolate POS", self.isolate_pos)
                      , ("naming scheme", self.naming_scheme)
                      , ("disconnect punctuation", self.disconnect_punctuation)
                      , ("normalize corpus", self.normalize)]
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
        return compute_reducts(self.base_grammar, training_corpus, self.induction_settings.terminal_labeling)

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
            return la
        else:
            return super(ConstituentExperiment, self).create_initial_la()

    def do_em_training(self):
        super(ConstituentExperiment, self).do_em_training()
        if self.induction_settings.feature_la:
            print("Merging feature splits with SCC merger and threshold", str(self.organizer.merge_threshold) + ".")
            mergedLa = self.organizer.emTrainer.merge(self.organizer.latent_annotations[0])
            self.organizer.latent_annotations[0] = mergedLa
            self.organizer.merge_sources[0] = self.organizer.emTrainer.get_current_merge_sources()
            print(self.organizer.merge_sources[0])


def main3():
    induction_settings = InductionSettings()
    induction_settings.recursive_partitioning = recursive_partitioning
    induction_settings.terminal_labeling = terminal_labeling
    induction_settings.normalize = True
    induction_settings.disconnect_punctuation = False
    induction_settings.feature_la = True
    experiment = ConstituentExperiment(induction_settings)
    experiment.organizer.em_epochs = em_epochs
    experiment.organizer.max_sm_cycles = sm_cycles
    experiment.organizer.refresh_score_validator = True
    experiment.organizer.disable_split_merge = True
    experiment.organizer.merge_percentage = 60.0
    experiment.organizer.merge_type = "PERCENT"
    experiment.resources[TRAINING] = CorpusFile(path=train_path, start=1, end=train_limit, exclude=train_exclude)
    experiment.resources[VALIDATION] = CorpusFile(path=validation_path, start=validation_start, end=validation_size
                                                  , exclude=train_exclude)
    experiment.resources[TESTING] = CorpusFile(path=test_path, start=test_start,
                                               end=test_limit, exclude=train_exclude)
    experiment.oracle_parsing = False
    experiment.k_best = k_best
    experiment.purge_rule_freq = None
    experiment.run_experiment()

if __name__ == '__main__':
    main3()
