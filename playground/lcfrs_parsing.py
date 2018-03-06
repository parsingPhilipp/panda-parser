from __future__ import print_function
import tempfile
import subprocess
import sys
from parser.discodop_parser.parser import DiscodopKbestParser
from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
from parser.sDCP_parser.sdcp_parser_wrapper import print_grammar
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
import plac
from constituent.induction import direct_extract_lcfrs, BasicNonterminalLabeling, NonterminalsWithFunctions, binarize, \
    direct_extract_lcfrs_from_prebinarized_corpus
from constituent.filter import check_single_child_label
from grammar.lcfrs import LCFRS_rule
from grammar.induction.terminal_labeling import PosTerminals, FeatureTerminals, FrequencyBiasedTerminalLabeling, \
    FormTerminals, StanfordUNKing, CompositionalTerminalLabeling
from playground.experiment_helpers import TRAINING, VALIDATION, TESTING, CorpusFile, RESULT, SplitMergeExperiment
from playground.constituent_split_merge import ConstituentExperiment, ScoringExperiment, token_to_features, \
    my_feature_filter, ScorerAndWriter, setup_corpus_resources
if sys.version_info < (3,):
    reload(sys)
    sys.setdefaultencoding('utf8')

# select one of the splits from {"SPMRL", "HN08", "WSJ", "WSJ-km2003"}
# SPLIT = "SPMRL"
# SPLIT = "HN08"
# SPLIT = "WSJ"
SPLIT = "WSJ-km2003"

DEV_MODE = True  # enable to parse the DEV set instead of the TEST set
QUICK = False  # enable for quick testing during debugging (small train/dev/test sets)

MULTI_OBJECTIVES = True  # runs evaluations with multiple parsing objectives but reuses the charts


# FINE_TERMINAL_LABELING = FeatureTerminals(token_to_features, feature_filter=my_feature_filter)
# FINE_TERMINAL_LABELING = FormTerminals()
FINE_TERMINAL_LABELING = CompositionalTerminalLabeling(FormTerminals(), PosTerminals())
FALLBACK_TERMINAL_LABELING = PosTerminals()
DEFAULT_RARE_WORD_THRESHOLD = 10


def terminal_labeling(corpus, threshold=DEFAULT_RARE_WORD_THRESHOLD):
    return FrequencyBiasedTerminalLabeling(FINE_TERMINAL_LABELING, FALLBACK_TERMINAL_LABELING, corpus, threshold)


class InductionSettings:
    def __init__(self):
        self.normalize = False
        self.disconnect_punctuation = True
        self.terminal_labeling = PosTerminals()
        # self.nont_labeling = NonterminalsWithFunctions()
        self.nont_labeling = BasicNonterminalLabeling()
        self.binarize = True
        self.isolate_pos = True
        self.hmarkov = 0
        self.use_discodop_binarization = False
        self.discodop_binarization_params = ["--headrules=../util/negra.headrules",
                                             "--binarize",
                                             "-h 1",
                                             "-v 1"  #,
                                             # '--labelfun=\"lambda n: n.label.split(\'^\')[0]\"'
                                            ]

    def __str__(self):
        __str = "Induction Settings {\n"
        for key in self.__dict__:
            if not key.startswith("__") and key not in []:
                __str += "\t" + key + ": " + str(self.__dict__[key]) + "\n"
        return __str + "}"


class LCFRSExperiment(ConstituentExperiment, SplitMergeExperiment):
    """
    Holds state and methods of a LCFRS-LA parsing experiment.
    """
    def __init__(self, induction_settings, directory=None, filters=None):
        ConstituentExperiment.__init__(self, induction_settings, directory=directory, filters=filters)
        SplitMergeExperiment.__init__(self)

        self.strip_vroot = False
        self.k_best = 500
        self.disco_binarized_corpus = None
        self.dummy_flag = True

    @staticmethod
    def __valid_tree(obj):
        return obj.complete() and not obj.empty_fringe()

    def run_discodop_binarization(self):
        """
        :rtype: None
        Binarize the training corpus using discodop. The resulting corpus is saved to to the the
        disco_binarized_corus member variable.
        """
        if self.disco_binarized_corpus is not None:
            return
        train_resource = self.resources[TRAINING]
        if self.induction_settings.normalize:
            train_normalized = self.normalize_corpus(train_resource.path, renumber=False)
        else:
            train_normalized = train_resource.path

        _, second_stage = tempfile.mkstemp(suffix=".export", dir=self.directory)

        subprocess.call(["discodop", "treetransforms"]
                        + self.induction_settings.discodop_binarization_params
                        + ["--inputfmt=export", "--outputfmt=export",
                           train_normalized, second_stage])

        disco_resource = CorpusFile(path=second_stage,
                                    start=train_resource.start,
                                    end=train_resource.end,
                                    limit=train_resource.limit,
                                    filter=train_resource.filter,
                                    exclude=train_resource.exclude,
                                    type=train_resource.type
                                   )

        self.disco_binarized_corpus = self.read_corpus_export(disco_resource, mode="DISCO-DOP", skip_normalization=True)

    def induce_from_disco_binarized(self, htree):
        """
        :type htree: HybridTree
        Induces a binarized LCFRS from a binarized version of htree, which is obtained by discodop.
        NB: the resulting grammar parses htree, due to a suitable choice of the tree component of the induced grammar.
        """
        self.run_discodop_binarization()
        htree_bin = None
        for _htree in self.disco_binarized_corpus:
            if _htree.sent_label() == htree.sent_label():
                htree_bin = _htree
                break
        assert htree_bin is not None
        grammar = direct_extract_lcfrs_from_prebinarized_corpus(htree_bin,
                                                                term_labeling=self.terminal_labeling,
                                                                nont_labeling=self.induction_settings.nont_labeling,
                                                                isolate_pos=self.induction_settings.isolate_pos,
                                                               )
        if self.backoff:
            self.terminal_labeling.backoff_mode = True
            grammar2 \
                = direct_extract_lcfrs_from_prebinarized_corpus(htree_bin,
                                                                term_labeling=self.terminal_labeling,
                                                                nont_labeling=self.induction_settings.nont_labeling,
                                                                isolate_pos=self.induction_settings.isolate_pos,
                                                               )
            self.terminal_labeling.backoff_mode = False
            grammar.add_gram(grammar2)

        if self.dummy_flag:
            print(grammar)
            self.dummy_flag = False

        return grammar, None

    def induce_from(self, obj):
        if self.induction_settings.use_discodop_binarization:
            return self.induce_from_disco_binarized(obj)

        if not self.__valid_tree(obj):
            print(obj, list(map(str, obj.token_yield())), obj.full_yield())
            return None, None
        grammar = direct_extract_lcfrs(obj,
                                       term_labeling=self.terminal_labeling,
                                       nont_labeling=self.induction_settings.nont_labeling,
                                       binarize=self.induction_settings.binarize,
                                       isolate_pos=self.induction_settings.isolate_pos,
                                       hmarkov=self.induction_settings.hmarkov)
        if self.backoff:
            self.terminal_labeling.backoff_mode = True
            grammar2 = direct_extract_lcfrs(obj,
                                            term_labeling=self.terminal_labeling,
                                            nont_labeling=self.induction_settings.nont_labeling,
                                            binarize=self.induction_settings.binarize,
                                            isolate_pos=self.induction_settings.isolate_pos,
                                            hmarkov=self.induction_settings.hmarkov)
            self.terminal_labeling.backoff_mode = False
            grammar.add_gram(grammar2)
        # print(grammar)
        # for rule in grammar.rules():
        #     print(rule)
        #     for lhs, rhs, dcp in binarize(rule.lhs(), rule.rhs(), rule.dcp()):
        #         bin_rule = LCFRS_rule(lhs=lhs, dcp=dcp)
        #         for rhs_nont in rhs:
        #             bin_rule.add_rhs_nont(rhs_nont)
        #         print("\t", bin_rule)
        # print()
        return grammar, None

    def initialize_parser(self):
        save_preprocess = (self.directory, "mygrammar")
        k = 1 if not self.organizer.disable_split_merge or self.oracle_parsing else self.k_best
        if "disco-dop" in self.parsing_mode:
            self.parser = DiscodopKbestParser(grammar=self.base_grammar, k=self.k_best,
                                              cfg_ctf=self.disco_dop_params["cfg_ctf"],
                                              pruning_k=self.disco_dop_params["pruning_k"],
                                              beam_beta=self.disco_dop_params["beam_beta"],
                                              beam_delta=self.disco_dop_params["beam_delta"]
                                             )
        else:
            self.parser = GFParser_k_best(self.base_grammar, save_preprocessing=save_preprocess, k=k)

    def compute_reducts(self, resource):

        # print_grammar(self.base_grammar)
        # for rule in self.base_grammar.rules():
        #     print(rule.get_idx(), rule)
        # sys.stdout.flush()

        training_corpus = list(filter(self.__valid_tree, self.read_corpus(resource)))
        parser = self.organizer.training_reducts.get_parser() if self.organizer.training_reducts is not None else None
        nonterminal_map = self.organizer.nonterminal_map
        frequency = self.backoff_factor if self.backoff else 1.0
        trace = compute_reducts(self.base_grammar, training_corpus, self.induction_settings.terminal_labeling,
                                parser=parser, nont_map=nonterminal_map, debug=False, frequency=frequency)
        if self.backoff:
            self.terminal_labeling.backoff_mode = True
            trace.compute_reducts(training_corpus, frequency=1.0)
            self.terminal_labeling.backoff_mode = False
        print("computed trace")
        return trace

    def print_config(self, file=None):
        if file is None:
            file = self.logger
        ConstituentExperiment.print_config(self, file=file)
        SplitMergeExperiment.print_config(self, file=file)

    def read_stage_file(self):
        ScoringExperiment.read_stage_file(self)

        if "training_reducts" in self.stage_dict:
            self.organizer.training_reducts = PySDCPTraceManager(self.base_grammar, self.terminal_labeling)
            self.organizer.training_reducts.load_traces_from_file(
                bytes(self.stage_dict["training_reducts"], encoding="utf-8"))

        if "validation_reducts" in self.stage_dict:
            self.organizer.validation_reducts = PySDCPTraceManager(self.base_grammar, self.terminal_labeling)
            self.organizer.validation_reducts.load_traces_from_file(
                bytes(self.stage_dict["validation_reducts"], encoding="utf-8"))

        SplitMergeExperiment.read_stage_file(self)


@plac.annotations(
    directory=('directory in which experiment is run', 'option', None, str)
    )
def main(directory=None):
    induction_settings = InductionSettings()
    induction_settings.hmarkov = 1
    induction_settings.disconnect_punctuation = False
    induction_settings.normalize = True
    induction_settings.use_discodop_binarization = True

    filters = []
    # filters += [check_single_child_label, lambda x: check_single_child_label(x, label="SB")]
    experiment = LCFRSExperiment(induction_settings, directory=directory, filters=filters)

    train, dev, test = setup_corpus_resources(SPLIT, DEV_MODE, QUICK)
    experiment.resources[TRAINING] = train
    experiment.resources[VALIDATION] = dev
    experiment.resources[TESTING] = test

    if "km2003" in SPLIT:
        experiment.eval_postprocess_options = ("--reversetransforms=km2003wsj",)
        backoff_threshold = 4
    else:
        backoff_threshold = 8

    induction_settings.terminal_labeling = terminal_labeling(experiment.read_corpus(experiment.resources[TRAINING]),
                                                             backoff_threshold)
    experiment.backoff = True


    experiment.terminal_labeling = induction_settings.terminal_labeling
    experiment.organizer.validator_type = "SIMPLE"
    experiment.organizer.project_weights_before_parsing = True
    experiment.organizer.disable_em = False
    experiment.organizer.disable_split_merge = False
    experiment.organizer.max_sm_cycles = 5
    experiment.organizer.threads = 8
    experiment.oracle_parsing = False
    experiment.k_best = 500
    experiment.disco_dop_params["pruning_k"] = 50000
    experiment.read_stage_file()

    if MULTI_OBJECTIVES:
        experiment.parsing_mode = "discodop-multi-method"
        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger,
                                                       secondary_scores=3)
        experiment.run_experiment()
    else:
        experiment.parsing_mode = "latent-viterbi-disco-dop"
        experiment.run_experiment()

        experiment.parsing_mode = "k-best-rerank-disco-dop"
        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.run_experiment()

        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.parsing_mode = "variational-disco-dop"
        experiment.run_experiment()

        experiment.resources[RESULT] = ScorerAndWriter(experiment,
                                                       directory=experiment.directory,
                                                       logger=experiment.logger)
        experiment.parsing_mode = "max-rule-prod-disco-dop"
        experiment.run_experiment()


if __name__ == '__main__':
    plac.call(main)
