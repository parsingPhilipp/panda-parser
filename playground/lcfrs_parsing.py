from __future__ import print_function
from playground.experiment_helpers import TRAINING, VALIDATION, TESTING, CorpusFile, RESULT, SplitMergeExperiment
from constituent.induction import direct_extract_lcfrs, BasicNonterminalLabeling, NonterminalsWithFunctions, binarize, LCFRS_rule
from parser.gf_parser.gf_interface import GFParser, GFParser_k_best
from grammar.induction.terminal_labeling import PosTerminals
from playground.constituent_split_merge import ConstituentExperiment, ScoringExperiment, terminal_labeling
from parser.sDCP_parser.sdcp_trace_manager import compute_reducts, PySDCPTraceManager
from parser.sDCP_parser.sdcp_parser_wrapper import print_grammar
from constituent.filter import check_single_child_label
import sys
import plac
if sys.version_info < (3,):
    reload(sys)
    sys.setdefaultencoding('utf8')

train_limit = 10000 # 2000
# train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train5k/train5k.German.gold.xml'
# train_limit = 40474
train_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/train/train.German.gold.xml'
train_exclude = [7561, 17632, 46234, 50224]
train_corpus = None


validation_start = 40475
validation_size = validation_start + 200 #4999
validation_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'

test_start = validation_size # 40475
test_limit = test_start + 200 # 4999
test_exclude = train_exclude
test_path = '../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/xml/dev/dev.German.gold.xml'


class InductionSettings:
    def __init__(self):
        self.normalize = False
        self.disconnect_punctuation = True
        self.terminal_labeling = PosTerminals()
        # self.nont_labeling = NonterminalsWithFunctions()
        self.nont_labeling = BasicNonterminalLabeling()
        self.binarize = True
        self.isolate_pos = True

    def __str__(self):
        s = "Induction Settings {\n"
        for key in self.__dict__:
            if not key.startswith("__") and key not in []:
                s += "\t" + key + ": " + str(self.__dict__[key]) + "\n"
        return s + "}"


class LCFRSExperiment(ConstituentExperiment, SplitMergeExperiment):
    def __init__(self, induction_settings, directory=None, filters=None):
        ConstituentExperiment.__init__(self, induction_settings, directory=directory, filters=filters)
        SplitMergeExperiment.__init__(self)

        self.strip_vroot = True
        self.k_best = 500
        self.backoff = False
        self.backoff_factor = 10.0

    def induce_from(self, obj):
        if not obj.complete() or obj.empty_fringe():
            return None, None
        grammar = direct_extract_lcfrs(obj, term_labeling=self.terminal_labeling,
                                       nont_labeling=self.induction_settings.nont_labeling,
                                       binarize=self.induction_settings.binarize,
                                       isolate_pos=self.induction_settings.isolate_pos)
        if self.backoff:
            self.terminal_labeling.backoff_mode = True
            grammar2 = direct_extract_lcfrs(obj, term_labeling=self.terminal_labeling,
                                            nont_labeling=self.induction_settings.nont_labeling,
                                            binarize=self.induction_settings.binarize,
                                            isolate_pos=self.induction_settings.isolate_pos)
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
        save_preprocess=(self.directory, "mygrammar")
        if not self.organizer.disable_split_merge \
                or self.oracle_parsing:
            self.parser = GFParser_k_best(self.base_grammar, save_preprocess=save_preprocess, k=self.k_best)
        else:
            self.parser = GFParser(self.base_grammar, save_preprocess=save_preprocess)

    def compute_reducts(self, resource):

        # print_grammar(self.base_grammar)
        # for rule in self.base_grammar.rules():
        #     print(rule.get_idx(), rule)
        # sys.stdout.flush()

        training_corpus = self.read_corpus(resource)
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
        print(self.induction_settings, file=file)
        print("VROOT stripping", self.strip_vroot, file=file)

    def read_stage_file(self):
        ScoringExperiment.read_stage_file(self)

        if "training_reducts" in self.stage_dict:
            self.organizer.training_reducts = PySDCPTraceManager(self.base_grammar, self.terminal_labeling)
            self.organizer.training_reducts.load_traces_from_file(self.stage_dict["training_reducts"])

        if "validation_reducts" in self.stage_dict:
            self.organizer.validation_reducts = PySDCPTraceManager(self.base_grammar, self.terminal_labeling)
            self.organizer.validation_reducts.load_traces_from_file(self.stage_dict["validation_reducts"])

        SplitMergeExperiment.read_stage_file(self)


@plac.annotations(
    directory=('directory in which experiment is run', 'option', None, str)
    )
def main(directory=None):
    induction_settings = InductionSettings()
    experiment = LCFRSExperiment(induction_settings, directory=directory, filters=[check_single_child_label, lambda x: check_single_child_label(x, label="SB")])
    experiment.resources[TRAINING] = CorpusFile(path=train_path, start=1, end=train_limit, exclude=train_exclude)
    experiment.resources[VALIDATION] = CorpusFile(path=validation_path, start=validation_start, end=validation_size
                                                  , exclude=train_exclude)
    experiment.resources[TESTING] = CorpusFile(path=test_path, start=test_start,
                                               end=test_limit, exclude=train_exclude)
    # experiment.resources[TESTING] = CorpusFile(path=train_path, start=1, end=10, exclude=train_exclude)

    induction_settings.terminal_labeling = terminal_labeling(experiment.read_corpus(experiment.resources[TRAINING]))
    experiment.terminal_labeling = induction_settings.terminal_labeling
    experiment.organizer.validator_type = "SIMPLE"
    experiment.organizer.project_weights_before_parsing = False
    experiment.organizer.disable_em = True
    experiment.organizer.max_sm_cycles = 4
    experiment.backoff = True
    experiment.oracle_parsing = False
    experiment.read_stage_file()
    experiment.run_experiment()


if __name__ == '__main__':
    plac.call(main)