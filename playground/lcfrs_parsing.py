from __future__ import print_function
from playground.experiment_helpers import TRAINING, VALIDATION, TESTING, CorpusFile, RESULT
from constituent.induction import direct_extract_lcfrs
from parser.gf_parser.gf_interface import GFParser
from grammar.induction.terminal_labeling import PosTerminals
from playground.constituent_split_merge import ConstituentExperiment
import sys
if sys.version_info < (3,):
    reload(sys)
    sys.setdefaultencoding('utf8')

train_limit = 5000 # 2000
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


class InductionSettings:
    def __init__(self):
        self.normalize = False
        self.disconnect_punctuation = True
        self.terminal_labeling = PosTerminals()


class LCFRSExperiment(ConstituentExperiment):
    def __init__(self, induction_settings, directory=None):
        super(LCFRSExperiment, self).__init__(induction_settings, directory=directory)
        self.strip_vroot = True

    def induce_from(self, obj):
        grammar = direct_extract_lcfrs(obj, term_labeling=self.terminal_labeling)
        # print(grammar)
        return grammar, None

    def initialize_parser(self):
        self.parser = GFParser(self.base_grammar, save_preprocess=(self.directory, "mygrammar"))


def main():
    induction_settings = InductionSettings()
    experiment = LCFRSExperiment(induction_settings)
    experiment.resources[TRAINING] = CorpusFile(path=train_path, start=1, end=train_limit, exclude=train_exclude)
    experiment.resources[VALIDATION] = CorpusFile(path=validation_path, start=validation_start, end=validation_size
                                                  , exclude=train_exclude)
    experiment.resources[TESTING] = CorpusFile(path=test_path, start=test_start,
                                               end=test_limit, exclude=train_exclude)
    # experiment.resources[TESTING] = CorpusFile(path=train_path, start=1, end=10, exclude=train_exclude)

    experiment.run_experiment()




if __name__ == "__main__":
    main()