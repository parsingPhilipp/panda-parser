from __future__ import print_function
from grammar.lcfrs import LCFRS
from parser.trace_manager.score_validator import PyCandidateScoreValidator
from parser.supervised_trainer.trainer import PyDerivationManager
from parser.trace_manager.sm_trainer import PySplitMergeTrainerBuilder, build_PyLatentAnnotation_initial
from parser.trace_manager.sm_trainer_util import PyGrammarInfo, PyStorageManager
from parser.gf_parser.gf_interface import GFParser_k_best
from parser.coarse_to_fine_parser.coarse_to_fine import Coarse_to_fine_parser
import tempfile
import multiprocessing
from sys import stdout
import codecs


TRAINING = "TRAIN"
VALIDATION = "VALIDATION"
TESTING = "TEST"
RESULT = "RESULT"


class SplitMergeOrganizer:
    def __init__(self):
        # basic objects
        self.grammarInfo = None
        self.storageManager = None
        self.nonterminal_map = None

        # settings and training ingredients
        self.training_reducts = None
        self.em_epochs = 20
        self.max_sm_cycles = 2
        self.threads = 1
        self.seed = 0
        self.validationDropIterations = 6
        self.smoothing_factor = 0.01
        self.split_randomization = 1.0  # in percent
        self.validator_type = "SCORE"  # SCORE or SIMPLE
        self.validator = None  # required for SCORE validation
        self.validation_reducts = None  # required for SIMPLE validation

        # the trainer state
        self.splitMergeTrainer = None
        self.latent_annotations = {}
        self.last_sm_cycle = None

    def __str__(self):
        s = ""
        for key in self.__dict__:
            if not key.startswith("__") and key not in ["grammarInfo", "storageManager", "nonterminal_map"]:
                s += key + ": " + str(self.__dict__[key]) + "\n"
        return s

class Resource(object):
    def __init__(self, path, start=1, end=None):
        self.start = start
        self.end = end
        self.path = path

    def init(self):
        pass

    def finalize(self):
        pass


class CorpusFile(Resource):
    def __init__(self, path=None, start=None, end=None, limit=None, length_limit=None, header=None, exclude=None):
        super(CorpusFile, self).__init__(path, start, end)
        self.limit = limit
        self.length_limit = length_limit
        self.file = None
        self.header = header
        if exclude is None:
            self.exclude = []
        else:
            self.exclude = exclude

    def init(self):
        if self.path is None:
            self.path = tempfile.mktemp()

        self.file = codecs.open(self.path, mode='w', encoding="utf-8")
        if self.header is not None:
            self.file.write(self.header)
        print('Opened', self.path)

    def finalize(self):
        self.file.close()

    def write(self, content):
        self.file.write(content)

    def __str__(self):
        attributes = [('path', self.path), ('length limit', self.length_limit), ('start', self.start),
                      ('end', self.end), ('limit', self.limit), ('exclude', self.exclude)]
        return '{' + ', '.join(map(lambda x: x[0] + ' : ' + str(x[1]), attributes)) + '}'


class Experiment(object):
    def __init__(self):
        self.base_grammar = None
        self.validator = None
        self.score_name = "score"
        self.parser = None
        self.result_file = None
        self.resources = {}
        self.parsing_timeout = None
        self.oracle_parsing = False
        self.max_score = None
        self.purge_rule_freq = None

    def induce_grammar(self, corpus, start="START"):
        grammar = LCFRS(start=start)
        for obj in corpus:
            obj = self.preprocess_before_induction(obj)
            obj_grammar, features = self.induce_from(obj)
            grammar.add_gram(obj_grammar, features)
        self.postprocess_grammar(grammar)
        self.base_grammar = grammar

    def postprocess_grammar(self, grammar):
        if self.purge_rule_freq is not None:
            grammar.purge_rules(self.purge_rule_freq)
        grammar.make_proper()

    def initialize_parser(self):
        assert False

    def preprocess_before_induction(self, obj):
        return obj

    def induce_from(self, obj):
        assert False

    def parsing_preprocess(self, obj):
        assert False

    def parsing_postprocess(self, sentence, derivation, label=None):
        assert False

    def obtain_sentence(self, obj):
        assert False

    def obtain_label(self, obj):
        return None

    def score_object(self, obj, gold):
        return 0.0

    def mk_obj(self, args):
        assert False

    def do_parse(self, corpus, result_resource):
        print("parsing, ", end='')
        result_resource.init()
        if self.parsing_timeout is None:
            for obj in corpus:
                parser_input = self.parsing_preprocess(obj)
                self.parser.set_input(parser_input)
                self.parser.parse()
                self.process_parse(obj, result_resource)
                self.parser.clear()
        else:
            self.do_parse_with_timeout(corpus, result_resource)
        result_resource.finalize()

    def process_parse(self, obj, result_resource):
        sentence = self.obtain_sentence(obj)

        if self.parser.recognized():
            print(".", end='')
            best_derivation = self.parser.best_derivation_tree()
            result = self.parsing_postprocess(sentence=sentence, derivation=best_derivation,
                                              label=self.obtain_label(obj))
        else:
            print("-", end='')
            result = self.compute_fallback(sentence=sentence)

        result_resource.write(self.serialize(result))

    def serialize(self, obj):
        assert False

    def run_experiment(self):
        self.print_config()

        # induction
        training_corpus = self.read_corpus(self.resources[TRAINING])
        self.induce_grammar(training_corpus)

        # weight training
        # omitted
        #
        self.initialize_parser()

        # testing
        test_corpus = self.read_corpus(self.resources[TESTING])
        self.do_parse(test_corpus, self.resources[RESULT])

        self.evaluate(self.resources[RESULT], self.resources[TESTING])

    def compute_fallback(self, sentence, label=None):
        assert False

    def read_corpus(self, resource):
        assert False

    def evaluate(self, result_resource, gold_resource):
        assert False

    def do_parse_with_timeout(self, corpus, result_resource):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        recognized = 0

        for obj in corpus:
            parser_input = self.parsing_preprocess(obj)
            self.parser.set_input(parser_input)

            timeout = False
            p = multiprocessing.Process(target=self.timeout_worker, args=(self.parser, obj, return_dict))
            p.start()
            p.join(timeout=self.parsing_timeout)
            if p.is_alive():
                p.terminate()
                p.join()
                timeout = True

            if 0 in return_dict and return_dict[0] is not None:
                recognized += 1
                print(".", end='')
                result = return_dict[0]
            else:
                if timeout:
                    print("t", end='')
                else:
                    print("-", end='')
                result = self.compute_fallback(self.obtain_sentence(obj), self.obtain_label(obj))

            result_resource.write(self.serialize(result))

            return_dict[0] = None
            self.parser.clear()

        print()
        print("From {} sentences, {} were recognized.".format(len(corpus), recognized))

    def timeout_worker(self, parser, obj, return_dict):
        parser.parse()
        if parser.recognized():
            derivation = parser.best_derivation_tree()
            return_dict[0] = self.parsing_postprocess(sentence=self.obtain_sentence(obj), derivation=derivation
                                                      , label=self.obtain_label(obj))

    def compute_oracle_derivation(self, derivations, gold):
        best_der = None
        best_score = -1.0
        sentence = self.obtain_sentence(gold)
        label = self.obtain_label(gold)
        min_score = 1.0

        for derivation in derivations:
            system = self.parsing_postprocess(sentence, derivation, label)
            score = self.score_object(system, gold)
            # print(score, end=' ')
            if score > best_score:
                best_der, best_score = derivation, score
            if self.max_score is not None and best_score >= self.max_score:
                break
            if score < min_score:
                min_score = score
        print('max', best_score, 'min', min_score)
        return best_der

    @staticmethod
    def precision_recall_f1(relevant, retrieved):
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

        return precision, recall, fmeasure

    def print_config(self, file=stdout):
        print("Experiment infos for", self.__class__.__name__, file=file)
        print("Purge rule freq:", self.purge_rule_freq, file=file)
        print("Max score", self.max_score, file=file)
        print("Score", self.score_name, file=file)
        print("Resources", '{\n' + '\n'.join([str(k) + ' : ' + str(self.resources[k]) for k in self.resources]) + '\n}', file=file)
        print("Parsing Timeout: ", self.parsing_timeout, file=file)
        print("Oracle parsing: ", self.oracle_parsing, file=file)


class ScoringExperiment(Experiment):
    def __init__(self):
        Experiment.__init__(self)

    def process_parse(self, gold, result_resource):
        sentence = self.obtain_sentence(gold)

        if self.parser.recognized():
            print('.', end='')
            if self.oracle_parsing:
                derivations = [der for _, der in self.parser.k_best_derivation_trees()]
                best_derivation = self.compute_oracle_derivation(derivations, gold)
            else:
                best_derivation = self.parser.best_derivation_tree()
            result = self.parsing_postprocess(sentence=sentence, derivation=best_derivation,
                                              label=self.obtain_label(gold))
            result_resource.score(result, gold)
        else:
            print('-', end='')
            result_resource.failure(gold)


class SplitMergeExperiment(Experiment):
    def __init__(self):
        self.organizer = SplitMergeOrganizer()
        self.parsing_mode = "k-best-rerank"  # or "best-latent-derivation"
        self.k_best = 50

    def build_score_validator(self, resource):
        self.organizer.validator = PyCandidateScoreValidator(self.organizer.grammarInfo
                                                             , self.organizer.storageManager, self.score_name)

        corpus_validation = self.read_corpus(resource)

        obj_count = 0
        der_count = 0
        for gold in corpus_validation:
            obj_count += 1
            self.parser.set_input(self.parsing_preprocess(gold))
            self.parser.parse()
            derivations = map(lambda x: x[1], self.parser.k_best_derivation_trees())
            manager = PyDerivationManager(self.base_grammar, self.organizer.nonterminal_map)
            manager.convert_derivations_to_hypergraphs(derivations)
            scores = []

            derivations = self.parser.k_best_derivation_trees()
            for _, der in derivations:
                der_count += 1
                result = self.parsing_postprocess(self.obtain_sentence(gold), der)
                score = self.score_object(result, gold)
                scores.append(score)

            self.organizer.validator.add_scored_candidates(manager, scores, self.max_score)
            print(obj_count, self.max_score, scores)
            self.parser.clear()
            print('.', end='')
        # print("trees used for validation ", obj_count, "with", der_count * 1.0 / obj_count, "derivations on average")

    def compute_reducts(self, resource):
        assert False

    def do_em_training(self):
        em_builder = PySplitMergeTrainerBuilder(self.organizer.training_reducts, self.organizer.grammarInfo)
        em_builder.set_em_epochs(self.organizer.em_epochs)
        em_builder.set_simple_expector(threads=self.organizer.threads)
        emTrainer = em_builder.build()

        # randomize initial weights and do em training
        la_no_splits = build_PyLatentAnnotation_initial(self.base_grammar, self.organizer.grammarInfo,
                                                        self.organizer.storageManager)
        la_no_splits.add_random_noise(self.organizer.grammarInfo, seed=self.organizer.seed)
        emTrainer.em_train(la_no_splits)
        la_no_splits.project_weights(self.base_grammar, self.organizer.grammarInfo)
        self.organizer.latent_annotations[0] = la_no_splits

    def prepare_split_merge_trainer(self):
        # prepare SM training
        builder = PySplitMergeTrainerBuilder(self.organizer.training_reducts, self.organizer.grammarInfo)
        builder.set_em_epochs(self.organizer.em_epochs)
        builder.set_split_randomization(1.0, self.organizer.seed + 1)
        builder.set_simple_expector(threads=self.organizer.threads)
        if self.organizer.validator_type == "SCORE":
            builder.set_score_validator(self.organizer.validator, self.organizer.validationDropIterations)
        elif self.organizer.validator_type == "SIMPLE":
            builder.set_simple_validator(self.organizer.validation_reducts, self.organizer.validationDropIterations)
        builder.set_smoothing_factor(smoothingFactor=self.organizer.smoothing_factor)
        builder.set_split_randomization(percent=self.organizer.split_randomization)
        # builder.set_scc_merger(-0.2)
        # builder.set_percent_merger(merge_percentage)
        self.organizer.splitMergeTrainer = builder.build()

        self.organizer.splitMergeTrainer.setMaxDrops(self.organizer.validationDropIterations, mode="smoothing")
        self.organizer.splitMergeTrainer.setEMepochs(self.organizer.em_epochs, mode="smoothing")

    def run_split_merge_cyclc(self):
        if self.organizer.last_sm_cycle is None:
            la_no_splits = build_PyLatentAnnotation_initial(self.base_grammar, self.organizer.grammarInfo,
                                                            self.organizer.storageManager)
            self.organizer.last_sm_cycle = 0
            self.organizer.latent_annotations[0] = la_no_splits

        current_la = self.organizer.latent_annotations[self.organizer.last_sm_cycle]
        next_la = self.organizer.splitMergeTrainer.split_merge_cycle(current_la)
        next_cycle = self.organizer.last_sm_cycle + 1
        self.organizer.last_sm_cycle = next_cycle
        self.organizer.latent_annotations[next_cycle] = next_la

    def initialize_training_environment(self):
        self.organizer.nonterminal_map = self.organizer.training_reducts.get_nonterminal_map()
        self.organizer.grammarInfo = PyGrammarInfo(self.base_grammar, self.organizer.nonterminal_map)
        self.organizer.storageManager = PyStorageManager()

    def prepare_sm_parser(self):
        last_la = self.organizer.latent_annotations[self.organizer.max_sm_cycles]
        if self.parsing_mode == "best-latent-derivation":
            grammar = last_la.build_sm_grammar(self.base_grammar, self.organizer.grammarInfo, rule_pruning=0.0001,
                                               rule_smoothing=0.1)
            self.parser = GFParser_k_best(grammar=grammar, k=1)
        elif self.parsing_mode == "k-best-rerank":
            last_la.project_weights(self.base_grammar, self.organizer.grammarInfo)
            self.parser = Coarse_to_fine_parser(self.base_grammar, GFParser_k_best, last_la, self.organizer.grammarInfo,
                                           self.organizer.nonterminal_map, k=self.k_best)

    def run_experiment(self):
        self.print_config()

        # induction
        training_corpus = self.read_corpus(self.resources[TRAINING])
        self.induce_grammar(training_corpus)

        # prepare reducts
        self.organizer.training_reducts = self.compute_reducts(self.resources[TRAINING])
        self.initialize_training_environment()
        self.do_em_training()

        self.initialize_parser()

        if True:
            if self.organizer.validator_type == "SCORE":
                self.build_score_validator(self.resources[TESTING])
            elif self.organizer.validator_type == "SIMPLE":
                self.organizer.validation_reducts = self.compute_reducts(self.resources[TESTING])
            self.prepare_split_merge_trainer()

            while self.organizer.last_sm_cycle is None or self.organizer.last_sm_cycle < self.organizer.max_sm_cycles:
                self.run_split_merge_cyclc()

        # testing
        self.prepare_sm_parser()
        test_corpus = self.read_corpus(self.resources[TESTING])
        self.do_parse(test_corpus, self.resources[RESULT])

        self.evaluate(self.resources[RESULT], self.resources[TESTING])

    def print_config(self, file=stdout):
        print("Split/Merge Settings: \n", self.organizer, file=file)
        print("Split/Merge Parsing mode: ", self.parsing_mode, file=file)


class ScorerResource(Resource):
    def __init__(self, path=None, start=None, end=None):
        super(ScorerResource, self).__init__(path, start, end)

    def score(self, system, gold):
        assert False

    def failure(self, gold):
        assert False
