from __future__ import print_function
from grammar.lcfrs import LCFRS
from parser.trace_manager.score_validator import PyCandidateScoreValidator
from parser.supervised_trainer.trainer import PyDerivationManager
import tempfile
import multiprocessing


TRAINING = "TRAIN"
VALIDATION = "VALIDATION"
TESTING = "TEST"
RESULT = "RESULT"


class SplitMergeOrganizer:
    def __init__(self):
        self.grammarInfo = None
        self.storageManager = None
        self.nonterminal_map = None


class Resource:
    def __init__(self, path, start=1, end=None):
        self.start = start
        self.end = end
        self.path = path

    def init(self):
        pass

    def finalize(self):
        pass


class CorpusFile(Resource):
    def __init__(self, path=None, start=None, end=None, limit=None, length_limit=None, header=None):
        Resource.__init__(self, path, start, end)
        self.limit = limit
        self.length_limit = length_limit
        self.file = None
        self.header = header

    def init(self):
        if self.path is None:
            self.path = tempfile.mktemp()

        self.file = open(self.path, mode='w')
        if self.header is not None:
            self.file.write(self.header)
        print('Opened', self.path)

    def finalize(self):
        self.file.close()

    def write(self, content):
        self.file.write(content)


class Experiment:
    def __init__(self):
        self.base_grammar = None
        self.validator = None
        self.__score_name = "score"
        self.__organizer = None
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
            self.preprocess_before_induction(obj)
            obj_grammar, features = self.induce_from(obj)
            grammar.add_gram(obj_grammar, features)
        self.postprocess_grammar(grammar)
        self.base_grammar = grammar

    def postprocess_grammar(self, grammar):
        if self.purge_rule_freq is not None:
            grammar.purge_rules(self.purge_rule_freq)
        grammar.make_proper()
    def initialize_parser(self):
        pass

    def preprocess_before_induction(self, obj):
        pass

    def induce_from(self, obj):
        pass

    def parsing_preprocess(self, obj):
        pass

    def parsing_postprocess(self, sentence, derivation, label=None):
        pass

    def obtain_sentence(self, obj):
        pass

    def obtain_label(self, obj):
        return None

    def score_object(self, obj, gold):
        return 0.0

    def mk_obj(self, args):
        pass

    def build_score_validator(self, corpus_validation):
        self.validator = PyCandidateScoreValidator(self.__organizer.grammarInfo, self.__organizer.storageManager,
                                                   self.__score_name)

        obj_count = 0
        der_count = 0
        for gold in corpus_validation.get_trees():
            obj_count += 1
            self.parser.set_input(self.parsing_preprocess(gold))
            self.parser.parse()
            derivations = map(lambda x: x[1], self.parser.k_best_derivation_trees())
            manager = PyDerivationManager(self.base_grammar, self.__organizer.nonterminal_map)
            manager.convert_derivations_to_hypergraphs(derivations)
            scores = []

            derivations = self.parser.k_best_derivation_trees()
            for _, der in derivations:
                der_count += 1
                result = self.parsing_postprocess(self.obtain_sentence(gold), der)
                score = self.score_object(result, gold)
                scores.append(score)

            max_score = len(gold.id_yield())
            self.validator.add_scored_candidates(manager, scores, max_score)
            # print(obj_count, max_score, scores)
            self.parser.clear()
        # print("trees used for validation ", obj_count, "with", der_count * 1.0 / obj_count, "derivations on average")

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
            best_derivation = self.parser.best_derivation_tree()
            result = self.parsing_postprocess(sentence=sentence, derivation=best_derivation,
                                              label=self.obtain_label(obj))
        else:
            result = self.compute_fallback(sentence=sentence)

        result_resource.write(self.serialize(result))

    def serialize(self, obj):
        pass

    def run_experiment(self):
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
        pass

    def read_corpus(self, resource):
        pass

    def evaluate(self, result_resource, gold_resource):
        pass

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

        for derivation in derivations:
            system = self.parsing_postprocess(sentence, derivation, label)
            score = self.score_object(system, gold)
            # print(score, end=' ')
            if score > best_score:
                best_der, best_score = derivation, score
            if self.max_score is not None and best_score >= self.max_score:
                break
        # print('max', best_score)
        return best_der
