from __future__ import print_function
from corpora.sdc_parse import parse_file, export_corpus, build_dummy_dsg, export_sentence
from parser.gf_parser.gf_interface import GFParser
from graphs.graph_decomposition import induce_grammar_from, compute_decomposition, dog_evaluation, consecutive_spans
from graphs.dog import DeepSyntaxGraph
from graphs.util import extract_recursive_partitioning
from decomposition import left_branching_partitioning, fanout_limited_partitioning_left_to_right
from subprocess import call
from experiment_helpers import Experiment, RESULT, CorpusFile


def worker(parser, graph, return_dict):
    parser.parse()
    if parser.recognized():
        derivation = parser.best_derivation_tree()
        assert derivation is not None
        dog, sync_list = dog_evaluation(derivation)
        result = DeepSyntaxGraph(graph.sentence, dog, sync_list, label=graph.label)
        return_dict[0] = result


class InductionSettings:
    def __init__(self):
        self.rec_part_strat = None
        self.terminal_labeling = None
        self.terminal_labeling_lcfrs = None
        self.nt_sub_labeling = None
        self.nonterminal_labeling = None


class SDPExperiment(Experiment):
    def __init__(self, induction_settings):
        Experiment.__init__(self)
        self.induction_settings = induction_settings
        self.resources[RESULT] = CorpusFile(header="#SDP 2015\n")

    def initialize_parser(self):
        self.parser = GFParser(self.base_grammar)

    def preprocess_before_induction(self, obj):
        pass

    def induce_from(self, graph):
        rec_part = self.induction_settings.rec_part_strat(graph)

        decomposition = compute_decomposition(graph, rec_part)

        graph_grammar = induce_grammar_from(graph, rec_part, decomposition,
                                            terminal_labeling=self.induction_settings.terminal_labeling,
                                            terminal_labeling_lcfrs=self.induction_settings.terminal_labeling_lcfrs,
                                            labeling=self.induction_settings.nonterminal_labeling,
                                            enforce_outputs=False, normalize=True)
        return graph_grammar, None

    def parsing_preprocess(self, graph):
        return map(self.induction_settings.terminal_labeling_lcfrs, graph.sentence)

    def parsing_postprocess(self, sentence, derivation, label=None):
        assert derivation is not None
        dog, sync_list = dog_evaluation(derivation)
        return DeepSyntaxGraph(sentence, dog, sync_list, label=label)

    def obtain_sentence(self, obj):
        return obj.sentence

    def obtain_label(self, obj):
        return obj.label

    def compute_fallback(self, sentence, label=None):
        return build_dummy_dsg(sentence, label)

    def read_corpus(self, resource):
        return parse_file(resource.path, start_id=resource.start, last_id=resource.end, max_n=resource.limit)

    def serialize(self, obj):
        lines = export_sentence(obj) + ['', '']
        return '\n'.join(lines)

    def evaluate(self, result_resource, gold_resource):
        if gold_resource.end is not None \
                or gold_resource.limit is not None\
                or gold_resource.length_limit is not None:
            corpus_gold_selection = self.read_corpus(gold_resource)
            gold_selection_resource = CorpusFile()
            gold_selection_resource.init()
            gold_selection_resource.finalize()
            export_corpus(corpus_gold_selection, gold_selection_resource.path)
            gold_resource = gold_selection_resource

        call(["sh", "../util/semeval-run.sh", "Scorer", gold_resource.path, result_resource.path, "representation=DM"])


def main():
    # path to corpus and ids of first/last sentences of sections
    train_dev_corpus_path = '../res/osdp-12/sdp/2015/en.dm.sdp'
    training_last = 21999042
    dev_start = 22000001
    # limit corpus sizes for testing purpose
    train_limit = 50
    dev_limit = 50

    def terminal_labeling(x):
        return '_', '_', x[2], x[3]

    def terminal_labeling_lcfrs(x):
        return x[2]

    def rec_part_strat(graph):
        # return left_branching_partitioning(len(graph.sentence))
        direct = extract_recursive_partitioning(graph)
        # return direct
        return fanout_limited_partitioning_left_to_right(direct, 1)

    def nt_sub_labeling(edge):
        return edge.label[2]

    def nonterminal_labeling(x, graph):
        bot = graph.dog.bottom(x)
        top = graph.dog.top(x)

        def labels(nodes):
            return [induction_settings.nt_sub_labeling(graph.dog.incoming_edge(node)) for node in nodes]

        fanout = consecutive_spans(graph.covered_sentence_positions(x))

        return '[' + ','.join(labels(bot)) + ';' + ','.join(labels(top)) + ';' + str(fanout) + ']'

    induction_settings = InductionSettings()
    induction_settings.terminal_labeling = terminal_labeling
    induction_settings.terminal_labeling_lcfrs = terminal_labeling_lcfrs
    induction_settings.rec_part_strat = rec_part_strat
    induction_settings.nt_sub_labeling = nt_sub_labeling
    induction_settings.nonterminal_labeling = nonterminal_labeling

    experiment = SDPExperiment(induction_settings)
    experiment.resources['TRAIN'] = CorpusFile(train_dev_corpus_path, end=training_last, limit=train_limit)
    experiment.resources['TEST'] = CorpusFile(train_dev_corpus_path, start=dev_start, limit=dev_limit)

    experiment.parsing_timeout = 150  # seconds

    experiment.run_experiment()


if __name__ == "__main__":
    main()
