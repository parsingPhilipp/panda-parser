from __future__ import print_function
from corpora.sdc_parse import parse_file, export_corpus, build_dummy_dsg
from grammar.lcfrs import LCFRS
from parser.gf_parser.gf_interface import GFParser
from graphs.graph_decomposition import induce_grammar_from, compute_decomposition, dog_evaluation, consecutive_spans
from graphs.dog import DeepSyntaxGraph
from graphs.util import extract_recursive_partitioning
from decomposition import left_branching_partitioning, fanout_limited_partitioning_left_to_right
from subprocess import call
import multiprocessing
import time

train_limit = 5000
train_dev_corpus_path = '../res/osdp-12/sdp/2015/en.dm.sdp'
training_last = 21999042
training_corpus = parse_file(train_dev_corpus_path, last_id=training_last, max_n=train_limit)

dev_start = 22000001
dev_limit = 50
dev_corpus = parse_file(train_dev_corpus_path, start_id=dev_start, max_n=dev_limit)

parsing_timeout = 20  # in seconds


def worker(parser, graph, return_dict):
    parser.parse()
    if parser.recognized():
        derivation = parser.best_derivation_tree()
        assert derivation is not None
        dog, sync_list = dog_evaluation(derivation)
        result = DeepSyntaxGraph(graph.sentence, dog, sync_list, label=graph.label)
        return_dict[0] = result


def main():
    grammar = LCFRS("START")

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

    # grammar induction
    for graph in training_corpus:
        rec_part = rec_part_strat(graph)

        decomp = compute_decomposition(graph, rec_part)
        # print(decomp)

        def nonterminal_labeling(x, graph):
            bot = graph.dog.bottom(x)
            top = graph.dog.top(x)

            def labels(nodes):
                return [nt_sub_labeling(graph.dog.incoming_edge(node)) for node in nodes]

            fanout = consecutive_spans(graph.covered_sentence_positions(x))

            return '[' + ','.join(labels(bot)) + ';' + ','.join(labels(top)) + ';' + str(fanout) + ']'

        graph_grammar = induce_grammar_from(graph, rec_part, decomp,
                                      terminal_labeling=terminal_labeling,
                                      terminal_labeling_lcfrs=terminal_labeling_lcfrs,
                                            labeling=nonterminal_labeling, enforce_outputs=False, normalize=True)
        grammar.add_gram(graph_grammar)

    # testing (on dev set)
    print("Nonterminals:", len(grammar.nonts()), "Rules:", len(grammar.rules()))
    print(grammar, file=open('/tmp/the_grammar.txt', 'w'))
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    parser = GFParser(grammar)
    print("parsing, ", end='')
    recognized = 0
    results = []
    for graph in dev_corpus:
        parser.set_input(map(terminal_labeling_lcfrs, graph.sentence))

        # parse with timeout
        start = time.time()
        timeout = False
        p = multiprocessing.Process(target=worker, args=(parser, graph, return_dict))
        p.start()
        p.join(timeout=parsing_timeout)
        if p.is_alive():
            p.terminate()
            # print("Timeout after", time.time() - start, "seconds.")
            p.join()
            timeout = True

        if 0 in return_dict and return_dict[0] is not None:
            recognized += 1
            print(".", end='')
            results.append(return_dict[0])
        else:
            if timeout:
                print("t", end='')
            else:
                print("-", end='')
            results.append(build_dummy_dsg(graph.sentence, graph.label))

        parser.clear()
        return_dict[0] = None

    print()
    print("From {} sentences, {} were recognized.".format(len(dev_corpus), recognized))
    print()
    gold_file = '/tmp/dev_corpus_gold.dm.sdp'
    system_file = '/tmp/dev_corpus_system.dm.sdp'
    export_corpus(dev_corpus, gold_file)
    export_corpus(results, system_file)

    call(["sh", "../util/semeval-run.sh", "Scorer", gold_file, system_file, "representation=DM"])


if __name__ == "__main__":
    main()
