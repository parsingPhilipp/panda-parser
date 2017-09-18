from __future__ import print_function
from graphs.dog import DirectedOrderedGraph, DeepSyntaxGraph
from collections import defaultdict
import re


def parse_file(path):
    corpus = []
    with open(path) as corpus_file:
        sentence = []
        sentence_id = 0
        for line in corpus_file:
            if re.match(r'^\s*$', line):
                if sentence != []:
                    corpus.append(parse_sentence(sentence, int(sentence_id)))
                    sentence = []
            elif line.startswith('#'):
                sentence_id = line[1:]
            else:
                sentence.append(line)
        if sentence != []:
            corpus.append(parse_sentence(sentence, int(sentence_id)))
    return corpus


def parse_sentence(lines, label=None):
    dog = DirectedOrderedGraph()
    arguments = defaultdict(list)
    predicates = {}
    predicate_list = []
    sentence = []
    synchronization = []
    for line in lines:
        contents = line.split()
        assert(len(contents) >= 7)
        idx = int(contents[0])
        dog.add_node(idx)
        form = contents[1]
        lemma = contents[2]
        pos = contents[3]
        frame = contents[6]
        top = contents[4] is '+'
        if top:
            dog.add_to_outputs(idx)
        pred = contents[5] is '+'
        if pred:
            predicates[idx] = (form, lemma, pos, frame)
            predicate_list.append(idx)
        else:
            dog.add_terminal_edge([], (form, lemma, pos, frame), idx)
        args = contents[7:]
        sentence.append((form, lemma, pos))
        synchronization.append([idx])
        for i, arg in enumerate(args):
            if arg is not '_':
                arguments[i].append((idx, arg))

    # print(predicates)
    # print(predicate_list)
    # print(arguments)
    for idx in predicates:
        edge = dog.add_terminal_edge(arguments[predicate_list.index(idx)], predicates[idx], idx)
        for i, arg in enumerate(arguments[predicate_list.index(idx)]):
            edge.set_function(i, arg[1])

    dsg = DeepSyntaxGraph(sentence, dog, synchronization, label=label)
    return dsg
