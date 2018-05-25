from __future__ import print_function
from graphs.dog import DirectedOrderedGraph, DeepSyntaxGraph
from collections import defaultdict
import re
from itertools import chain


def parse_file(path, start_id=None, last_id=None, max_n=None):
    corpus = []
    with open(path) as corpus_file:
        sentence = []
        sentence_id = 0
        for line in corpus_file:
            if max_n is not None and len(corpus) >= max_n\
                    or last_id is not None and sentence_id > last_id:
                break
            if re.match(r'^\s*$', line) and (start_id is None or sentence_id >= start_id):
                if sentence != []:
                    entry = parse_sentence(sentence, sentence_id)
                    if (start_id is None or entry.label >= start_id)\
                       and (last_id is None or entry.label <= last_id):
                        corpus.append(entry)
                    sentence = []
            elif line.startswith('#'):
                if re.match(r'^#\d+$', line):
                    sentence_id = int(line[1:])
                    sentence = []
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


def export_corpus(dsgs, path, header='#SDP 2015'):
    with open(path, 'w') as file:
        file.write(header + '\n')
        lines = chain(*map(lambda x: export_sentence(x) + [''], dsgs))
        file.writelines(map(lambda x: x + '\n', lines))


def export_sentence(dsg):
    assert len(dsg.synchronization) == len(dsg.dog.nodes)
    predicates = {}

    for s in dsg.synchronization:
        assert len(s) == 1
        assert s[0] not in predicates
        node = s[0]
        if len(dsg.dog.incoming_edge(node).inputs) > 0:
            predicates[node] = len(predicates)

    lines = ['#' + str(dsg.label)]
    for i,(elem,s) in enumerate(zip(dsg.sentence, dsg.synchronization)):
        line = [str(i+1)]
        line += [elem[0], elem[1], elem[2]]
        line += ['+' if s[0] in dsg.dog.outputs else '-']
        line += ['+' if s[0] in predicates else '-']
        frame = dsg.dog.incoming_edge(s[0]).label[3]
        line.append(frame)
        relations = {i: '_' for i in range(len(predicates))}
        for parent in dsg.dog.parents[s[0]]:
            edge = dsg.dog.incoming_edge(parent)
            i = edge.inputs.index(s[0])
            relations[predicates[parent]] = edge.get_function(i)
        for i in range(len(predicates)):
            line.append(relations[i])

        lines.append('\t'.join(line))

    return lines


def export_sentence_dummy(sentence):
    return ['\t'.join([str(i+1), word[0], word[1], word[2], '-', '-', '-'])
            for i, word in enumerate(sentence)]


def build_dummy_dsg(sentence, label):
    dog = DirectedOrderedGraph()
    sync = []
    for i, word in enumerate(sentence):
        dog.add_node(i)
        dog.add_terminal_edge([], (word[0], word[1], word[2], '_'), i)
        sync.append([i])

    return DeepSyntaxGraph(sentence, dog, sync, label=label)

