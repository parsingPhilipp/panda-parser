from __future__ import print_function
import unittest
from corpora.sdc_parse import parse_sentence, parse_file, export_sentence, export_corpus
from grammar.induction.decomposition import left_branching_partitioning
from graphs.util import render_and_view_dog, extract_recursive_partitioning, pretty_print_rec_partitioning
from graphs.graph_decomposition import compute_decomposition, induce_grammar_from, dog_evaluation
from graphs.dog_generator import generate, generate_sdg
from parser.naive.parsing import LCFRS_parser
from itertools import product
from random import randint

content = """
#20001001
1	Pierre	Pierre	NNP	-	-	_	NE	_	_	_	_	_
2	Vinken	vinken	NNP	-	+	_	_	_	_	ACT-arg	_	_
3	,	,	,	-	-	_	_	_	_	_	_	_
4	61	61	CD	-	-	_	_	RSTR	_	_	_	_
5	years	year	NNS	-	+	_	_	_	EXT	_	_	_
6	old	old	JJ	-	+	_	DESCR	_	_	_	_	_
7	,	,	,	-	-	_	_	_	_	_	_	_
8	will	will	MD	-	-	_	_	_	_	_	_	_
9	join	join	VB	+	+	ev-w1777f1	_	_	_	_	_	_
10	the	the	DT	-	-	_	_	_	_	_	_	_
11	board	board	NN	-	-	_	_	_	_	PAT-arg	_	_
12	as	as	IN	-	-	_	_	_	_	_	_	_
13	a	a	DT	-	-	_	_	_	_	_	_	_
14	nonexecutive	nonexecutive	JJ	-	-	_	_	_	_	_	RSTR	_
15	director	director	NN	-	+	_	_	_	_	COMPL	_	_
16	Nov.	nov.	NNP	-	+	_	_	_	_	TWHEN	_	_
17	29	29	CD	-	-	_	_	_	_	_	_	RSTR
18	.	.	.	-	-	_	_	_	_	_	_	_
"""


class SDPTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        unittest.TestCase.__init__(self, methodName)
        self.rec_part_strategies = [extract_recursive_partitioning,
                               lambda dsg: left_branching_partitioning(len(dsg.sentence))]

    def test_sdp_format(self):
        lines = content.splitlines()
        # print(lines[2:])
        dsg = parse_sentence(lines[2:])
        dog = dsg.dog
        # print(dog)
        # render_and_view_dog(dog, 'SDCtest')
        # print(export_sentence(dsg)[1:])
        self.assertListEqual(lines[2:], export_sentence(dsg)[1:])

    def test_sdp_parsing(self):
        for style, rec_part_strat in product(['dm', 'pas', 'psd'], self.rec_part_strategies):
            path = 'res/sdp/trial/' + style + '.sdp'
            corpus = parse_file(path)
            print(len(corpus))

            for i, dsg in enumerate(corpus):
                self.__process_single_dsg(i, dsg, rec_part_strat, terminal_labeling=lambda x: x[0])

    # the comparison takes to long in some cases!
    def __test_sdp_parsing_full(self):
        path = 'res/osdp-12/sdp/2015/en.dm.sdp'
        corpus = parse_file(path)
        print(len(corpus))
        for rec_part_strat in self.rec_part_strategies:
            for i, dsg in enumerate(corpus):
                if len(dsg.sentence) > 50:
                    continue
                self.__process_single_dsg(i, dsg, rec_part_strat, terminal_labeling=lambda x: x[0])

    def test_advanced_corpus_parsing(self):
        train_limit = 500
        train_dev_corpus_path = 'res/osdp-12/sdp/2015/en.dm.sdp'
        training_last = 21999042
        training_corpus = parse_file(train_dev_corpus_path, last_id=training_last, max_n=train_limit)

        dev_start = 22000001
        dev_limit = 10000
        dev_corpus = parse_file(train_dev_corpus_path, start_id=dev_start, max_n=dev_limit)

        cyclic = 0
        checked = 0
        for sdg in training_corpus:
            checked += 1
            if sdg.dog.cyclic():
                cyclic += 1
        self.assertEqual(checked, 500)
        self.assertEqual(cyclic, 0)


        cyclic = 0
        checked = 0
        for sdg in dev_corpus:
            checked += 1
            if sdg.dog.cyclic():
                cyclic += 1
        self.assertEqual(checked, 1692)
        self.assertEqual(cyclic, 0)
        export_corpus(dev_corpus, '/tmp/dev_corpus_export.dm.sdp')

    def __process_single_dsg(self, i, dsg, rec_part_strat, terminal_labeling):
        if True or len(dsg.dog.outputs) > 1:
            print(i, dsg, dsg.label)
            # if i == 89:
            # render_and_view_dog(dsg.dog, 'dm0', 'dm0')
            # render_and_view_dog(corpus[1].dog, 'dm1', 'dm1')
            # print(dsg.sentence, dsg.synchronization, dsg.label)

            # dog39 = dsg.dog.extract_dog([39], [], enforce_outputs=False)
            # render_and_view_dog(dog39, "dog39")

            rec_part = rec_part_strat(dsg)

            if False and i == 89:
                pretty_print_rec_partitioning(rec_part)

            decomp = compute_decomposition(dsg, rec_part)
            # print(decomp)

            grammar = induce_grammar_from(dsg, rec_part, decomp,
                                          terminal_labeling=terminal_labeling, enforce_outputs=False, normalize=True)
            if False and i == 89:
                print(grammar)

            parser = LCFRS_parser(grammar)
            parser.set_input(list(map(terminal_labeling, dsg.sentence)))

            print("parsing")

            parser.parse()
            self.assertTrue(parser.recognized())

            derivation = parser.best_derivation_tree()
            self.assertNotEqual(derivation, None)

            dog, sync_list = dog_evaluation(derivation)

            dsg.dog.project_labels(terminal_labeling)

            if False and i == 89:
                render_and_view_dog(dsg.dog, "corpus", "corpus_graph")
                render_and_view_dog(dog, "parse_result", "parse_result")

            print("comparing")

            self.assertEqual(dog, dsg.dog)

    def test_dog_generation(self):
        for rec_part_strat in self.rec_part_strategies:
            for i in range(5000):
                dsg = generate_sdg(randint(2, 12), maximum_inputs=3)
                if rec_part_strat == extract_recursive_partitioning and dsg.dog.cyclic():
                    continue
                # render_and_view_dog(dsg.dog, 'random_dog_' + str(i))
                try:
                    self.__process_single_dsg(i, dsg, rec_part_strat, terminal_labeling=str)
                except AssertionError:
                    render_and_view_dog(dsg.dog, 'random_dog_' + str(i))
                    self.__process_single_dsg(i, dsg, rec_part_strat, terminal_labeling=str)


if __name__ == '__main__':
    unittest.main()
