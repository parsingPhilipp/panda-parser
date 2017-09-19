from __future__ import print_function
import unittest
from corpora.sdc_parse import parse_sentence, parse_file
from graphs.util import render_and_view_dog, extract_recursive_partitioning, pretty_print_rec_partitioning
from graphs.graph_decomposition import compute_decomposition, induce_grammar_from, dog_evaluation
from parser.naive.parsing import LCFRS_parser

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


class MyTestCase(unittest.TestCase):
    def test_sdp_format(self):
        lines = content.splitlines()
        print(lines[2:])
        dog = parse_sentence(lines[2:]).dog
        print(dog)
        render_and_view_dog(dog, 'SDCtest')

    def test_sdp_parsing(self):
        for style in ['dm', 'pas', 'psd']:
            path = 'res/sdp/trial/' + style + '.sdp'
            corpus = parse_file(path)
            print(len(corpus))

            for i, dsg in enumerate(corpus):
                # if i != 89:
                #     continue
                if True or len(dsg.dog.outputs) > 1:
                    print(i, dsg, dsg.label)
                    # if i == 89:
                    # render_and_view_dog(dsg.dog, 'dm0', 'dm0')
                    # render_and_view_dog(corpus[1].dog, 'dm1', 'dm1')
                    # print(dsg.sentence, dsg.synchronization, dsg.label)

                    # dog39 = dsg.dog.extract_dog([39], [], enforce_outputs=False)
                    # render_and_view_dog(dog39, "dog39")


                    rec_part = extract_recursive_partitioning(dsg)

                    if False and i == 89:
                        pretty_print_rec_partitioning(rec_part)

                    decomp = compute_decomposition(dsg, rec_part)
                    # print(decomp)

                    grammar = induce_grammar_from(dsg, rec_part, decomp,
                                                  terminal_labeling=lambda x: x[0], enforce_outputs=False, normalize=False)
                    if False and i == 89:
                        print(grammar)

                    parser = LCFRS_parser(grammar)
                    parser.set_input(map(lambda x: x[0], dsg.sentence))

                    print("parsing")

                    parser.parse()
                    self.assertTrue(parser.recognized())

                    derivation = parser.best_derivation_tree()
                    self.assertNotEqual(derivation, None)

                    dog, sync_list = dog_evaluation(derivation)

                    dsg.dog.project_labels(lambda x: x[0])

                    if False and i == 89:
                        render_and_view_dog(dsg.dog, "corpus", "corpus_graph")
                        render_and_view_dog(dog, "parse_result", "parse_result")

                    print("comparing")

                    self.assertEqual(dog, dsg.dog)



if __name__ == '__main__':
    unittest.main()
