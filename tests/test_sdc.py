from __future__ import print_function
import unittest
from corpora.sdc_parse import parse_sentence, parse_file
from graphs.util import render_and_view_dog, extract_recursive_partitioning

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
        path = 'res/sdp/trial/psd.sdp'
        corpus = parse_file(path)
        print(len(corpus))

        for i, dsg in enumerate(corpus):
            if len(dsg.dog.outputs) > 1:
                print(i, len(dsg.dog.nodes))

        render_and_view_dog(corpus[52].dog, 'dm0', 'dm0')
        # render_and_view_dog(corpus[1].dog, 'dm1', 'dm1')
        print(corpus[52].sentence, corpus[52].synchronization, corpus[52].label)

        print(extract_recursive_partitioning(corpus[52]))


if __name__ == '__main__':
    unittest.main()
