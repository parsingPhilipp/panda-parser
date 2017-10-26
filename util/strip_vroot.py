from __future__ import print_function
import sys
import codecs
from corpora.negra_parse import sentence_names_to_hybridtrees, hybridtrees_to_sentence_names


def main():
    inpath = sys.argv[1]
    outpath = sys.argv[2]
    begin = int(sys.argv[3])
    end = int(sys.argv[4])
    print(inpath)
    sent_ids = [str(i) for i in range(begin, end+1)]
    corpus = sentence_names_to_hybridtrees(sent_ids, inpath)
    map(lambda x: x.strip_vroot(), corpus)
    with codecs.open(outpath, mode='w', encoding="utf-8") as file:
        lines = hybridtrees_to_sentence_names(corpus, begin, 2000)
        for line in lines:
            if not (isinstance(line, unicode) or isinstance(line, str)):
                print(line)
            try:
                file.write(line)
            except UnicodeEncodeError:
                print(line, type(line))
                raise


if __name__ == "__main__":
    main()
