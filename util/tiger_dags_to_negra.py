#! /usr/bin/python3
import plac
import corpora.tiger_parse as tp
import corpora.negra_parse as np


@plac.annotations(
    src=('path/to/corpus.tigerxml', 'positional', None, str),
    target=('path/to/corpus.export', 'positional', None, str),
    start=('first sentence id', 'positional', None, int),
    stop=('last sentence id', 'positional', None, int)
    )
def main(src, target, start, stop):
    """
    Convert corpus in tigerxml format to negra export format. Secondary edges are preserved.
    """
    corpus = tp.sentence_names_to_deep_syntax_graphs(["s%i" % i for i in range(start, stop+1)], src, hold=False,
                                                     ignore_puntcuation=False)

    with open(target, 'w') as target:
        for dsg in corpus:
            dsg.set_label(dsg.label[1:])
            lines = np.serialize_hybrid_dag_to_negra([dsg], 0, 500, use_sentence_names=True)
            print(''.join(lines), file=target, end='')


if __name__ == "__main__":
    plac.call(main)
