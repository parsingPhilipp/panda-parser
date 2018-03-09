import pickle
import plac
from grammar.lcfrs import LCFRS


@plac.annotations(
    path=('path to pickled grammar file', 'option', None, str)
    )
def main(path=None):
    with open(path, 'rb') as grammar_file:
        grammar = pickle.load(grammar_file)

        assert(isinstance(grammar, LCFRS))

        print("Nonterminals", len(grammar.nonts()))
        print("Rules", len(grammar.rules()))
        print("lex. Rules", len([r for r in grammar.rules() if r.rhs() == []]))


if __name__ == "__main__":
    plac.call(main)