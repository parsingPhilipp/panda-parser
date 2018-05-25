import playground_rparse.process_rparse_grammar
train = 'res/negra-dep/negra-lower-punct-train.conll'
test = 'res/negra-dep/negra-lower-punct-test.conll'
corpus_name = 'negra'

if __name__ == '__main__':
    for v in [1]:  # [1,2,3,"infinity"]:
        for h in [4, 5, 6, 7]:  # [1,2,3,"infinity"]:
            name = "playground_rparse/results/{!s}_v{!s}_h{!s}".format(corpus_name, str(v), str(h))
            playground_rparse.process_rparse_grammar.main(train, test, name, vMarkov=v, hMarkov=h)
