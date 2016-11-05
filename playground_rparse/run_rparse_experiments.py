import process_rparse_grammar

train = '/compute/kilian/rparse/negra-lower-punct/negra-lower-punct-train.conll'
test = '/compute/kilian/rparse/negra-lower-punct/negra-lower-punct-test.conll'

if __name__ == '__main__':
    for v in [1,2,3,"infinity"]:
        for h in [1,2,3,"infinity"]:
            name = "/compute/kilian/rparse_python/playground_rparse/negra_v{!s}_h{!s}".format(str(v),str(h))
            process_rparse_grammar.main(train, test, name, vMarkov=v, hMarkov=h)