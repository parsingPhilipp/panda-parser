import process_rparse_grammar
train = '/compute/kilian/rparse/negra-lower-punct/negra-lower-punct-train.conll'
#train = '/compute/kilian/dependency_conll/german/tiger/train/german_tiger_train.conll'
test = '/compute/kilian/rparse/negra-lower-punct/negra-lower-punct-test.conll'
# test = '/compute/kilian/rparse/tiger_smaller_20.conll'
corpus_name = 'negra'
# corpus_name = 'tiger'

if __name__ == '__main__':
    for v in [1]: # [1,2,3,"infinity"]:
        for h in [4,5,6,7]: # [1,2,3,"infinity"]:
            name = "/compute/kilian/rparse_python/playground_rparse/{!s}_v{!s}_h{!s}".format(corpus_name, str(v), str(h))
            process_rparse_grammar.main(train, test, name, vMarkov=v, hMarkov=h)
