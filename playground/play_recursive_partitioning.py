from corpora.conll_parse import parse_conll_corpus, tree_to_conll_str
from hybridtree.dependency_tree import disconnect_punctuation
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token
import dependency.induction as d_i
import dependency.labeling as d_l
import time
import parser.parser_factory
import copy
from parser.sDCP_parser.sdcp_parser_wrapper import print_grammar, PysDCPParser, LCFRS_sDCP_Parser, SDCPDerivation
from playground_rparse.process_rparse_grammar import fall_back_left_branching
import subprocess
import grammar.linearization as g_l
import decomposition as dec

test = '../res/negra-dep/negra-lower-punct-test.conll'
train ='../res/negra-dep/negra-lower-punct-train.conll'
result = 'recursive-partitoning-results.conll'
start = 'START'
term_labelling = d_i.the_terminal_labeling_factory().get_strategy('pos')
recursive_partitioning = d_i.the_recursive_partitioning_factory().getPartitioning('fanout-1')
primary_labelling = d_l.the_labeling_factory().create_simple_labeling_strategy('child', 'pos+deprel')

# parser_type = parser.parser_factory.GFParser  # slower, can be used for arbitrary fanout
parser_type = parser.parser_factory.CFGParser  # potentially faster, only for fanout 1
tree_parser = LCFRS_sDCP_Parser  # tree parser to count derivations per hybrid tree


tree_yield = term_labelling.prepare_parser_input

train_limit = 2000
test_limit = 2000

#define a few recursive partitionings for testing
test_par1 = (set([1,2,3,4]), [(set([1,3,4]), [(set([1,4]), [(set([1]), []) , (set([4]), [])]) , (set([3]), [])]) , (set([2]), [])])
test_par2 = (set([1,2,3,4,5]), [(set([1,2,4]), [(set([1,2]), [(set([1]), []), (set([2]), [])]), (set([4]), [])]), (set([3]), []), (set([5]), [])])
test_par3 = (set([1,2,3,4,5]), [(set([2]), []), (set([4]), []), (set([1,3,5]), [(set([1]), []), (set([3,5]), [(set([3]), []), (set([5]), [])])])])

def main(ignore_punctuation=False):


    #for-loops for trying out the various combinations of parameters
    #for strategy of choosing p
        #strict vs. child labeling
            #POS/DEPREL
                #fanout
                    #parser_type


    trees = parse_conll_corpus(train, False, train_limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    (n_trees, grammar) = d_i.induce_grammar(trees, primary_labelling, term_labelling.token_label, recursive_partitioning, start)

    #
    #
    # grammar is the induced hybrid grammar
    #
    #
    print "#nonts: ", len(grammar.nonts())
    print "#rules: ", len(grammar.rules())

    total_time = 0.0

    # The following code works for string parsers for evaluating
    
    parser_type.preprocess_grammar(grammar)
    
    trees = parse_conll_corpus(test, False, test_limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)
    
    with open(result, 'w') as result_file:
        failures = 0
        for tree in trees:
            time_stamp = time.clock()
    
            parser = parser_type(grammar, tree_yield(tree.token_yield()))
    
            time_stamp = time.clock() - time_stamp
            total_time += time_stamp
    
    
            cleaned_tokens = copy.deepcopy(tree.full_token_yield())
            for token in cleaned_tokens:
                token.set_deprel('_')
            h_tree = HybridTree(tree.sent_label())
            h_tree = parser.dcp_hybrid_tree_best_derivation(h_tree, cleaned_tokens, ignore_punctuation,
                                                            construct_conll_token)
   
            if h_tree:
                result_file.write(tree_to_conll_str(h_tree))
                result_file.write('\n\n')
            else:
                failures += 1
                forms = [token.form() for token in tree.full_token_yield()]
                poss = [token.pos() for token in tree.full_token_yield()]
                result_file.write(tree_to_conll_str(fall_back_left_branching(forms, poss)))
                result_file.write('\n\n')
    
    print "parse failures", failures
    print "parse time", total_time
    
    print "eval.pl", "no punctuation"
    p = subprocess.Popen(["perl", "../util/eval.pl", "-g", test, "-s", result, "-q"])
    p.communicate()
    
    print "eval.pl", "punctation"
    p = subprocess.Popen(
        ["perl", "../util/eval.pl", "-g", test, "-s", result, "-q", "-p"])
    p.communicate()

    print "total time: ", total_time
    # The following code is to count the number of derivations for a hypergraph (tree parser required)
    tree_parser.preprocess_grammar(grammar)

    trees = parse_conll_corpus(train , False, train_limit)
    if ignore_punctuation:
        trees = disconnect_punctuation(trees)

    derCount = 0
    derMax = 0
    for tree in trees:
        parser = tree_parser(grammar, tree)  # if tree parser is used
        der = parser.count_derivation_trees()
        if der > derMax:
            derMax = der
        derCount += der

    print "average number of derivations: ", 1.0*derCount/train_limit
    print "maximal number of derivations: ", derMax
    #file = open("grammar.txt", 'w')
    #g_l.linearize(grammar, primary_labelling, term_labelling, file)
    #file.close()
    #tree = trees.next()
    #print tree.id_yield()
    #print tree.recursive_partitioning()
    #siblings = tree.siblings('0')
    #print siblings

if __name__ == '__main__':
    main()



