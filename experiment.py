# The code to run experiments.

import time

import negra_parse
import tiger_parse
from parse_accuracy import ParseAccuracy
# from hybridtree import *
from constituency_tree import *
from induction import direct_extract_lcfrs, fringe_extract_lcfrs, \
	start as induction_start
from lcfrs import *
from parser.naive.parsing import *
from decomposition import *

# Different corpora and subsets of the corpora
# can be used for the experiments.
negra_any = ['negra nonproj', 'negra proj', 'negra nonproj subset']
tiger_any = ['tiger', 'tiger subset']

# Unselect precisely one:
# corpus = 'negra nonproj'
# corpus = 'negra proj'
# corpus = 'negra nonproj subset'
corpus = 'tiger'
# corpus = 'tiger subset'

### negra
# The one sentence with fanout 3:
negra_proj_fanout_3 = [18332]
# The sentences with fanout 4:
negra_nonproj_fanout_4 = \
	[448, 1572, 2055, 3269, 4664, 6684, 8232, 19485]

### tiger
# The sentences with fanout 4:
tiger_fanout_4 = \
	[3789, 8632, 11766, 11788, \
        12085, 13915, 15469, 17940, \
        23100, 24262, 25798, 32179, 39489, 39650]

############################################################
# Constants and corpus-specific procedures.

# Turn list of sentence names to list of trees.
# names: list of string
# return: list of HybridTree
def sentence_names_to_hybridtrees(names):
    if corpus == 'negra nonproj':
	return negra_parse.sentence_names_to_hybridtrees(names, \
			negra_parse.negra_nonproj)
    elif corpus == 'negra proj':
	return negra_parse.sentence_names_to_hybridtrees(names, \
			negra_parse.negra_proj)
    elif corpus == 'tiger':
	return tiger_parse.sentence_names_to_hybridtrees(names, \
			tiger_parse.tiger)
    elif corpus == 'tiger subset':
	return tiger_parse.sentence_names_to_hybridtrees(names, \
			tiger_parse.tiger_test)
    else:
	raise Exception('wrong corpus')

# Get first and last sentence (number) of corpus.
# return int
def first_sentence():
    if corpus in negra_any:
        return 1
    elif corpus in tiger_any:
        return 1
    else:
        raise Exception('wrong corpus')
# return int
def last_sentence():
    if corpus in negra_any:
        return 20602
    elif corpus == 'tiger':
        return 50474
    elif corpus == 'tiger subset':
        return 500
    else:
        raise Exception('wrong corpus')
# To be excluded for different reasons.
# return list of int
def excluded_sentences():
    if corpus in negra_any:
        return []
    elif corpus in tiger_any:
	# Missing, reentrant, resp.
        return [7561,17632] + [46234,50224]
    else:
        raise Exception('wrong corpus')
# return int
def first_training_sentence():
    if corpus in negra_any:
        return 1
    elif corpus in tiger_any:
        return 1
    else:
        raise Exception('wrong corpus')
# return int
def last_training_sentence():
    if corpus == 'negra nonproj subset':
        return 30 # for testing
    elif corpus in negra_any:
        return 18602 # corresponding to Dubey & Keller (ACL2003)
    elif corpus == 'tiger':
        return 7000
    elif corpus == 'tiger subset':
        return 450
    else:
        raise Exception('wrong corpus')
# return int
def first_test_sentence():
    if corpus == 'negra nonproj subset':
        return 7001 # for testing
    elif corpus in negra_any:
        return 18603 # corresponding to Dubey & Keller (ACL2003)
    elif corpus == 'tiger':
        return 7001
    elif corpus == 'tiger subset':
        return 451
    else:
        raise Exception('wrong corpus')
# return int
def last_test_sentence():
    if corpus == 'negra nonproj subset':
        return 7050 # for testing
    elif corpus in negra_any:
        return 19602 # corresponding to Dubey & Keller (ACL2003)
    elif corpus == 'tiger':
        return 7500
    elif corpus == 'tiger subset':
        return 500
    else:
        raise Exception('wrong corpus')

# Turn number of sentence into sentence name.
# i: int
# return: string
def make_name(i):
    if corpus in negra_any:
	return negra_parse.num_to_name(i)
    elif corpus in tiger_any:
	return tiger_parse.num_to_name(i)
    else:
        raise Exception('wrong corpus')

############################################################

# Turn range into list of sentence names.
# i: int
# j: int
# return: list of string
def make_names(i, j):
    return [make_name(k) for k in range(i,j+1) \
		if not k in excluded_sentences()]

# Turn range into hybrid trees.
# i: int
# j: int
# return: list of HybridTree
def make_trees(i, j):
    names = make_names(i,j)
    return sentence_names_to_hybridtrees(names)

# Do action f for trees in range, if test g hold.
# Return number of trees for which action was done.
# i: int
# j: int
# f: function taking HybridTree
# g: function taking HybridTree returning bool
# return: int
def do_range(i, j, f, g):
    n = 0
    trees = make_trees(i, j)
    for tree in trees:
	if g(tree):
	    f(tree)
	    n += 1
    return n

# Examples of f as third argument of do_range.
# tree: HybridTree
def print_canvas(tree):
    tree.canvas()
def print_graph(tree):
    print tree.graph()
def print_label(tree):
    print tree.sent_label()
def print_label_and_spans(tree):
    print tree.sent_label(), tree.max_n_spans()
def print_unicode(tree):
    print unicode(tree)

# In list of tuples, replace e.g. NP/2 by NP.
# Also remove ['START', 0, end-of-sentence]
# spans: list of spans
# return: list of spans
def normalize_labelled_spans(spans):
    normal = []
    for span in spans:
	if span[0] != induction_start:
	    arity_match = re.search(r'^(.*)/[0-9]+$', span[0])
	    if arity_match:
		stripped = arity_match.group(1)
		span[0] = stripped
	    normal += [span]
    return normal

############################################################
# Investigate corpus.

# Print trees with nodes with empty yield.
def print_trees_with_empty_yields():
    first = first_sentence()
    last = last_sentence()
    print 'Print trees with empty yields', \
	corpus, first, '-', last
    n = do_range(first, last, 
	print_label,
	lambda tree: tree.complete() and tree.empty_fringe() )
    print 'Investigated:', n
# UNCOMMENT to run experiment
# print_trees_with_empty_yields()

# Print trees with large spans.
# min_span: int
def print_trees_with_spans(min_span):
    first = first_sentence()
    last = last_sentence()
    print 'Print trees with spans exceeding', min_span, \
	corpus, first, '-', last
    n = do_range(first, last, 
	print_label_and_spans,
	lambda tree: tree.complete() and not tree.empty_fringe() \
			and tree.max_n_spans() >= min_span)
    print 'Investigated:', n
# UNCOMMENT to run experiment
# print_trees_with_spans(0)

# Print properties of grammar.
# g: LCFRS
def print_grammar_properties(g):
    print 'Number of rules:', len(g.rules())
    print 'Number of nonterminals:', len(g.nonts())
    print 'Grammar size:', g.size()

# Particular global variables for statistics about number of gaps.
n_nodes_gold = 0
n_gaps_gold = 0
n_nodes_test = 0
n_gaps_test = 0

# Clear these variables.
def clear_globals():
    n_nodes_gold = 0
    n_gaps_gold = 0
    n_nodes_test = 0
    n_gaps_test = 0

# Average global variables.
def eval_globals():
    if n_nodes_gold > 0:
	print "gaps per word (gold)", 1.0 * n_gaps_gold / n_nodes_gold
    if n_nodes_test > 0:
	print "gaps per word (test)", 1.0 * n_gaps_test / n_nodes_test

#############################################################
# Different ways of extracting the grammar from a tree.

# The standard way of extracting a LCFRS from a hybrid tree,
# with higher fanout of tree has higher fanout.
# t: HybridTree
# return: LCFRS
def basic_extraction(t):
    tree_part = t.unlabelled_structure()
    return fringe_extract_lcfrs(t, tree_part)

# Novel way of extracting LCFRS, placing maximum on fanout of LCFRS.
def cfg_extraction(t):
    return fanout_extraction(t, 1)
def fanout_two_extraction(t):
    return fanout_extraction(t, 2)
def fanout_three_extraction(t):
    return fanout_extraction(t, 3)
def fanout_four_extraction(t):
    return fanout_extraction(t, 4)
def fanout_five_extraction(t):
    return fanout_extraction(t, 5)
def fanout_six_extraction(t):
    return fanout_extraction(t, 6)
def fanout_seven_extraction(t):
    return fanout_extraction(t, 7)
def fanout_eight_extraction(t):
    return fanout_extraction(t, 8)
def cfg_extraction_child(t):
    return fanout_extraction_child(t, 1)
def fanout_two_extraction_child(t):
    return fanout_extraction_child(t, 2)
def fanout_three_extraction_child(t):
    return fanout_extraction_child(t, 3)
def fanout_four_extraction_child(t):
    return fanout_extraction_child(t, 4)
def fanout_five_extraction_child(t):
    return fanout_extraction_child(t, 5)
def fanout_six_extraction_child(t):
    return fanout_extraction_child(t, 6)
def fanout_seven_extraction_child(t):
    return fanout_extraction_child(t, 7)
def fanout_eight_extraction_child(t):
    return fanout_extraction_child(t, 8)
# strict labelling
def fanout_extraction(t, fanout):
    tree_part = t.unlabelled_structure()
    part = fanout_limited_partitioning(tree_part, fanout)
    return fringe_extract_lcfrs(t, part)
# child labelling
def fanout_extraction_child(t, fanout):
    tree_part = t.unlabelled_structure()
    part = fanout_limited_partitioning(tree_part, fanout)
    return fringe_extract_lcfrs(t, part, naming='child')

def left_branch_extraction(t):
    return fringe_extract_lcfrs(t, \
		left_branching_partitioning(len(t.pos_yield())))
def left_branch_extraction_child(t):
    return fringe_extract_lcfrs(t, \
		left_branching_partitioning(len(t.pos_yield())), \
		naming='child')

def right_branch_extraction(t):
    return fringe_extract_lcfrs(t, \
		right_branching_partitioning(len(t.pos_yield())))
def right_branch_extraction_child(t):
    return fringe_extract_lcfrs(t, \
		right_branching_partitioning(len(t.pos_yield())), \
		naming='child')

#############################################################
# Grammar induction and parsing.

# Test whether induced grammar for each sentence can be
# used to parse sentence.
# Is basic sanity check of implementation.
def test_induction(method=direct_extract_lcfrs):
    first = 1 # first_sentence()
    last = 100 # last_sentence()
    print 'Testing grammar induction from', corpus, first, '-', last, \
	'using method', method.__name__
    n = do_range(first, last, 
	lambda tree: induct_and_parse(tree, method),
	lambda tree: tree.complete() and not tree.empty_fringe() )
    print 'Tested on size:', n
# Do induction and parsing.
# tree: HybridTree
# method: function on HybridTree
def induct_and_parse(tree, method):
    gram = method(tree)
    # print gram
    print "n gaps:", tree.n_gaps()
    # if tree.n_gaps() > 0:
	# tree.canvas()
    inp = tree.pos_yield()
    p = LCFRS_parser(gram, inp)
    if not p.recognized():
	print 'failure', tree.sent_label()
# UNCOMMENT for running sanity check
# test_induction()
# test_induction(method=basic_extraction)
# test_induction(method=cfg_extraction)
# test_induction(method=left_branch_extraction_child)
# test_induction(method=right_branch_extraction_child)

# Induce grammar.
# method: function on HybridTree
# return: LCFRS
def induce(method=direct_extract_lcfrs):
    merged_gram = LCFRS(start=induction_start)
    first = first_training_sentence()
    last = last_training_sentence()
    print 'Inducing grammar from', corpus, first, '-', last, \
	'using method', method.__name__
    n = do_range(first, last, 
	lambda tree: add_gram(tree, merged_gram, method),
	lambda tree: tree.complete() and not tree.empty_fringe() )
    print 'Trained on size:', n
    merged_gram.make_proper()
    return merged_gram
# Add subgrammar describing tree to existing grammar.
# Use method of turning hybrid tree into LCFRS.
# tree: HybridTree
# gram: LCFRS
# method: function from HybridTree to LCFRS
def add_gram(tree, gram, method):
    added_gram = method(tree)
    gram.add_gram(added_gram)
# UNCOMMENT for stand-alone testing of induction
# print induce()
# print induce(method=basic_extraction)

# Parse test sentences with induced corpus.
# max_length: int
# method: function from HybridTree to LCFRS
def parse_test(max_length, method=direct_extract_lcfrs):
    g = induce(method=method)
    # print g # for testing
    accuracy = ParseAccuracy()
    clear_globals()
    first = first_test_sentence()
    last = last_test_sentence()
    print_grammar_properties(g)
    print 'Parsing', corpus, first, '-', last
    start_at = time.time()
    n = do_range(first, last, 
	lambda tree: parse_tree_by_gram(tree, g, accuracy),
	lambda tree: tree.complete() and not tree.empty_fringe() \
		and len(tree.word_yield()) >= 2 \
		and len(tree.word_yield()) <= max_length \
		# UNCOMMENT following line to restrict attention to sentences with gaps 
		# and tree.n_gaps() > 0 
		)
    end_at = time.time()
    print 'Parsed:', n
    if accuracy.n() > 0:
	print 'Recall:', accuracy.recall()
	print 'Precision:', accuracy.precision()
	print 'F-measure:', accuracy.fmeasure()
	print 'Parse failures:', accuracy.n_failures()
    else:
	print 'No successful parsing'
    eval_globals()
    print 'time:', end_at - start_at
    print ''
    sys.stdout.flush()
# Parse test sentence (yield of tree) using grammar.
def parse_tree_by_gram(tree, gram, accuracy):
    global n_nodes_gold
    global n_gaps_gold
    global n_nodes_test
    global n_gaps_test
    poss = tree.pos_yield()
    words = tree.word_yield()
    n_nodes_gold += tree.n_nodes()
    n_gaps_gold += tree.n_gaps()
    # tree.canvas() # for testing
    # print tree.n_gaps() # for testing
    p = LCFRS_parser(gram, poss)
    if not p.recognized():
        accuracy.add_failure()
    # print 'failure', tree.sent_label() # for testing
    else:
        # dcp_tree = p.dcp_hybrid_tree(poss, words)
        dcp_tree = HybridTree()
        dcp_tree = p.new_DCP_Hybrid_Tree(dcp_tree, poss, words)
        retrieved = dcp_tree.labelled_spans()
        relevant = tree.labelled_spans()
        accuracy.add_accuracy(retrieved, relevant)
        n_nodes_test += dcp_tree.n_nodes()
        n_gaps_test += dcp_tree.n_gaps()
	# print 'success', tree.sent_label() # for testing
# UNCOMMENT one or more of the following for running experiments
# parse_test(25)
# parse_test(20, method=basic_extraction)
# parse_test(20, method=cfg_extraction)
# parse_test(20, method=fanout_two_extraction)
# parse_test(20, method=fanout_three_extraction)
# parse_test(20, method=fanout_four_extraction)
# parse_test(20, method=left_branch_extraction)
# parse_test(20, method=right_branch_extraction)

# parse_test(20, method=left_branch_extraction_child)
# parse_test(20, method=right_branch_extraction_child)
parse_test(20, method=cfg_extraction_child)
# parse_test(20, method=fanout_two_extraction_child)
# parse_test(20, method=fanout_three_extraction_child)
# parse_test(20, method=fanout_four_extraction_child)

# Parse test sentences with induced corpus and compare them graphically.
# This can be useful for testing.
# max_length: int
# method: function from HybridTree to LCFRS
def parse_compare(max_length, method=direct_extract_lcfrs):
    g = induce(method)
    # print g # testing
    first = first_test_sentence()
    last = last_test_sentence()
    print 'Parsing', corpus, first, '-', last
    n = do_range(first, last, 
	lambda tree: parse_tree_by_gram_and_compare(tree, g),
	lambda tree: tree.complete() and not tree.empty_fringe() \
		and len(tree.leaves()) <= max_length )
# Parse test sentence (yield of tree) using grammar.
def parse_tree_by_gram_and_compare(tree, gram):
    # print tree.unlabelled_structure() # for testing
    poss = tree.pos_yield()
    words = tree.word_yield()
    p = LCFRS_parser(gram, poss)
    if not p.recognized():
        print 'failure', tree.sent_label()
    else:
        # dcp_tree = p.dcp_hybrid_tree(poss, words)
        dcp_tree = p.new_DCP_Hybrid_Tree(poss, words)
        # retrieved = normalize_labelled_spans(p.labelled_spans())
        retrieved = dcp_tree.labelled_spans()
        relevant = tree.labelled_spans()
        print 'retrieved', retrieved
        print 'relevant', relevant
        # tree.canvas()
        # dcp_tree.canvas()
# UNCOMMENT if desired
# parse_compare(20)
# parse_compare(20, method=basic_extraction)
