import re
import pgf
import os
from hybridtree.general_hybrid_tree import HybridTree
from hybridtree.monadic_tokens import construct_conll_token, construct_constituent_token
from collections import defaultdict
import string
import subprocess
from corpora.conll_parse import tree_to_conll_str
import plac
import time

def new_lex_files(abstract_path, output_abstract, output_concrete):
    with open(abstract_path) as abstract, open(output_abstract, 'w') as abstract_new, open(output_concrete, 'w') as concrete_new:
        abstract_new.write('abstract bingrammargflexabstract = {\n\n')
        concrete_new.write('concrete bingrammargflexconcrete of bingrammargflexabstract = {\n\n')
        while True:
            try:
                line = abstract.next()
            except:
                break
            match = re.search(r'^cat\s([^\s]+)1\s;$', line)
            if match:
                pos = match.group(1)
                abstract_new.write(line)
                abstract_new.write('fun ' + 'fun_' + pos + ' : ' + pos + '1;\n')
                concrete_new.write('lincat ' + pos + '1 = { p1 : Str };\n')
                concrete_new.write('lin fun_' + pos + ' = { p1="' + pos + '" };\n')
        abstract_new.write('}')
        concrete_new.write('}')


def escape(s):
    s = s.replace("$", "")
    s = s.replace("(", "_LBR_")
    s = s.replace(")", "_RBR_")
    s = s.replace("[", "_LSQBR_")
    s = s.replace("]", "_RSQBR_")
    s = s.replace("*", "_STAR_")
    s = s.replace("|", "_PIPE_")
    s = s.replace(".", "_PUNCT_")
    s = s.replace(",", "_COMMA_")
    s = s.replace("--", "_MDASH_")
    s = s.replace("-", "_DASH_")
    s = s.replace("/", "_SLASH_")
    s = s.replace("\\", "_BACKSLASH_")
    s = s.replace("\"", "_DQ_")
    s = s.replace("\'", "_SQ_")
    s = s.replace("@", "_AT_")
    s = s.replace("^", "_HAT_")
    return s

def un_escape(s):
    s = s.replace( "LBR" , "(" )
    s = s.replace( "RBR" , ")" )
    s = s.replace( "_LSQBR_" , "[" )
    s = s.replace( "_RSQBR_" , "]" )
    s = s.replace( "_STAR_" , "*" )
    s = s.replace( "_PIPE_" , "|" )
    s = s.replace( "_PUNCT_"  , "." )
    s = s.replace( "_COMMA_"  , "," )
    s = s.replace("_MDASH_", "--")
    s = s.replace( "_DASH_"  , "-" )
    s = s.replace( "SLASH"  , "/" )
    s = s.replace("BACKSLASH", "\\")
    s = s.replace("DQ", "\"")
    s = s.replace("SQ", "\'")
    return s

def parse_with_pgf(gr, forms, poss):
    """"
    :type gr: PGF
    :return:
    :rtype:
    """
    lcfrs = gr.languages['bingrammargfconcrete']

    # sentence = "ADJD ADV _COMMA_ KOUS ADV PIS PROAV VVINF VMFIN _PUNCT_"
    sentence = ' '.join(map(escape, poss))

    try:
        i = lcfrs.parse(sentence, n=1)
    except (StopIteration, pgf.ParseError):
        return None

    p, e = i.next()

    # print_ast(gr, e, 0)
    s = lcfrs.graphvizParseTree(e)
    assert isinstance(s, str)
    s_ = s.splitlines()

    tree = HybridTree()

    # print s
    i = 0
    for line in s.splitlines():
        match = re.search(r'^\s*(n\d+)\[label="([^\s]+)"\]\s*$', line)
        if match:
            node_id = match.group(1)
            label = match.group(2)
            order = int(node_id[1:]) >= 100000
            if order:
                assert escape(poss[i]) == label
                tree.add_node(node_id, construct_constituent_token(form=forms[i], pos=poss[i], terminal=True), True)
                i += 1
            else:
                tree.add_node(node_id, construct_constituent_token(form=label, pos='_', terminal=False), False)
            # print node_id, label
            if label == 'VROOT1':
                tree.add_to_root(node_id)
            continue
        match = re.search(r'^  (n\d+) -- (n\d+)\s*$', line)
        if match:
            parent = match.group(1)
            child = match.group(2)
            tree.add_child(parent, child)
            # print line
            # print parent, child
            continue

    # print tree

    assert poss == [token.pos() for token in tree.token_yield()]
    # print the_yield

    dep_tree = HybridTree()
    head_table = defaultdict(lambda: None)
    attachment_point = defaultdict(lambda: None)
    for i, node in enumerate(tree.id_yield()):
        token = tree.node_token(node)
        dep_token = construct_conll_token(token.form(), un_escape(token.pos()))
        current = tree.parent(node)
        current = tree.parent(current)
        while current:
            current_label = tree.node_token(current).category()
            if not re.search(r'\d+X\d+$', current_label):
                s = un_escape(current_label)
                if s == 'TOP1':
                    s = 'ROOT1'
                dep_token.set_deprel(s[:-1])
                head_table[current] = i + 1
                attachment_point[node] = current
                break
            else:
                current = tree.parent(current)
        dep_tree.add_node(i + 1, dep_token, order=True)

    # print head_table

    for node, dep_node in zip(tree.id_yield(), dep_tree.id_yield()):
        node = tree.parent(attachment_point[node])
        while node:
            if head_table[node]:
                dep_tree.add_child(head_table[node], dep_node)
                break
            node = tree.parent(node)
        if not node:
            dep_tree.add_to_root(dep_node)

    # print "dep_tree"
    # print dep_tree
    # print ' '.join(['(' + token.form() + '/' + token.deprel() + ')' for token in dep_tree.token_yield()])
    return dep_tree


def print_ast(gr, e, indent):
    node, children = e.unpack()
    print ' ' * indent, gr.functionType(node)
    for c in children:
        print_ast(gr, c, indent + 2)


def match_line(line):
    match = re.search(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)'
                      r'\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)$', line)
    return match


def fall_back_left_branching(forms, poss):
    tree = HybridTree()
    n = len(poss)
    for i, (form, pos) in enumerate(zip(forms, poss)):
        token = construct_conll_token(form, poss)
        token.set_deprel('_')
        tree.add_node(i, token, True)
        if i == 0:
            tree.add_to_root(i)
        else:
            tree.add_child(i-1, i)
    return tree


def parse(gf, input, output, verbose=False):
    parse_failures = 0
    parse_time = 0.0
    with open(input) as input_file, open(output, 'w') as output_file:
        forms = []
        poss = []
        while True:
            try:
                line = input_file.next()
                while line.startswith('#'):
                    line = input_file.next()
            except StopIteration:
                break

            match = match_line(line)
            if match:
                form = match.group(2)
                pos = match.group(5)
                forms.append(form)
                poss.append(pos)
            elif re.search(r'^[^\s]*$', line):
                if verbose:
                    print poss
                time_stamp = time.clock()
                result = parse_with_pgf(gf, forms, poss)
                parse_time = parse_time + (time.clock() - time_stamp)
                if not result:
                    parse_failures += 1
                    result = fall_back_left_branching(forms, poss)
                    if verbose:
                        print "parse failure"
                # print result
                output_file.write(tree_to_conll_str(result))
                output_file.write('\n\n')
                forms = []
                poss = []
            else:
                print line
                raise Exception("unexpected input")
    return parse_failures, parse_time


rparse_path = "../util/rparse.jar"

@plac.annotations(
      forceRecompile=('force Recompilation', 'flag')
    , binarization=('binarization strategy', 'option')
    , vMarkov=('vertical Markovization', 'option')
    , hMarkov=('horizontal Markov', 'option')
    , v=('verbose', 'flag')
)
def main(train, test, grammarName, binarization="km", vMarkov=2, hMarkov=1, forceRecompile=False, optional_args="", v=False):
    rparse_params = ["-dep", "-doTrain", "-trainFormat", "conll", "-train", train, "-binType", binarization, "-vMarkov", str(vMarkov), "-hMarkov", str(hMarkov), "-binSave", grammarName] + optional_args.split(' ')
    #
    if forceRecompile or not os.path.isfile(grammarName + "/bingrammargfabstract.gf"):
        print "Extracting grammar with rparse"
        p = subprocess.Popen(['java', "-jar", rparse_path] + rparse_params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        print out
        print err
    else:
        print "Found grammar", grammarName + "/bingrammargfabstract.gf"

    #
    if forceRecompile or not os.path.isfile(grammarName + "/bingrammargflexabstract.gf.lex"):
        print "Replacing Lexicon by Part-of-Speech tags"
        p = subprocess.Popen(
            ["mv", grammarName + "/bingrammargflexabstract.gf", grammarName + "/bingrammargflexabstract.gf.lex"])
        p.communicate()
        p = subprocess.Popen(
            ["mv", grammarName + "/bingrammargflexconcrete.gf", grammarName + "/bingrammargflexconcrete.gf.lex"])
        p.communicate()
        new_lex_files(grammarName + "/bingrammargflexabstract.gf.lex",
                      grammarName + "/bingrammargflexabstract.gf",
                      grammarName + "/bingrammargflexconcrete.gf")

    #
    print "Compiling grammar with gf"
    p = subprocess.Popen(["gf", "-make", "-probs=" + grammarName + "/bingrammargf.probs", "--cpu", "-D", grammarName, grammarName + "/bingrammargfconcrete.gf"])
    p.communicate()
    gr = pgf.readPGF(grammarName + "/bingrammargfabstract.pgf")

    if os.path.isfile(test):
        if forceRecompile or not os.path.isfile(grammarName + "/parse-results.conll"):
            print "Parsing test sentences"
            failures, time = parse(gr, test, grammarName + "/parse-results.conll", v)
            print "Parse time", time
            print "Parse failures", failures

    #
    print "eval.pl", "no punctuation"
    p = subprocess.Popen(["perl", "../util/eval.pl", "-g", test, "-s", grammarName + "/parse-results.conll", "-q"])
    p.communicate()
    print "eval.pl", "punctation"
    p = subprocess.Popen(["perl", "../util/eval.pl", "-g", test, "-s", grammarName + "/parse-results.conll", "-q", "-p"])
    p.communicate()



if __name__ == '__main__':
    plac.call(main)
    if False:
        lexpath = "/home/kilian/uni/implementation/rparse/bin_grammar_small.gf/bingrammargflex"
        # abstract = "abstract.gf.old"
        # concrete = "concrete.gf.old"
        # suffix = ".new"
        # lexabstract = lexpath + abstract

        # if False:
            # new_lex_files(lexabstract, lexpath + concrete, suffix)

        gr = pgf.readPGF("/home/kilian/uni/implementation/rparse/bin_grammar_small.gf/bingrammargfabstract.pgf")

        # lcfrs = gr.languages['bingrammargfconcrete']

        print parse_with_pgf(gr, [], "ADJD ADV _COMMA_ KOUS ADV PIS PROAV VVINF VMFIN _PUNCT_".split(' '))

        parse(gr, "/home/kilian/uni/implementation/rparse/negra-lower-punct-test.conll", "/tmp/parse-results.conll")
