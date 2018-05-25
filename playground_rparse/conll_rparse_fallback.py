import re
import sys
try:
    import string
    no_translation = string.maketrans("", "")

    def is_punctuation(form):
        # this is string.punctuation with $, % removed (which are PMOD, NMOD, COORD, NMOD with dependents in WSJ)
        return not str(form).translate(no_translation, '!"&()*+#,/-:.?;<=>@[\\]^_{|}~')
        # we allow the dollar sign $ and the quotation marks `` and ''
except AttributeError:
    def is_punctuation(form):
        return not str(form).translate(str.maketrans("", "", '!"&()*+#,/-:.?;<=>@[\\]^_{|}~'))


def match_line(line):
    match = re.search(r'^([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)'
                      r'\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)$', line)
    return match


def left_branch(x):
    return x - 1


def right_branch(x):
    return x + 1


def all_root(x):
    return 0


def conll_to_rparse_input(input, output):
    with open(input) as input_file, open(output, 'w') as output_file:
        while True:
            try:
                line = next(input_file)
                while line.startswith('#'):
                    line = next(input_file)
            except StopIteration:
                break

            match = match_line(line)
            if match:
                form = match.group(2)
                pos = match.group(5)
                output_file.write(form + '/' + pos + '\n')
            elif re.search(r'^[^\s]*$', line):
                output_file.write('\n')
            else:
                print(line)
                raise Exception("unexpected input")


def filter_conll_by_length(input, output, length, ignore_punctation):
    with open(input) as input_file, open(output, 'w') as output_file:
        while True:
            try:
                line = next(input_file)
                while line.startswith('#g'):
                    line = next(input_file)
            except StopIteration:
                break
            tmp = []
            counter = 0
            match = match_line(line)
            while match:
                form = match.group(2)
                if not ignore_punctation or not is_punctuation(form):
                    counter += 1
                tmp.append(line)

                try:
                    line = next(input_file)
                    while line.startswith('#g'):
                        line = next(input_file)
                except StopIteration:
                    line = ''
                match = match_line(line)
            tmp.append('\n')

            if counter > 0 and counter <= length:
                output_file.write(''.join(tmp))


def fallback_fill_conll_results(path, gold_path, extended_path, limit=sys.maxsize):
    """
    :param path: path to corpus
    :type: str
    :param limit: stop generation after limit trees
    :type: int
    :return: a series of hybrid trees read from file
    :rtype: __generator[HybridTree]
    :raise Exception: unexpected input in corpus file
    Lazily parses a dependency corpus (in CoNLL format) and generates GeneralHybridTrees.
    """

    # print path

    parse_failures = 0
    strategy = all_root

    with open(path) as file_content, open(gold_path) as gold_file_content, open(extended_path, 'w') as output_file:
        tree_count = 0
        test_eof = False

        while tree_count < limit:
            try:
                gold_line = gold_file_content.next()
                while gold_line.startswith('#g'):
                    gold_line = gold_file_content.next()
            except StopIteration:
                break
            try:
                line = file_content.next()
                while line.startswith('#'):
                    line = file_content.next()
            except StopIteration:
                line = ''
                test_eof = True

            match = match_line(line)
            match_gold = match_line(gold_line)
            if match and match_gold:
                if match.group(1) == match_gold.group(1):
                    if match.group(8) != 'TOP':
                        output_file.write(line)
                    else:
                        # TOP -> ROOT (strange rparse behaviour)
                        delimiter = '\t'
                        node_id = match.group(1)
                        form = match.group(2)
                        lemma = match.group(3)
                        cpos = match.group(4)
                        pos = match.group(5)
                        feats = match.group(6)
                        parent = match.group(7)
                        deprel = match.group(8)
                        s = ''
                        s += node_id + delimiter
                        s += form + delimiter
                        s += lemma + delimiter
                        s += cpos + delimiter
                        s += pos + delimiter
                        s += feats + delimiter

                        s += parent + delimiter
                        s += "ROOT" + delimiter
                        s += parent + delimiter
                        s += "ROOT" + '\n'
                        output_file.write(s)
                    continue
                else:
                    print(line)
                    print(gold_line)
                    raise Exception("Unexpected input in CoNLL corpus file.")

            match = re.search(r'^[^\s]*$', line)
            gold_match = re.search(r'^[^\s]*$', gold_line)
            if match and gold_match:
                output_file.write("\n")
                continue

            try:
                line = file_content.next()
                match = re.search(r'^[^\s]*$', line)
                while match or line.startswith('#'):
                    line = file_content.next()
                    match = re.search(r'^[^\s]*$', line)
            except StopIteration:
                line = ''

            match = re.search(r'^\s*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\s\d+:\sNo\sparse\sfound\s*$', line)
            gold_match = match_line(gold_line)
            # Produce fall-back
            if (match or test_eof) and gold_match:
                parse_failures += 1
                while gold_match:
                    node_id = gold_match.group(1)
                    form = gold_match.group(2)
                    lemma = gold_match.group(3)
                    cpos = gold_match.group(4)
                    pos = gold_match.group(5)
                    feats = gold_match.group(6)
                    # parent = gold_match.group(7)
                    # deprel = gold_match.group(8)

                    try:
                        gold_line = gold_file_content.next()
                        while gold_line.startswith('#'):
                            gold_line = gold_file_content.next()
                        gold_match = match_line(gold_line)
                    except StopIteration:
                        gold_line = ''
                        gold_match = None

                    if strategy == right_branch and gold_match is None:
                        the_parent = 0
                    else:
                        the_parent = strategy(int(node_id))

                    the_deprel = "_"

                    delimiter = '\t'
                    s = ''
                    s += node_id + delimiter
                    s += form + delimiter
                    # TODO the database does not store these fields of a CoNLLToken yet,
                    # TODO but eval.pl rejects to compare two tokens if they differ
                    # TODO extend the database, then fix this
                    s += lemma + delimiter
                    s += cpos + delimiter
                    s += pos + delimiter
                    s += feats + delimiter

                    s += str(the_parent) + delimiter
                    s += the_deprel + delimiter
                    s += str(the_parent) + delimiter
                    s += the_deprel + '\n'
                    if not gold_match:
                        s += '\n'

                    output_file.write(s)
                try:
                    line = file_content.next()
                except StopIteration:
                    line = ''
            else:
                print("test", line)
                print("gold", gold_line)
                raise Exception()

    print("Parse failures: ", parse_failures)


if __name__ == '__main__':
    test = '/home/kilian/mnt/tulip/compute/kilian/rparse/negra_parse_results_v2_h2.conll'
    gold = '/home/kilian/uni/dependency_conll/german/negra/negra_test.conll'
    output = '/tmp/negra_fallback_completed.conll'
    fallback_fill_conll_results(test, gold, output)

    # input = '/home/kilian/uni/dependency_conll/german/tiger/test/german_tiger_test.conll'
    # output = '/tmp/tiger_test.foo'
    # test = '/home/kilian/mnt/tulip/compute/kilian/rparse/tiger_parse_results_v2_hinf_2nd_try.conll'
    # gold = '/home/kilian/mnt/tulip/compute/kilian/rparse/tiger_smaller_20.conll'
    # output = '/tmp/tiger_fallback_v2_hinf_2nd_try.conll'
    # output = '/tmp/tiger_smaller_20.rparse'
    # conll_to_rparse_input(input, output)
    # filter_conll_by_length(input, output, 20, True)
    # test = '/home/kilian/mnt/tulip/compute/kilian/rparse/tiger_parse_results_v2_hinf.conll'
    # gold = '/tmp/tiger_smaller_20.conll'
    # output = '/tmp/tiger_fallback.conll'
    # fallback_fill_conll_results(test, gold, output)


    # input = '/home/kilian/uni/implementation/rparse/negra-lower-punct-test.conll'
    # output = '/home/kilian/uni/implementation/rparse/negra-lower-punct-test.rparse'
    # conll_to_rparse_input(input, output)
