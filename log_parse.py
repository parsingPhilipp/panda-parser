__author__ = 'kilian'
import re


def log_parse(path):
    """
    Lazily parses the logfile of an experiment and creates a latex tabular.
    :param path: path to corpus
    :raise Exception: unexpected input in corpus file
    """

    file_content = open(path)
    entries = 0

    while True:
        result = {}
        blank_line = True

        while blank_line:
            try:
                line = file_content.next()
            except StopIteration:
                return

            match = re.search(r'^\s*$', line)
            if not match:
                blank_line = False

        # match header
        match = re.search(r'^Inducing grammar$', line)
        if not match:
            raise Exception

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match file
        match = re.search(r'^file.*$', line)
        if not match:
            raise Exception

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match nont labelling strategy
        match = re.search(r'^Nonterminal labelling strategy:\s*(.*)\s*$', line)
        if not match:
            raise Exception
        else:
            nont_labelling = match.group(1)
            if nont_labelling == 'strict_pos':
                result['nont_labelling'] = 'strict'
            elif nont_labelling == 'child_pos':
                result['nont_labelling'] = 'child'
            else:
                result['nont_labelling'] = nont_labelling

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match nont labelling strategy
        match = re.search(r'^Terminal labelling strategy:\s*(.*)\s*$', line)
        if not match:
            raise Exception

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match rec partitioning
        match = re.search(r'^Recursive partitioning strategy:\s*(.*)\s*$', line)
        if not match:
            raise Exception
        else:
            rec_par = match.group(1)
            if rec_par == 'direct_extraction':
                result['rec_par'] = 'direct'
            else:
                result['rec_par'] = rec_par.replace('_', ' ')

        try:
            line = file_content.next()
            line = file_content.next()
            line = file_content.next()
        except StopIteration:
            break


        # match training-tree-count
        match = re.search(r'^Number of trees:\s*(.*)\s*$', line)
        if not match:
            raise Exception
        else:
            result['training_corpus'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break


        # match number of nonterm
        match = re.search(r'^Number of nonterimals:\s*(.*)\s*$', line)
        if not match:
            raise Exception
        else:
            result['n_nonterminals'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break


        # match number of rules
        match = re.search(r'^Number of rules:\s*(.*)\s*$', line)
        if not match:
            raise Exception
        else:
            result['n_rules'] = match.group(1)

        try:
            line = file_content.next()
            line = file_content.next()
        except StopIteration:
            break


        # match number of rules
        match = re.search(r'^Fanout:\s*(.*)\s*$', line)
        if not match:
            raise Exception
        else:
            result['fanout'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match number of rules
        match = re.search(r'^Induction time:\s*(.*)\s*seconds\s*$', line)
        if not match:
            raise Exception
        else:
            result['induction_time'] = match.group(1)

        try:
            line = file_content.next()
            line = file_content.next()
            line = file_content.next()
        except StopIteration:
            break

        # match test corpus size
        match = re.search(r'^Parsed\s*([0-9]+)\s*out of ([0-9]+) \(skipped ([0-9]+)\)\s*$', line)
        if not match:
            raise Exception
        else:
            result['test_total'] = match.group(2)
            result['test_succ'] = match.group(1)
            result['skipped'] = match.group(3)
        try:
            line = file_content.next()
        except StopIteration:
            break

        # match parse failures
        match = re.search(r'^fail:\s*([0-9]+)\s*$', line)
        if not match:
            raise Exception
        else:
            result['fail'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match parse failures
        match = re.search(r'^UAS:\s*([0-9]+(\.[0-9]+)?)\s*$', line)
        if not match:
            raise Exception
        else:
            result['UAS'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match parse failures
        match = re.search(r'^LAS:\s*([0-9]+(\.[0-9]+)?)\s*$', line)
        if not match:
            raise Exception
        else:
            result['LAS'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match parse failures
        match = re.search(r'^UEM:\s*([0-9]+(\.[0-9]+)?)\s*$', line)
        if not match:
            raise Exception
        else:
            result['UEM'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match parse failures
        match = re.search(r'^LEM:\s*([0-9]+(\.[0-9]+)?)\s*$', line)
        if not match:
            raise Exception
        else:
            result['LEM'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match parse failures
        match = re.search(r'^n gaps \(gold\):\s*([0-9]+(\.[0-9]+)?)\s*$', line)
        if not match:
            raise Exception
        else:
            result['n_gaps_gold'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match parse failures
        match = re.search(r'^n gaps \(test\):\s*([0-9]+(\.[0-9]+)?)\s*$', line)
        if not match:
            raise Exception
        else:
            result['n_gaps_test'] = match.group(1)

        try:
            line = file_content.next()
        except StopIteration:
            break

        # match parse failures
        match = re.search(r'^parse time:\s*([0-9]+)(\.[0-9]+)?\s*s\s*$', line)
        if not match:
            raise Exception
        else:
            result['parse_time'] = match.group(1)

        yield result


def test_log_parse():
    table_columns = ['nont_labelling', 'rec_par', 'training_corpus', 'n_nonterminals', 'n_rules', 'fanout', 'test_total',
                     'fail', 'UAS', 'LAS', 'n_gaps_test', 'n_gaps_gold', 'parse_time']
    header = {}
    header['nont_labelling'] = 'nont.~lab.'
    header['rec_par'] = 'extraction'
    header['training_corpus'] = 'training sent.'
    header['n_nonterminals'] = 'nont.'
    header['n_rules'] = 'rules'
    header['fanout'] = 'fanout'
    header['test_total'] = 'test sent.'
    header['test_succ'] = 'succ'
    header['fail'] = 'fail'
    header['UAS'] = 'UAS'
    header['LAS'] = 'LAS'
    header['n_gaps_test'] = '\\# gaps (test)'
    header['n_gaps_gold'] = '\\# gaps (gold)'
    header['parse_time']  = 'parse time (sec)'

    print '\\begin{tabular}{llllllllllllr}'
    print '\t\\toprule'
    print '\t' + ' & '.join([header[id] for id in table_columns]) + '\\\\'
    for line in log_parse('../../logs.txt'):
          print '\t' + ' & '.join([line[id] for id in table_columns]) + '\\\\'
    print '\t\\bottomrule'
    print '\\end{tabular}'
test_log_parse()