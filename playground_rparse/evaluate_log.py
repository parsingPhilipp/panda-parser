import re
import os
from collections import defaultdict


def average_fanout(path):
    with open(path) as concrete:
        count = defaultdict(lambda: 0)
        for line in concrete:
            match = re.match(r'^lincat [^\s]+ = {(\s*p\d+ : Str ;\s*)*\s*p(\d+) : Str } ;\s*$', line)
            if match:
                count[match.group(2)] += 1
        nonterminals = sum(count.values())
        fanout_total = sum(map(lambda x: int(x) * count[x], count.keys()))
        return nonterminals, fanout_total, fanout_total * 1.0 / nonterminals

def process_log(path, the_name):
    assert os.path.isfile(path)
    vMarkov = None
    hMarkov = None
    rules = None
    nonterminals = None
    fanout = None
    parse_failures = None
    parse_time = None
    punctuation = None
    uas_p, las_p, lac_p = None, None, None
    uas, las, lac = None, None, None

    with open(path) as log:
        for line in log:
            match = re.search(r'Extracting grammar with rparse', line)
            if match:
                vMarkov = None
                hMarkov = None
                rules = None
                nonterminals = None
                fanout = None
                parse_failures = None
                parse_time = None
                punctuation = None
                uas_p, las_p, lac_p = None, None, None
                uas, las, lac = None, None, None
                continue

            match = re.search(r'^  vMarkov            : (\d+)\s*$', line)
            if match:
                vMarkov = match.group(1)
                if vMarkov == '2147483647':
                    vMarkov = '\\infty'
                continue

            match = re.search(r'^  hMarkov            : (\d+)\s*$', line)
            if match:
                hMarkov = match.group(1)
                if hMarkov == '2147483647':
                    hMarkov = '\\infty'
                continue

            match = re.search(r'^Clauses: (\d+)\s*$', line)
            if match:
                rules = "{:,}".format(int(match.group(1)))
                continue

            match = re.search(r'^Labels: (\d+), thereof (\d+) preterminals\s*$', line)
            if match:
                nonterminals = "{:,}".format(int(match.group(1)) - int(match.group(2)))
                continue

            match = re.search(r'^Max arity: (\d+)\s*$', line)
            if match:
                fanout = match.group(1)
                continue

            match = re.search(r'^Replacing Lexicon by Part-of-Speech tags\s*$', line)
            if match:
                if vMarkov == '\\infty':
                    vMarkov_ = 'infinity'
                else:
                    vMarkov_ = vMarkov
                if hMarkov == '\\infty':
                    hMarkov_ = 'infinity'
                else:
                    hMarkov_ = hMarkov
                _, _, avg_fanout = average_fanout('.' + the_name + '_v' + vMarkov_ + '_h' + hMarkov_ + '/bingrammargfconcrete.gf')
                print '\t&\t'.join([the_name, '$' + vMarkov + '$', '$' + hMarkov + '$', nonterminals, rules, fanout, percentify(avg_fanout, 2)]),
                continue

            match = re.search(r'^Parse time (\d+\.\d+)\s*$', line)
            if match:
                parse_time = "{:,.0f}".format(float(match.group(1)))
                continue

            match = re.search('^Parse failures (\d+)\s*$', line)
            if match:
                parse_failures = match.group(1)
                continue

            match = re.search(r'^\s*eval\.pl no punctuation\s*$', line)
            if match:
                punctuation = False
                continue

            match = re.search(r'^\s*eval\.pl punctu?ation\s*$', line)
            if match:
                punctuation = True
                continue

            match = re.search(r'^\s*Labeled\s*attachment\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%\s*$', line)
            if match:
                if punctuation:
                    las_p = percentify(match.group(1), precision)
                else:
                    las = percentify(match.group(1), precision)
                continue


            match = re.search(r'^\s*Unlabeled\s*attachment\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%\s*$', line)
            if match:
                if punctuation:
                    uas_p = percentify(match.group(1), precision)
                else:
                    uas = percentify(match.group(1), precision)
                continue


            match = re.search(r'^\s*Label\s*accuracy\s*score:\s*\d+\s*/\s*\d+\s*\*\s*100\s*=\s*(\d+\.\d+)\s*%\s*$', line)
            if match:
                if punctuation:
                    lac_p = percentify(match.group(1), precision)
                    print '\t&\t',
                    print '\t&\t'.join([parse_time, parse_failures, uas_p, las_p, lac_p, uas, las, lac]),
                    print "\\\\"
                else:
                    lac = percentify(match.group(1), precision)
                continue


precision = 1


def percentify(value, precicion):
    p = float(value)
    return ("{:." + str(precicion) + "f}").format(p)


if __name__ == '__main__':
    path = '/home/kilian/uni/hybrid-grammars-python/playground_rparse/negra_experiments.log'
    process_log(path, 'negra')
    print
    print
    path = '/home/kilian/uni/hybrid-grammars-python/playground_rparse/tiger_experiments.log'
    process_log(path, 'tiger')