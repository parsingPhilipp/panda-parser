from grammar.rtg import RTG
import re

def read_rtg(path):
    with open(path) as file:
        first = True
        rtg = None
        for line in file.readlines():
            if first:
                rtg = RTG(line)
                first = False
            else:
                match = re.search(r'^(\(.*\)) -> (\d*)\((.*)\) #\d(\.\d*)?$', line)
                if match:
                    lhs_tmp = match.group(1)
                    symbol = int(match.group(2))
                    rhs_tmp = match.group(3)
                    rhs = []

                    match2 = re.search(r'^(\(.*\), )+(\(.*\))$', rhs_tmp)
                    while match2:
                        rhs.append(match2.group(2))
                        assert len(match2.group(1)) > 1
                        rhs_tmp = match2.group(1)[:-2]
                        match2 = re.search(r'^(\(.*\), )(\(.*\))$', rhs_tmp)
                    match2 = re.search(r'^(\(.*\))$', rhs_tmp)
                    if match2:
                        rhs.append(match2.group(1))
                    rhs.reverse()
                    # TODO: split the nonterminals in tuples of original nonterminal and annotation (graph ids)
                    rtg.construct_and_add_rule(lhs_tmp, symbol, rhs)
                else:
                    raise IOError("Could not parse line " + line)
        return rtg
