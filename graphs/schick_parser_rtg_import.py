from grammar.rtg import RTG
import re

NONTERMINAL_EXPRESSION = re.compile(r'^\((.*), (<(H(\d)+, )*H\d+>)\)$')


def read_nonterminal(nonterminal):
    match = NONTERMINAL_EXPRESSION.search(nonterminal)
    if match:
        return match.group(1), match.group(2)
    else:
        return nonterminal


def read_rtg(path):
    with open(path) as file:
        first = True
        rtg = None
        for line in file.readlines():
            if first:
                rtg = RTG(read_nonterminal(line))
                first = False
            else:
                match = re.search(r'^(\(.*\)) -> (\d*)\((.*)\) #\d(\.\d*)?$', line)
                if match:
                    lhs_tmp = match.group(1)
                    lhs = read_nonterminal(lhs_tmp)
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
                    rtg.construct_and_add_rule(lhs, symbol, list(map(read_nonterminal, rhs)))
                else:
                    raise IOError("Could not parse line " + line)
        return rtg


__all__ = ["read_rtg"]
