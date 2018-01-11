from collections import defaultdict


class RTG:
    def __init__(self, initial):
        self._initial = initial
        self._rules = []
        self._nonterminals = {initial}
        self._lhs_nont_to_rules = defaultdict(list)

    def add_rule(self, rule):
        self._rules.append(rule)
        for nont in [rule.lhs] + rule.rhs:
            self._nonterminals.add(nont)
        self._lhs_nont_to_rules[rule.lhs].append(rule)

    def construct_and_add_rule(self, lhs, symbol, rhs):
        self.add_rule(RTGRule(lhs, symbol, rhs))

    @property
    def initial(self):
        return self._initial

    @property
    def rules(self):
        return self._rules

    @property
    def nonterminals(self):
        return self._nonterminals

    def lhs_nont_to_rules(self, nont):
        return self._lhs_nont_to_rules[nont]


class RTGRule:
    def __init__(self, lhs, symbol, rhs):
        self._lhs = lhs
        self._symbol = symbol
        self._rhs = rhs

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def symbol(self):
        return self._symbol