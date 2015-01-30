__author__ = 'kilian'

from parser.derivation_interface import AbstractDerivation

class Derivation(AbstractDerivation):
    def gorn_delimiter(self):
        return '.'

    def gorn_delimiter_regex(self):
        return '\.'

    def __init__(self):
        self.__rules = {}
        self.__root = ''
        self.__weights = {}

    # add a rule to the derivation at position id
    # id : string (Gorn position / identifier)
    # rule: Rule_instance
    def add_rule(self, id, rule, weight):
        self.__rules[id] = rule
        self.__weights[id] = weight

    def getRule(self, id):
        return self.__rules[id]

    # id : string
    # return: list of rules
    def children(self, id):
        return [self.getRule(id + self.gorn_delimiter() + str(i))
                for i in range(self.getRule(id).rule().rank())]

    def child_ids(self, id):
        return [id + self.gorn_delimiter() + str(i) for i in range(self.getRule(id).rule().rank())]

    def root_id(self):
        return self.__root

    def root(self):
        return (self.getRule(''))

    def __str__(self):
        return der_to_str(self)

    def terminal_positions(self, id):
        child_positions = []
        for child in self.child_ids(id):
            child_positions += self.__all_positions(child)
        return [p for p in self.__all_positions(id) if p not in child_positions]

    def __all_positions(self, id):
        rule = self.getRule(id)
        positions = []
        for i in range(rule.lhs().fanout()):
            span = rule.lhs().arg(i)[0]
            positions += range(span.low() + 1, span.high() + 1)
        return positions

    def ids(self):
        return self.__rules.keys()

# return string
def der_to_str(der):
    return der_to_str_rec(der, der.root_id())

# return: string
def der_to_str_rec(der, id):
    s = ' ' * len(id) + str(der.getRule(id).rule()) + '\t(' + str(der.getRule(id).lhs()) + ')\n'
    for child in der.child_ids(id):
        s += der_to_str_rec(der, child)
    return s