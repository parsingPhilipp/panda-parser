from __future__ import print_function, unicode_literals
from grammar.lcfrs import LCFRS, LCFRS_lhs, LCFRS_var
from discodop.grammar import sortgrammar
from discodop.tree import escape
from discodop.containers import Grammar
from discodop.plcfrs import parse


def transform_grammar(grammar):
    # TODO assert ordered rules, terminals only in rules with len(rhs) = 0
    for rule in grammar.rules():
        fake_nont = rule.lhs().nont() + "-" + str(rule.get_idx())
        trans_rule_fake = (rule.lhs().nont(), fake_nont), tuple([(0,) for _ in rule.lhs().args()])
        yield trans_rule_fake, rule.weight()
        rhs = rule.rhs() if rule.rhs() else ['Epsilon']
        trans_rule = tuple([fake_nont] + rhs), transform_args(rule.lhs().args())
        yield trans_rule, 1.0


def transform_args(args):
    def transform_arg(arg):
        arg_new = []
        for elem in arg:
            if isinstance(elem, LCFRS_var):
                arg_new.append(elem.mem)
            else:
                assert len(arg) == 1
                return escape(elem)
        return tuple(arg_new)
    return tuple([transform_arg(arg) for arg in args])