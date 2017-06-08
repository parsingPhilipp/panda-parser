from __future__ import print_function
from grammar.lcfrs import *
from grammar.linearization import Enumerator
from subprocess import call
from math import isnan
import os

LANGUAGE = "LCFRS"
SUFFIX = ".gf"
COMPILED_SUFFIX = ".pgf"
PROBS_SUFFIX = ".probs"


def export(grammar, prefix, name, override=False):
    """
    :param grammar:
    :type grammar: LCFRS
    :param prefix:
    :type prefix:
    :param name:
    :type name:
    :return:
    :rtype:
    """
    nonterminals = Enumerator()
    # rules = Enumerator()

    name_ = name
    # do not overwrite existing grammar files
    if not override:
        i = 1
        while os.path.isfile(os.path.join(prefix, name_ + SUFFIX)):
            i += 1
            name_ = name + '_' + str(i)
        name = name_

    nan_probs = 0

    with open(os.path.join(prefix, name + SUFFIX), 'w') as abstract \
            , open(os.path.join(prefix, name + LANGUAGE + SUFFIX), 'w') as concrete \
            , open(os.path.join(prefix, name + PROBS_SUFFIX), 'w') as probs:

        def print_nont(nont):
            return "Nont" + str(nonterminals.object_index(nont))

        # init abstract
        abstract.write("abstract " + name + ' = {\n\n')
        abstract.write("  cat\n")

        # init concrete
        concrete.write("concrete " + name + LANGUAGE + " of " + name + " = {\n\n")

        # iterate over nonterminals = categories
        for nont in grammar.nonts():
            # define nonterminal
            abstract.write("    " + print_nont(nont) + " ;\n")
            # define fanout
            if grammar.fanout(nont) > 1:
                concrete.write("  lincat " + print_nont(nont) + " = { " + ' ; '.join(
                    ['s' + str(i) + ' : Str' for i in range(grammar.fanout(nont))]) + " } ; \n\n")
            else:
                concrete.write("  lincat " + print_nont(nont) + " = Str ; \n\n")

        abstract.write("\n  fun\n")
        concrete.write("\n  lin\n")

        # iterate over rules
        for rule in grammar.rules():
            id = rule.get_idx() # rules.object_index(rule)

            def transform_def(lhs, i):
                def to_string(x):
                    if isinstance(x, LCFRS_var):
                        if grammar.fanout(rule.rhs_nont(x.mem)) > 1:
                            return 'rhs' + str(x.mem) + '.s' + str(x.arg)
                        else:
                            return 'rhs' + str(x.mem)
                    else:
                        return '"' + x.replace('"', r'\"') + '"'

                return ' ++ '.join(
                    [to_string(x) for x in lhs.arg(i)]
                )

            def transform(lhs):
                """
                :param lhs:
                :type lhs: LCFRS_lhs
                :return:
                :rtype:
                """

                return ' ; '.join([
                                      "s" + str(i) + " = " +
                                      transform_def(lhs, i)
                                      for i in range(lhs.fanout())
                                      ])

            # define rtg rule
            abstract.write("    Func" + str(id) + " : ")
            for rhs in rule.rhs():
                abstract.write(print_nont(rhs) + " -> ")
            abstract.write(print_nont(rule.lhs().nont()) + " ;\n")

            # define interpretation
            concrete.write("    Func" + str(id) + ' '
                           + ' '.join(["rhs" + str(i) for i in range(rule.rank())])
                           + " = ")
            if grammar.fanout(rule.lhs().nont()) > 1:
                concrete.write("{ "
                               + transform(rule.lhs())
                               + " } ; \n")
            else:
                concrete.write(transform_def(rule.lhs(), 0) + " ; \n")

            # define probability
            if not isnan(rule.weight()):
                probs.write("Func" + str(id) + " " + str(rule.weight()) + "\n")
            else:
                nan_probs += 1
                probs.write("Func" + str(id) + " 0.0" + "\n")

        abstract.write("\n  flags startcat = " + print_nont(grammar.start()) + ";\n")
        abstract.write("\n}\n")
        concrete.write("\n}\n")

        if nan_probs > 0:
            print(nan_probs, "occurrences of rule weight NaN replaced by 0.0 during gf export")

    return name # rules, name


def compile_gf_grammar(prefix, name):
    #TODO 1) it seems that some additional arguments could make gf compile faster
    #TODO 2) compile to other grammars than "/tmp/tmpGrammarLCFRS.gf"
    #TODO 3) adding "shell=True" and joining the arguments to one string solved some
    #TODO "IOError: [Errno 2] No such file or directory" on an arch linux machine
    #TODO I (@kilian) have no idea why
    return call(' '.join(["gf", "-make", "-D", prefix, "--probs=" + os.path.join(prefix, name + PROBS_SUFFIX)
                , "+RTS", "-K100M", "-RTS" # compilation of large grammars requires larger run-time stack
                , os.path.join(prefix, name + LANGUAGE + SUFFIX)]), shell=True)
