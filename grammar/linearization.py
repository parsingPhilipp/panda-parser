from __future__ import print_function
from sys import stdout

from grammar.lcfrs import *
from grammar.dcp import *
from collections import defaultdict


def linearize(grammar, nonterminal_labeling, terminal_labeling, file):
    """
    :type grammar: LCFRS
    :param nonterminal_labeling:
    :param terminal_labeling:
    :return:
    """
    print("Nonterminal Labeling: ", nonterminal_labeling, file=file)
    print("Terminal Labeling: ", terminal_labeling, file=file)
    print(file=file)

    terminals = Enumerator(file)
    nonterminals = Enumerator(file)
    num_inherited_args = {}
    num_synthezied_args = {}

    for i, rule in enumerate(grammar.rules()):
        rid = 'r'+str(i+1)
        print(rid, 'RTG   ', nonterminals.object_index(rule.lhs().nont()), '->', file=file, end=" ")
        print(list(map(lambda nont: nonterminals.object_index(nont), rule.rhs())), ';', file=file)
        #for rhs_nont in rule.rhs():
        #    print nonterminals[rhs_nont],
        print(rid , 'WEIGHT', rule.weight(), ';', file=file)

        sync_index = {}
        inh_args = defaultdict(lambda: 0)
        lhs_var_counter = CountLHSVars()
        synth_attributes = 0
        for dcp in rule.dcp():
            if dcp.lhs().mem() != -1:
                inh_args[dcp.lhs().mem()] += 1
            else:
                synth_attributes += 1
            inh_args[-1] += lhs_var_counter.evaluateList(dcp.rhs())
        num_inherited_args[nonterminals.object_index(rule.lhs().nont())] = inh_args[-1]
        num_synthezied_args[nonterminals.object_index(rule.lhs().nont())] = synth_attributes

        for dcp in rule.dcp():
            printer = OUTPUT_DCP(terminals.object_index, rule, sync_index, inh_args)
            printer.evaluateList(dcp.rhs())
            var = dcp.lhs()
            if var.mem() == -1:
                var_string = 's<0,' + str(var.arg() + 1 - inh_args[-1]) + ">"
            else:
                var_string = 's<' + str(var.mem() + 1) + "," + str(var.arg() + 1) + ">"
            print (rid, 'sDCP  ', var_string, '==', printer.string, ';', file=file)

        s = 0
        for j, arg in enumerate(rule.lhs().args()):
            print(rid, 'LCFRS ', 's<0,' + str(j + 1) + '>', '==', '[', end=' ', file=file)
            first = True
            for a in arg:
                if not first:
                    print(",", end=' ', file=file)
                if isinstance(a, LCFRS_var):
                    print("x<{0!s},{1!s}>".format(a.mem + 1, a.arg + 1), end=' ', file=file)
                    pass
                else:
                    if s in sync_index:
                        print(str(terminals.object_index(a)) + '^{' + str(sync_index[s]) +'}', end=' ', file=file)
                    else:
                        print(str(terminals.object_index(a)), end=' ', file=file)
                    s += 1
                first = False
            print('] ;', file=file)
        print(file=file)

    print("Terminals: ", file=file)
    terminals.print_index()
    print(file=file)

    print("Nonterminal ID, nonterminal name, fanout, #inh, #synth: ", file=file)
    max_fanout, max_inh, max_syn, max_args, fanouts, inherits, synths, args = nonterminals.print_index_and_stats(grammar, num_inherited_args, num_synthezied_args)
    print(file=file)
    print("max fanout:", max_fanout, file=file)
    print("max inh:", max_inh, file=file)
    print("max synth:", max_syn, file=file)
    print("max args:", max_args, file=file)
    print(file=file)
    for s, d, m in [('fanout', fanouts, max_fanout), ('inh', inherits, max_inh), ('syn', synths, max_syn), ('args', args, max_args)]:
        for i in range(m + 1):
            print('# the number of nonterminals with', s, '=', i, 'is', d[i], file=file)
        print(file=file)
    print(file=file)

    print("Initial nonterminal: ", nonterminals.object_index(grammar.start()), file=file)
    print(file=file)


class Enumerator:
    def __init__(self, file=stdout, first_index=1):
        self.first_index = first_index
        self.counter = first_index - 1
        self.obj_to_ind = {}
        self.ind_to_obj = {}
        self.file = file

    def index_object(self, i):
        """
        :type i: int
        :return:
        """
        return self.ind_to_obj[i]

    def object_index(self, obj):
        if obj in self.obj_to_ind:
            return self.obj_to_ind[obj]
        else:
            self.counter += 1
            self.obj_to_ind[obj] = self.counter
            self.ind_to_obj[self.counter] = obj
            return self.counter

    def print_index(self):
        for i in range(self.first_index, self.counter + 1):
            print(i, self.index_object(i), file=self.file)

    def print_index_and_stats(self, grammar, inh, syn):
        fanouts = defaultdict(lambda: 0)
        inherits = defaultdict(lambda: 0)
        synths = defaultdict(lambda: 0)
        args = defaultdict(lambda: 0)
        max_fanout = 0
        max_inh = 0
        max_syn = 0
        max_args = 0
        for i in range (self.first_index, self.counter + 1):
            fanout = grammar.fanout(self.index_object(i))
            fanouts[fanout] += 1
            max_fanout = max(max_fanout, fanout)
            inherits[inh[i]] += 1
            max_inh = max(max_inh, inh[i])
            synths[syn[i]] += 1
            max_syn = max(max_syn, syn[i])
            args[inh[i] + syn[i]] += 1
            max_args = max(max_args, inh[i] + syn[i])
            print(i, self.index_object(i), fanout, inh[i], syn[i], file=self.file)
        return max_fanout, max_inh, max_syn, max_args, fanouts, inherits, synths, args


class DCP_Labels(DCP_visitor):
    def visit_string(self, s, id):
        self.labels.add(s)

    def visit_term(self, term, id):
        """
        :type term: DCP_term
        :param id:
        :return:
        """
        term.head().visitMe(self)
        for child in term.arg():
            child.visitMe(self)

    def visit_index(self, index, id):
        """
        :type index: DCP_index
        :param id:
        :return:
        """
        self.labels.add(index.edge_label())

    def visit_variable(self, var, id):
        pass

    def __init__(self):
        self.labels = set()


class CountLHSVars(DCP_visitor):
    def visit_variable(self, var, id):
        if var.mem() == -1:
            return 1
        else:
            return 0

    def visit_string(self, s, id):
        return 0

    def visit_term(self, term, id):
        return term.head().visitMe(self) + self.evaluateList(term.arg())

    def evaluateList(self, xs):
        return sum([x.visitMe(self) for x in xs])

    def visit_index(self, index, id):
        return 0


class OUTPUT_DCP(DCP_visitor):
    def visit_variable(self, var, id):
        if (var.mem() != -1):
            self.string += 'x<' + str(var.mem() + 1) + "," + str(var.arg() + 1 - self.inh_args[var.mem()]) + "> "
        else:
            self.string += 'x<' + str(var.mem() + 1) + "," + str(var.arg() + 1) + "> "

    def __init__(self, terminal_to_index, rule, sync_index, inh_args):
        self.terminal_to_index = terminal_to_index
        self.rule = rule
        self.string = ''
        self.sync_index = sync_index
        self.inh_args = inh_args

    def visit_string(self, s, id):
        self.string += str(self.terminal_to_index[s]) + ' '

    def visit_term(self, term, id):
        term.head().visitMe(self)
        self.string += '('
        self.evaluateList(term.arg())
        self.string += ') '

    def evaluateList(self, list):
        self.string += "[ "
        first = True
        for arg in list:
            if not first:
                self.string += ', '
            arg.visitMe(self)
            first = False
        self.string += "]"

    def visit_index(self, index, id):
        if not index.index() in self.sync_index:
            self.sync_index[index.index()] = len(self.sync_index) + 1
        i = 0
        for arg in self.rule.lhs().args():
            for obj in arg:
                if not isinstance(obj, LCFRS_var):
                    if i == index.index():
                        self.string += str(self.terminal_to_index(obj + '::' + index.edge_label())) + '^{' + str(self.sync_index[index.index()]) + '}'
                        return
                    else:
                        i += 1