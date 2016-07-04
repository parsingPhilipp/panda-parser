from LCFRS.lcfrs import *
from sDCP.dcp import *


def linearize(grammar, nonterminal_labeling, terminal_labeling):
    """
    :type grammar: LCFRS
    :param nonterminal_labeling:
    :param terminal_labeling:
    :return:
    """
    print "Nonterminal Labeling: ", nonterminal_labeling
    print "Terminal Labeling: ", terminal_labeling
    print

    terminals = Enumerator()
    nonterminals = Enumerator()

    for i, rule in enumerate(grammar.rules()):
        rid = 'r'+str(i+1)
        print rid, 'RTG   ', nonterminals.object_index(rule.lhs().nont()), '->',
        print map(lambda nont: nonterminals.object_index(nont), rule.rhs()), ';'
        #for rhs_nont in rule.rhs():
        #    print nonterminals[rhs_nont],
        print rid , 'WEIGHT', rule.weight(), ';'

        sync_index = {}
        inh_args = defaultdict(lambda: 0)
        for dcp in rule.dcp():
            inh_args[dcp.lhs().mem()] += 1
        for dcp in rule.dcp():
            printer = OUTPUT_DCP(terminals.object_index, rule, sync_index, inh_args)
            printer.evaluateList(dcp.rhs())
            var = dcp.lhs()
            var_string = 's<' + str(var.mem() + 1) + "," + str(var.arg() + 1) + ">"
            print rid, 'sDCP  ', var_string, '==', printer.string, ';'

        s = 0
        for j, arg in enumerate(rule.lhs().args()):
            print rid, 'LCFRS ', 's<0,' + str(j + 1) + '>', '==', '[',
            first = True
            for a in arg:
                if not first:
                    print ",",
                if isinstance(a, LCFRS_var):
                    print "x<{0!s},{1!s}>".format(a.mem + 1, a.arg + 1),
                    pass
                else:
                    if s in sync_index:
                        print str(terminals.object_index(a)) + '^{' + str(sync_index[s]) +'}',
                    else:
                        print str(terminals.object_index(a)),
                    s += 1
                first = False
            print '] ;'
        print

    print "Terminals: "
    terminals.print_index()
    print

    print "Nonterminals: "
    nonterminals.print_index()
    print

    print "Initial nonterminal: ", nonterminals.object_index(grammar.start())

class Enumerator:
    def __init__(self):
        self.counter = 0
        self.obj_to_ind = {}
        self.ind_to_obj = {}

    def index_object(self, i):
        """
        :type i: int
        :return:
        """
        return self.ind_to_obj[i]

    def object_index(self, obj):
        i = self.obj_to_ind.get(obj, None)
        if i:
            return i
        else:
            self.counter += 1
            self.obj_to_ind[obj] = self.counter
            self.ind_to_obj[self.counter] = obj
            return self.counter

    def print_index(self):
        for i in range (1, self.counter + 1):
            print i, self.index_object(i)


class DCP_Labels(DCP_evaluator):
    def evaluateString(self, s, id):
        self.labels.add(s)

    def evaluateTerm(self, term, id):
        """
        :type term: DCP_term
        :param id:
        :return:
        """
        term.head().evaluateMe(self)
        for child in term.arg():
            child.evaluateMe(self)

    def evaluateIndex(self, index, id):
        """
        :type index: DCP_index
        :param id:
        :return:
        """
        self.labels.add(index.dep_label())

    def evaluateVariable(self, var, id):
        pass

    def __init__(self):
        self.labels = set()

class OUTPUT_DCP(DCP_evaluator):
    def evaluateVariable(self, var, id):
        self.string += 'x<' + str(var.mem() + 1) + "," + str(var.arg() + 1 - self.inh_args[var.mem()]) + "> "

    def __init__(self, terminal_to_index, rule, sync_index, inh_args):
        self.terminal_to_index = terminal_to_index
        self.rule = rule
        self.string = ''
        self.sync_index = sync_index
        self.inh_args = inh_args

    def evaluateString(self, s, id):
        self.string += str(self.terminal_to_index[s]) + ' '

    def evaluateTerm(self, term, id):
        term.head().evaluateMe(self)
        self.string += '('
        self.evaluateList(term.arg())
        self.string += ') '

    def evaluateList(self, list):
        self.string += "[ "
        first = True
        for arg in list:
            if not first:
                self.string += ', '
            arg.evaluateMe(self)
            first = False
        self.string += "]"

    def evaluateIndex(self, index, id):
        if not index.index() in self.sync_index:
            self.sync_index[index.index()] = len(self.sync_index) + 1
        i = 0
        for arg in self.rule.lhs().args():
            for obj in arg:
                if not isinstance(obj, LCFRS_var):
                    if i == index.index():
                        self.string += str(self.terminal_to_index(obj + '::' + index.dep_label())) + '^{' + str(self.sync_index[index.index()]) + '}'
                        return
                    else:
                        i += 1