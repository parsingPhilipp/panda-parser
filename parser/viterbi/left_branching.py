from parser.viterbi.viterbi import ViterbiParser, Range, PassiveItem
from grammar.lcfrs import LCFRS, LCFRS_var
from sys import maxint
from math import log
import heapq


class ActiveItem(PassiveItem):
    def __init__(self, nonterminal, rule):
        PassiveItem.__init__(self, nonterminal, rule)
        self.next_high = None

    def next_nont(self):
        return self.rule.rhs_nont(- (len(self.children) + 1))

    def complete(self):
        """
        :rtype: bool
        """
        return len(self.children) == self.rule.rank()

    def merge_ranges(self):
        for i in range(len(self.ranges)):
            j = 0
            while j < len(self.ranges[i]) - 1:
                if isinstance(self.ranges[i][j], LCFRS_var) or isinstance(self.ranges[i][j+1], LCFRS_var):
                    j += 1
                else:
                # elif self.ranges[i][j].right == self.ranges[i][j+1].left:
                    self.ranges[i][j] = Range(self.ranges[i][j].left, self.ranges[i][j+1].right)
                    del self.ranges[i][j+1]
                    # self.ranges[i] = self.ranges[i][:j+1] + self.ranges[i][j+2:]
                #else:
                #    raise Exception()

    def copy(self):
        new_item = self.__class__(self.nonterminal, self.rule)
        new_item.weight = self.weight
        new_item.ranges = [list(rs) for rs in self.ranges]
        new_item.children = list(self.children)
        return new_item

    def replace_consistent(self, passive_item):
        """
        :type passive_item: PassiveItem
        :rtype: list[list[LCFRS_var|Range]], int, int
        """
        arg = self.rule.rank() - len(self.children) - 1
        new_ranges = []
        next_high = None
        pos = maxint
        for i in range(len(self.ranges) - 1, -1, -1):
            r = self.ranges[i]
            new_range = []
            gap = True
            for j in range(len(r) - 1, - 1, -1):
                elem = r[j]
                if isinstance(elem, Range):
                    if elem.right > pos:
                        return None, None, None
                    elif not gap and elem.right != pos:
                        return None, None, None
                    if not gap:
                        new_range[0] = Range(elem.left, new_range[0].right)
                    else:
                        new_range = [elem] + new_range
                    pos = elem.left
                    gap = False
                elif elem.mem == arg:
                    subst_range = passive_item.ranges[elem.arg]
                    if subst_range.right > pos:
                        return None, None, None
                    elif not gap and subst_range.right != pos:
                        return None, None, None
                    if not gap:
                        new_range[0] = Range(subst_range.left, new_range[0].right)
                    else:
                        new_range = [subst_range] + new_range
                        gap = False
                    pos = subst_range.left
                elif elem.mem == arg - 1 and elem.arg == 0:
                    next_high = pos
                    new_range = [elem] + new_range
                    gap = True
                else:
                    new_range = [elem] + new_range
                    gap = True
            new_ranges = [new_range] + new_ranges
        return new_ranges, next_high

    def make_passive(self):
        self.__class__ = PassiveItem
        del self.next_high
        i = 0
        while i < len(self.ranges):
            # assert len(self.ranges[i]) == 1
            self.ranges[i] = self.ranges[i][0]
            i += 1

    def __str__(self):
        return "[{0!s}] {1!s} [{2}]".format(self.weight, self.nonterminal, ', '.join(map(lambda r: "[{0}]".format(', '.join(map(str, r))), self.ranges)))

    def agenda_key(self):
        return id(self.rule), self.__ranges_to_tuple(), len(self.children)

    def agenda_key_earley(self):
        return id(self.rule), self.__ranges_to_tuple(), len(self.children), self.next_high

    def __ranges_to_tuple(self):
        return tuple([tuple(rs) for rs in self.ranges])

    def is_active(self):
        return True


def rule_to_active_item(rule, input, high):
    empty = ActiveItem(rule.lhs().nont(), rule)
    empty.weight = log(rule.weight())

    # TODO: We assume that LCFRS_var(0,0) occurs in the first component of the word tuple function

    empty.next_high = high
    for item in item_to_active_item_rec(empty, input, high, 0, len(rule.lhs().arg(0))):
        yield item


def item_to_active_item_rec(item, input, high, arg, pattern_pos):
    right = high
    while arg >= 0:
        pattern = item.rule.lhs().arg(arg)
        while pattern_pos > 0 and right > 0:
            i = 1
            while pattern_pos - i >= 0 and right - i >= 0:
                if pattern[pattern_pos - i] == input[right - i]:
                    i += 1
                else:
                    if isinstance(pattern[pattern_pos - i], LCFRS_var) and i > 1:
                        tmp_item = item.copy()
                        if len(tmp_item.ranges) == 0:
                            tmp_item.ranges = [[]] + tmp_item.ranges
                        if i > 1:
                            tmp_item.ranges[0] = [Range(right - i, right)] + tmp_item.ranges[0]
                        tmp_item.ranges[0] = [pattern[pattern_pos - i]] + tmp_item.ranges[0]
                        for new_item in item_to_active_item_rec(
                                tmp_item,
                                input,
                                right - i,
                                arg,
                                pattern_pos - i - 1):
                            yield new_item
                    else:
                        if not item.ranges:
                            item.ranges = [[]]
                        item.ranges[0] = [pattern[pattern_pos - i]] + item.ranges[0]
                        pattern_pos -= 1
                        right -= 1

        if pattern_pos != 0:
            return
        else:
            arg -= 1
    # print "Completely derived: ", item
    yield item

def rule_to_passive_items_lc_opt(rule, high):
    item = PassiveItem(rule.lhs().nont(), rule)
    item.weight = log(rule.weight())
    item.ranges.append(Range(high - 1, high))
    return item


class LeftBranchingParser(ViterbiParser):
    def key(self, x):
        return x.agenda_key_earley()

    # def __init__(self, grammar, input):
    #     """
    #     :type grammar: LCFRS
    #     :type input: list[str]
    #     """
    #     self.grammar = grammar
    #     self.input = input
    #     self.agenda = []
    #     self.active_chart = defaultdict(list)
    #     self.passive_chart = defaultdict(list)
    #     self.actives = defaultdict()
    #     self.passives = defaultdict()
    #     self.goal = None
    #
    #     self.key = lambda x: x.agenda_key_earley()
    #
    #     self.__parse_earley()

    def _parse(self):
        self.__lc__query = set()
        last_position = len(self.input)
        if last_position == 0:
            raise Exception("not implemented")
        for rule in self.grammar.left_branch_predict(self.grammar.start(), self.input[last_position - 1]):
            if rule.rhs():
                for active_item in rule_to_active_item(rule, self.input, last_position):
                    active_item.next_high = last_position
                    self._record_item(active_item)
            else:
                self._record_item(rule_to_passive_items_lc_opt(rule, last_position))

        self.__lc__query.add((self.grammar.start(), last_position))

        while self.agenda:
            item = heapq.heappop(self.agenda)
            if not item.valid:
                continue
            if item.is_active():
                # print "Process: ", item.next_high, item
                high = item.next_high
                key = item.next_nont(), high

                self.active_chart[key].append(item)

                if key not in self.__lc__query:
                    for rule in self.grammar.left_branch_predict(key[0], self.input[high - 1]):
                        if rule.rhs():
                            for active_item in rule_to_active_item(rule, self.input, high):
                                active_item.next_high = high
                                self._record_item(active_item)
                        else:
                            self._record_item(rule_to_passive_items_lc_opt(rule, high))
                    self.__lc__query.add(key)

                for passive_item in self.passive_chart.get(key, []):
                    self._combine(item, passive_item)

            else:  # if isinstance(item, PassiveItem):
                # STOPPING EARLY:
                # print "Process: ", item
                if item.nonterminal == self.grammar.start() and item.ranges[0] == Range(0, last_position):
                    self.goal = item
                    return
                key = item.nonterminal, item.right_position()

                self.passive_chart[key].append(item)

                for active_item in self.active_chart.get(key, []):
                    self._combine(active_item, item)

    def _combine(self, active_item, passive_item):
        ranges, next_high = active_item.replace_consistent(passive_item)
        if ranges:
            new_active = ActiveItem(active_item.nonterminal, active_item.rule)
            new_active.ranges = ranges
            new_active.next_high = next_high
            new_active.weight = active_item.weight + passive_item.weight
            new_active.children = [passive_item] + list(active_item.children)

            if new_active.complete():
                new_active.make_passive()

            self._record_item(new_active)

    @staticmethod
    def preprocess_grammar(grammar):
        """
        :type grammar: LCFRS
        """
        # precompute the left branching prediction table
        grammar.left_branch_predict(None, None)