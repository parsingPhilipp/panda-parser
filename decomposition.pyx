import random
from dependency.top_bottom_max import top_max, bottom_max

# Auxiliary routines for dealing with spans in an input string.

# ################################################################


# For list of indices, order and join into contiguous sequences.
# A sequence is represented by (low, high), where high is last
# element.
# indices: list of int
# return: list of pair of int
def join_spans(indices):
    indices = sorted(set(indices))
    spans = []
    low = -1
    for i in indices:
        if low < 0:
            low = i
            high = i
        elif i == high + 1:
            high = i
        else:
            spans += [(low, high)]
            low = i
            high = i
    if low >= 0:
        spans += [(low, high)]
    return spans


# For a list of spans, replace by indices.
# spans: list of pair of int
# return: list of int
def expand_spans(spans):
    return sorted(set([i for span in spans \
                       for i in range(span[0], span[1] + 1)]))


###############################################################

# Recursive partitioning of input string into substrings.
# A recursive partitioning is represented as a pair of ints
# and a list of recursive partitionings.

# E.g. (set([0, 1]), [(set([0]), []), (set([1]), [])])
# len: int
# return: recursive partitioning
def left_branching_partitioning(len):
    if len == 0:
        return set(), []
    elif len == 1:
        return {0}, []
    else:
        return (set(range(len)), [ \
            left_branching_partitioning(len - 1), \
            ({len - 1}, [])])


def right_branching_partitioning(len):
    return right_branching_partitioning_recur(0, len)


def right_branching_partitioning_recur(low, high):
    if low >= high:
        return set(), []
    elif low == high - 1:
        return {low}, []
    else:
        return (set(range(low, high)), [ \
            ({low}, []), \
            right_branching_partitioning_recur(low + 1, high)])


# Transform existing partitioning to limit number of
# spans.
# Breadth-first search among descendants for subpartitioning
# that stays within fanout.
# part: recursive partitioning
# fanout: int
# return: recursive partitioning
def fanout_limited_partitioning(part, fanout):
    (root, children) = part
    agenda = children[::-1]  # reversed to favour left branching
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                child2 = restrict_part([(rest, children)], rest)[0]
                child1_restrict = fanout_limited_partitioning(child1, fanout)
                child2_restrict = fanout_limited_partitioning(child2, fanout)
                return root, sort_part(child1_restrict, child2_restrict)
            else:
                next_agenda += subchildren[::-1]  # reversed
        agenda = next_agenda
    return part


# Transform existing partitioning to limit number of
# spans.
# Breadth-first search among descendants for subpartitioning
# that stays within fanout.
# left-to-right breadth first-search instead of right-to-left
# part: recursive partitioning
# fanout: int
# return: recursive partitioning
def fanout_limited_partitioning_left_to_right(part, fanout):
    (root, children) = part
    agenda = children  
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                child2 = restrict_part([(rest, children)], rest)[0]
                child1_restrict = fanout_limited_partitioning_left_to_right(child1, fanout)
                child2_restrict = fanout_limited_partitioning_left_to_right(child2, fanout)
                return root, sort_part(child1_restrict, child2_restrict)
            else:
                next_agenda += subchildren
        agenda = next_agenda
    return part


# Transform existing partitioning to limit number of
# spans.
# Choose position p such that p = argmax |part(p)|
# part: recursive partitioning
# fanout: int
# return: recursive partitioning
def fanout_limited_partitioning_argmax(part, fanout):
    (root, children) = part
    if children == []:
        return part
    agenda = children
    argmax = None
    argroot = {}
    argchildren = []
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                    if argmax == None or len(subroot) > len(argroot):
                        argmax = child1
                        (argroot, argchildren) = argmax
            else:
                next_agenda += subchildren
        agenda = next_agenda
    rest = remove_spans_from_spans(root, argroot)
    child2 = restrict_part([(rest, children)], rest)[0]
    child1_restrict = fanout_limited_partitioning_argmax(argmax, fanout)
    child2_restrict = fanout_limited_partitioning_argmax(child2, fanout)
    return root, sort_part(child1_restrict, child2_restrict)



# Transform existing partitioning to limit number of
# spans.
# Choose position p such that no new nonterminal is added to grammar if possible.
# Use right-to-left bfs as fallback.
# part: recursive partitioning
# fanout: int
# tree: HybridTree
# nont_labelling: AbstractLabeling
# nonts: list of nonterminals
# return: recursive partitioning
def fanout_limited_partitioning_no_new_nont(part, fanout, tree, nonts, nont_labelling):
    (root, children) = part
    agenda = children
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                subindex = []
                for pos in subroot:
                    subindex += [tree.index_node(pos+1)]
                positions = map(int, subroot)
                b_max = bottom_max(tree, subindex)
                t_max = top_max(tree, subindex)
                spans = join_spans(positions)
                nont = nont_labelling.label_nonterminal(tree, subindex, t_max, b_max, len(spans))
                if nont in nonts:
                    child2 = restrict_part([(rest, children)], rest)[0]
                    child1_restrict = fanout_limited_partitioning_no_new_nont(child1, fanout, tree, nonts, nont_labelling)
                    child2_restrict = fanout_limited_partitioning_no_new_nont(child2, fanout, tree, nonts, nont_labelling)
                    return root, sort_part(child1_restrict, child2_restrict)
            next_agenda += subchildren
        agenda = next_agenda
    agenda = children[::-1]  # reversed to favour left branching
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                child2 = restrict_part([(rest, children)], rest)[0]
                child1_restrict = fanout_limited_partitioning(child1, fanout)
                child2_restrict = fanout_limited_partitioning(child2, fanout)
                return root, sort_part(child1_restrict, child2_restrict)
            else:
                next_agenda += subchildren[::-1]  # reversed
        agenda = next_agenda
    return part


# Transform existing partitioning to limit number of
# spans.
# Choose subpartitioning that stays within fanout randomly.
# part: recursive partitioning
# fanout: int
# return: recursive partitioning
def fanout_limited_partitioning_random_choice(part, fanout):
    (root, children) = part
    agenda = children
    possibleChoices = []
    while len(agenda) > 0:
        next_agenda = []
        while len(agenda) > 0:
            child1 = agenda[0]
            agenda = agenda[1:]
            (subroot, subchildren) = child1
            rest = remove_spans_from_spans(root, subroot)
            if n_spans(subroot) <= fanout and n_spans(rest) <= fanout:
                possibleChoices += [child1]
            next_agenda += subchildren
        agenda = next_agenda
    if possibleChoices == []:
        return part
    chosen = random.choice(possibleChoices)
    chosen = tuple(chosen)
    (subroot, subchildren) = chosen
    rest = remove_spans_from_spans(root, subroot)
    child2 = restrict_part([(rest, children)], rest)[0]
    child1_restrict = fanout_limited_partitioning_random_choice(chosen, fanout)
    child2_restrict = fanout_limited_partitioning_random_choice(child2, fanout)
    return root, sort_part(child1_restrict, child2_restrict)



# With spans2 together covering a subset of what spans1 covers,
# remove spans2 from spans1. 
# spans1, spans2: list of int
# return: set of int
def remove_spans_from_spans(spans1, spans2):
    set1 = set(spans1)
    set2 = set(spans2)
    return set1 - set2


# Keep only partitionings that overlap with relevant set.
# Restrict to that set.
# part: recursive partitioning
# relevant: list of int
# return: list of recursive partitioning
def restrict_part(part, relevant):
    part_restrict = []
    for (root, children) in part:
        root_restrict = root & relevant
        if root_restrict != set():
            children_restrict = restrict_part(children, relevant)
            if len(children_restrict) == 1 and \
                            children_restrict[0][0] == root_restrict:
                part_restrict += [children_restrict[0]]
            else:
                part_restrict += [(root_restrict, children_restrict)]
    return part_restrict


# Number of spans.
# l: list of int
# return: int
def n_spans(l):
    return len(join_spans(l))


# For two disjoint partitionings, determine which one comes first.
# This is determined by first position.
# part1, part2: recursive partitioning
# return: list of two recursive partitionings
def sort_part(part1, part2):
    (root1, _) = part1
    (root2, _) = part2
    if sorted(root1)[0] < sorted(root2)[0]:
        return [part1, part2]
    else:
        return [part2, part1]


# For debugging, print partitioning.
# part: recursive partitioning
# level: int
def print_partitioning(part, level=0):
    (root, children) = part
    print ' ' * level, root
    for child in children:
        print_partitioning(child, level + 1)
