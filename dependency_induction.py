__author__ = 'kilian'

from general_hybrid_tree import GeneralHybridTree
from dcp import *
from lcfrs import *
from decomposition import join_spans
from parsing import LCFRS_parser

# Compute list of node ids that delimit id_set from the top.
# tree: GeneralHybridTree
# id_set: list of string  (node ids)
# return: list of string  (node ids)
def top(tree, id_set):
    top_nodes = [id for id in id_set if tree.parent(id) not in id_set]
    return top_nodes

# Compute list of node ids that delimit id_set from the bottom.
# id_set: list of string  (node ids)
# return: list of string  (node ids)
def bottom(tree, id_set):
    bottom_nodes = [id for id in tree.id_yield()
                    if tree.parent(id) in id_set and id not in id_set]
    return bottom_nodes

# Group maximal subsets of neighbouring nodes together.
# tree: GeneralHybridTree
# id_set: list of string
# return: list of list of string
def max(tree, id_set):
    nodes = id_set[:]
    max_list = []
    while len(nodes) > 0:
        id = nodes[0]

        # Assume that the following two lists contain tree nodes, that are
        # siblings, ordered from left to right.
        all_siblings = tree.siblings(id)
        present_siblings = [id for id in all_siblings if id in nodes]

        nodes = [id for id in nodes if id not in present_siblings]

        while len(present_siblings) > 0:
            i = all_siblings.index(present_siblings[0])
            j = 1
            while j < len(present_siblings) and all_siblings[i + j] == present_siblings[j]:
                j += 1
            max_list += [present_siblings[:j]]
            present_siblings = present_siblings[j:]

    return max_list

def top_max(tree, id_set):
    return max(tree, top(tree, id_set))

def bottom_max(tree, id_set):
    return max(tree, bottom(tree, id_set))


#   Create DCP equation for some synthesized attributes of LHS nont
#   or inherited attributes of RHS nont of an LCFRS-sDCP-hybrid rule.
#   mem: int                            (member part of attribute: -1 for LHS, >=0 for RHS)
#   arg: int                            (argument part of attribute: >= 0)
#   top_max:    list of list of string  (top_max of nont on LHS)
#   bottom_max: list of list of string  (bottom_max of nont on LHS)
#   childern: list of pairs of list of list of string
#                                       (pair of (top_max, bottom_max) for every nont on RHS)
#   return: DCP_rule
def create_DCP_rule(mem, arg, top_max, bottom_max, children):
    lhs = DCP_var(mem, arg)
    rhs = []
    if mem < 0:
        conseq_ids = top_max[arg-len(bottom_max)][:]
    else:
        conseq_ids = children[mem][1][arg][:]
    while conseq_ids:
        id = conseq_ids[0]
        if mem >= 0:
            c_index = -1
        else:
            c_index = 0
        match = False
        while c_index < len(children) and not match:
            if c_index >= 0:
                child = children[c_index]
            else:
                # If equation for inherited arguments of some nont on RHS is computed,
                # the inherited arguments of the LHS are used in addition.
                # The second component is empty, which allows for some magic below!
                child = (bottom_max, [])
            t_seq_index = 0
            while t_seq_index < len(child[0]) and not match:
                t_seq = child[0][t_seq_index]
                # check if correct child synthesized attribute was found
                if id == t_seq[0]:
                    # sanity check, is t_seq a prefix of conseq_ids
                    if conseq_ids[:len(t_seq)] != t_seq:
                        raise Exception
                    # Append variable corresponding to synthesized attribute of nont on RHS.
                    # Or, append variable corresponding to inherited attribute of nont on LHS,
                    # where len(child[1]) evaluates to 0 as intended.
                    rhs.append(DCP_var(c_index, len(child[1]) + t_seq_index))
                    # remove matched prefix from conseq_ids
                    conseq_ids = conseq_ids[len(t_seq):]
                    # exit two inner while loops
                    match = True
                t_seq_index += 1
            c_index += 1
        # Sanity check, that attribute was matched:
        if not match:
            raise Exception
    return DCP_rule(lhs, rhs)

def create_LCFRS_lhs(tree, node_ids, children):
    positions = map(tree.node_index, node_ids)
    spans = join_spans(positions)

    children_spans = map(join_spans, [map(tree.node_index, ids) for (ids,_) in children])

    lhs = LCFRS_lhs(nonterminal_str(node_ids))
    for (low, high) in spans:
        arg = []
        i = low
        while (i <= high):
            mem = 0
            match = False
            while mem < len(children_spans) and not match:
                child_spans = children_spans[mem]
                mem_arg = 0
                while mem_arg < len(child_spans) and not match:
                    child_span = child_spans[mem_arg]
                    if child_span[0] == i:
                        arg.append(LCFRS_var(mem, mem_arg))
                        i = child_span[1] + 1
                        match = True
                    mem_arg += 1
                mem += 1
            # Sanity check
            if not match:
                raise Exception
        lhs.add_arg(arg)

    return lhs



# Creates a DCP rule for a leaf of the recursive partitioning.
# Note that the linked LCFRS-rule has an empty RHS.
# If the corresponding node in the hybrid tree is not a leaf, i.e. bottom_max != [],
# there is exactly one inherited argument <0,0>. The only synthesized argument
# generates a DCP_term of the form "[0]( <0,0> ).
# Otherwise, there is no inherited argument, and the only synthesized argument
# generates a DCP_term of the form "[0]( )".
# bottom_max: list of list of string
def create_leaf_DCP_rule(bottom_max):
    if bottom_max:
        arg = 1
    else:
        arg = 0
    lhs = DCP_var(-1, arg)
    term_head = DCP_index(0)
    if bottom_max:
        term_arg = [DCP_var(-1, 0)]
    else:
        term_arg = []
    rhs = [DCP_term(term_head, term_arg)]
    return DCP_rule(lhs, rhs)

def create_leaf_LCFRS_lhs(tree, node_ids):
    # Build LHS
    lhs = LCFRS_lhs(nonterminal_str(node_ids))
    id = node_ids[0]
    arg = [tree.node_label(id)]
    lhs.add_arg(arg)
    return lhs
#
# return: pair of list of list of string
def add_rules_to_grammar_rec(tree, rec_par, grammar):
    (node_ids, children) = rec_par

    # Sanity check
    if children and len(node_ids) == 1:
        raise Exception

    if len(node_ids) == 1:
        t_max = top_max(tree, node_ids)
        b_max = bottom_max(tree, node_ids)

        dcp = [create_leaf_DCP_rule(b_max)]
        lhs = create_leaf_LCFRS_lhs(tree, node_ids)

        grammar.add_rule(lhs, [], 1.0, dcp)

        return (t_max, b_max)

    else:
        # Create rules for children and obtain top_max and bottom_max of child nodes
        child_t_b_max = []
        for child in children:
            child_t_b_max.append(add_rules_to_grammar_rec(tree, child, grammar))

        # construct dcp equations
        dcp = []
        t_max = top_max(tree, node_ids)
        b_max = bottom_max(tree, node_ids)

        # create equations for synthesized attributes on LHS
        for arg in range(len(t_max)):
            dcp.append(create_DCP_rule(-1, len(b_max) + arg, t_max, b_max, child_t_b_max))

        # create equations for inherited attributes on RHS
        for cI in range(len(child_t_b_max)):
            child = child_t_b_max[cI]
            for arg in range(len(child[1])):
                dcp.append(create_DCP_rule(cI, arg, t_max, b_max, child_t_b_max))

         # create lcfrs-rule, attach dcp and add to grammar
        lhs = create_LCFRS_lhs(tree, node_ids, children)
        rhs = map(nonterminal_str,[ids for (ids, _) in children])

        grammar.add_rule(lhs, rhs, 1.0, dcp)

        return (t_max, b_max)

def nonterminal_str(nont):
    return str(nont).replace('[','{').replace(']','}').replace(' ','')

def test_dependency_induction():
    tree = GeneralHybridTree()
    tree.add_node("v1",'Piet',"NP",True)
    tree.add_node("v21",'Marie',"N",True)
    tree.add_node("v",'helpen',"V",True)
    tree.add_node("v2",'lezen', "V", True)
    tree.add_child("v","v2")
    tree.add_child("v","v1")
    tree.add_child("v2","v21")
    tree.set_root("v")
    tree.reorder()
    print tree.children("v")

    for id_set in ['v v1 v2 v21'.split(' '), 'v1 v2'.split(' '),
                   'v v21'.split(' '), ['v'], ['v1'], ['v2'], ['v21']]:
        print id_set, 'top:', top(tree, id_set), 'bottom:', bottom(tree, id_set)
        print id_set, 'top_max:', max(tree, top(tree, id_set)), 'bottom_max:', max(tree, bottom(tree, id_set))

    print "some rule"
    for mem, arg in [(-1, 0), (0,0), (1,0)]:
        print create_DCP_rule(mem, arg, top_max(tree, ['v','v1','v2','v21']), bottom_max(tree, ['v','v1','v2','v21']),
                   [(top_max(tree, l), bottom_max(tree, l)) for l in [['v1', 'v2'], ['v', 'v21']]])

    print "some other rule"
    for mem, arg in [(-1,1),(1,0)]:
        print create_DCP_rule(mem, arg, top_max(tree, ['v1','v2']), bottom_max(tree, ['v1','v2']),
                   [(top_max(tree, l), bottom_max(tree, l)) for l in [['v1'], ['v2']]])

    rec_par = ('v v1 v2 v21'.split(' '),
                    [('v1 v2'.split(' '), [(['v1'],[]), (['v2'],[])])
                    ,('v v21'.split(' '), [(['v'],[]), (['v21'],[])])
                    ])

    grammar = LCFRS(nonterminal_str(rec_par[0]))

    add_rules_to_grammar_rec(tree, rec_par, grammar)

    grammar.make_proper()
    print grammar

    parser = LCFRS_parser(grammar, 'Piet Marie helpen lezen'.split(' '))
    parser.print_parse()

    parser.new_DCP_Hybrid_Tree('P M h l'.split(' '), 'Piet Marie helpen lezen'.split(' '))

test_dependency_induction()