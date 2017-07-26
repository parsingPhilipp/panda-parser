from __future__ import print_function
from graphs.dog import DirectedOrderedGraph, DeepSyntaxGraph
from grammar.lcfrs import LCFRS, LCFRS_rule, LCFRS_lhs, LCFRS_var
from decomposition import join_spans

def upward_closure(dog, nodes):
    assert isinstance(dog, DirectedOrderedGraph)
    closure = list(nodes)
    changed = True
    while changed:
        changed = False
        for edge in dog.terminal_edges:
            if edge.inputs != [] and all([edge.inputs[i] in closure for i in edge.primary_inputs]):
                for node in edge.outputs:
                    if node not in closure:
                        closure.append(node)
                        changed = True
    closure.sort()
    return closure

def compute_decomposition(dsg, recursive_partitioning):
    nodes = [node for sent_pos in recursive_partitioning[0] for node in dsg.get_graph_position(sent_pos)]
    closed_nodes = upward_closure(dsg.dog, nodes)
    return closed_nodes, map(lambda rp: compute_decomposition(dsg, rp), recursive_partitioning[1])

def induce_grammar_from(dsg, rec_par, decomp, labeling=id, start="START", normalize=True):
    lcfrs = LCFRS(start=start)
    rhs_nont = induce_grammar_rec(lcfrs, dsg, rec_par, decomp, labeling, normalize)

    # construct a chain rule from START to initial nonterminal of decomposition
    lcfrs_lhs = LCFRS_lhs(start)
    lcfrs_lhs.add_arg([LCFRS_var(0, 0)])

    dog = DirectedOrderedGraph()
    dog.add_node(0)
    dog.add_nonterminal_edge([], [0])
    dog.add_to_outputs(0)

    lcfrs.add_rule(lcfrs_lhs, [rhs_nont], weight=1.0, dcp=[dog])

    return lcfrs

def induce_grammar_rec(lcfrs, dsg, rec_par, decomp, labeling, normalize):
    lhs_nont = labeling(decomp[0])

    # build lcfrs part
    lcfrs_lhs = LCFRS_lhs(lhs_nont)
    rhs_sent_pos = map(lambda x: x[0], rec_par[1])
    fill_lcfrs_lhs(lcfrs_lhs, rec_par[0], rhs_sent_pos, dsg.sentence)

    # build dog part
    rhs_nodes = map(lambda x: x[0], decomp[1])
    dog = dsg.dog.extract_dog(decomp[0], rhs_nodes)
    if normalize:
        dog.compress_node_names()

    # recursively compute rules for rhs
    rhs_nonts = []
    for child_rec_par, child_decomp in zip(rec_par[1], decomp[1]):
        rhs_nonts.append(induce_grammar_rec(lcfrs, dsg, child_rec_par, child_decomp, labeling, normalize))

    # create rule
    lcfrs.add_rule(lcfrs_lhs, rhs_nonts, weight=1.0, dcp=[dog])
    return lhs_nont

def fill_lcfrs_lhs(lhs, sent_positions, children, sentence):
    """
    Create the LCFRS_lhs of some LCFRS-DCP hybrid rule.
    :rtype: LCFRS_lhs
    :param tree:     HybridTree
    :param node_ids: list of string (node in an recursive partitioning)
    :param t_max:    top_max of node_ids
    :param b_max:    bottom_max of node ids
    :param children: list of pairs of list of list of string
#                    (pairs of top_max / bottom_max of child nodes in recursive partitioning)
    :type nont_labelling: AbstractLabeling
    :return: LCFRS_lhs :raise Exception:
    """
    spans = join_spans(sent_positions)

    children_spans = map(join_spans, children)

    for (low, high) in spans:
        arg = []
        i = low
        while i <= high:
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
            # Add terminal
            if not match:
                arg.append(sentence[i])
                i += 1
                # raise Exception('Expected ingredient for LCFRS argument was not found.')
        lhs.add_arg(arg)

    return lhs