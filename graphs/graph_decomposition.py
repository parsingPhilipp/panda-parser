from __future__ import print_function
from graphs.dog import DirectedOrderedGraph, DeepSyntaxGraph
from grammar.lcfrs import LCFRS, LCFRS_lhs, LCFRS_var
from grammar.lcfrs_derivation import LCFRSDerivation
from grammar.induction.decomposition import join_spans
from copy import deepcopy


def upward_closure(dog, nodes):
    """
    :param dog:
    :type dog:
    :param nodes:
    :type nodes:
    :return:
    :rtype:
    The name is a bit outdated. Actually we perform an upward closure on normal edges and a downward closure
    on strange edges [see the comment below.]
    """
    assert isinstance(dog, DirectedOrderedGraph)

    # First we need to compute strange edges, i.e., the smallest set U of edges s.t.
    #   - every edge, which has some input and where every primary input is the output of an edge in U, is in U.
    # In particular, edges with some input but without primary inputs are in U.
    strange = dog.internal_edges_without_primary_input()
    changed = True
    while changed:
        changed = False
        for edge in dog.terminal_edges:
            if edge not in strange \
                    and edge.inputs != [] \
                    and all([any([edge.inputs[i] == strange_edge.outputs[0] for strange_edge in strange])
                             for i in edge.primary_inputs]):
                strange.append(edge)
                changed = True
    strange_outs = [edge.outputs[0] for edge in strange]

    # Compute the actual node closure
    closure = list(nodes)
    changed = True
    while changed:
        changed = False
        for edge in dog.terminal_edges:
            if ((edge.inputs != []                           # consider only non-leaves
                        and all([edge.inputs[i] in closure  # who have all their primary predecessors in the closure
                                 or edge.inputs[i] in strange_outs # or in strange
                                 for i in edge.primary_inputs])
                        # and at least one primary input not not in strange
                        and any([edge.inputs[i] in closure for i in edge.primary_inputs]))
                    # alternatively: if edge in strange and its output is in the closure
                    or (edge in strange and edge.outputs[0] in closure)):
                # do an upward closure on outputs of edge
                for node in edge.outputs:
                    if node not in closure:
                        closure.append(node)
                        changed = True
                # do a downward closure on strange, primary inputs of edge
                for i in edge.primary_inputs:
                    if edge.inputs[i] in strange_outs and edge.inputs[i] not in closure:
                        closure.append(edge.inputs[i])
    closure.sort()
    return closure


def compute_decomposition(dsg, recursive_partitioning):
    nodes = [node for sent_pos in recursive_partitioning[0] for node in dsg.get_graph_position(sent_pos)]
    closed_nodes = upward_closure(dsg.dog, nodes)
    return closed_nodes, list(map(lambda rp: compute_decomposition(dsg, rp), recursive_partitioning[1]))


def induction_on_a_corpus(dsgs, rec_part_strategy, nonterminal_labeling, terminal_labeling, start="START",
                          normalize=True):
    grammar = LCFRS(start=start)
    for dsg in dsgs:
        rec_part = rec_part_strategy(dsg)
        # if calc_fanout(rec_part) > 1 or calc_rank(rec_part) > 2:
        #     rec_part = rec_part_strategy(dsg)
        #     assert False
        decomp = compute_decomposition(dsg, rec_part)
        dsg_grammar = induce_grammar_from(dsg, rec_part, decomp, nonterminal_labeling, terminal_labeling, terminal_labeling, start,
                                          normalize)
        grammar.add_gram(dsg_grammar)
    return grammar


def consecutive_spans(positions):
    if len(positions) == 0:
        return 0
    positions = sorted(list(positions))
    pos = positions[0]
    spans = 1
    i = 1
    while i < len(positions):
        if pos + 1 < positions[i]:
            spans += 1
        pos = positions[i]
        i += 1
    return spans


def calc_fanout(rec_part):
    node, children = rec_part
    return max([consecutive_spans(node)] + [f for f in map(calc_fanout, children)])


def calc_rank(rec_part):
    _, children = rec_part
    return max([1, len(children)] + [r for r in map(calc_rank, children)])


def induce_grammar_from(dsg, rec_par, decomp, labeling=(lambda x, y: str(x)), terminal_labeling=id, terminal_labeling_lcfrs=None, start="START",
                        normalize=True, enforce_outputs=True):
    if terminal_labeling_lcfrs is None:
        terminal_labeling_lcfrs = terminal_labeling
    lcfrs = LCFRS(start=start)
    ordered_nodes = dsg.dog.ordered_nodes()
    rhs_nont = induce_grammar_rec(lcfrs, dsg, rec_par, decomp, labeling, terminal_labeling, terminal_labeling_lcfrs
                                  , normalize, enforce_outputs, ordered_nodes=ordered_nodes)
    rhs_top = dsg.dog.top(decomp[0])

    # construct a chain rule from START to initial nonterminal of decomposition
    # LCFRS part
    lcfrs_lhs = LCFRS_lhs(start)
    lcfrs_lhs.add_arg([LCFRS_var(0, 0)])

    # DOG part
    dog = DirectedOrderedGraph()
    assert len(dsg.dog.inputs) == 0
    assert not enforce_outputs or len(dsg.dog.outputs) > 0
    for i in range(len(rhs_top)):
        dog.add_node(i)
    for output in dsg.dog.outputs:
        dog.add_to_outputs(rhs_top.index(output))
    dog.add_nonterminal_edge([], [i for i in range(len(rhs_top))], enforce_outputs)

    # no sync
    sync = []
    lcfrs.add_rule(lcfrs_lhs, [rhs_nont], weight=1.0, dcp=[dog, sync])

    return lcfrs


def induce_grammar_rec(lcfrs, dsg, rec_par, decomp, labeling, terminal_labeling, terminal_labeling_lcfrs, normalize,
                       enforce_outputs=True, ordered_nodes=None):
    lhs_nont = labeling(decomp[0], dsg)

    # build lcfrs part
    lcfrs_lhs = LCFRS_lhs(lhs_nont)
    rhs_sent_pos = map(lambda x: x[0], rec_par[1])
    generated_sent_positions = fill_lcfrs_lhs(lcfrs_lhs, rec_par[0], rhs_sent_pos, dsg.sentence, terminal_labeling_lcfrs)

    # build dog part
    rhs_nodes = list(map(lambda x: x[0], decomp[1]))
    dog = dsg.dog.extract_dog(decomp[0], rhs_nodes, enforce_outputs, ordered_nodes=ordered_nodes)
    for edge in dog.terminal_edges:
        edge.label = terminal_labeling(edge.label)
    if normalize:
        node_renaming = dog.compress_node_names()
    else:
        node_renaming = {}

    # build terminal synchronization
    sync = [[node_renaming.get(node, int(node)) for node in dsg.get_graph_position(sent_position)]
            for sent_position in generated_sent_positions]

    # recursively compute rules for rhs
    rhs_nonts = []
    for child_rec_par, child_decomp in zip(rec_par[1], decomp[1]):
        rhs_nonts.append(
            induce_grammar_rec(lcfrs, dsg, child_rec_par, child_decomp, labeling, terminal_labeling,
                               terminal_labeling_lcfrs, normalize, enforce_outputs, ordered_nodes=ordered_nodes))

    # create rule
    lcfrs.add_rule(lcfrs_lhs, rhs_nonts, weight=1.0, dcp=[dog, sync])
    return lhs_nont


def fill_lcfrs_lhs(lhs, sent_positions, children, sentence, terminal_labeling):
    """
    Create the LCFRS_lhs of some LCFRS-DCP hybrid rule.
    :rtype: list[int]
    :param tree:     HybridTree
    :param node_ids: list of string (node in an recursive partitioning)
    :param t_max:    top_max of node_ids
    :param b_max:    bottom_max of node ids
    :param children: list of pairs of list of list of string
#                    (pairs of top_max / bottom_max of child nodes in recursive partitioning)
    :type nont_labelling: AbstractLabeling
    :return:         List of sentence position generated by this rule in ascending order.
    """
    spans = join_spans(sent_positions)
    children_spans = list(map(join_spans, children))
    generated_sentence_positions = []

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
                arg.append(terminal_labeling(sentence[i]))
                generated_sentence_positions.append(i)
                i += 1
                # raise Exception('Expected ingredient for LCFRS argument was not found.')
        lhs.add_arg(arg)

    return generated_sentence_positions


def dog_evaluation(derivation, compress=True):
    assert isinstance(derivation, LCFRSDerivation)
    dog, sync = dog_evaluation_rec(derivation, derivation.root_id())

    if compress:
        renaming = dog.compress_node_names()
        for sent_pos in sync:
            sync[sent_pos] = [renaming.get(node, node) for node in sync[sent_pos]]

    sync_list = list(map(lambda x: x[1], sorted([(key, sync[key]) for key in sync], key=lambda x: x[0])))

    return dog, sync_list


def dog_evaluation_rec(derivation, idx):
    rule = derivation.getRule(idx)
    host_graph = deepcopy(rule.dcp()[0])

    generated_sent_positions = derivation.terminal_positions(idx)
    assert len(generated_sent_positions) == len(rule.dcp()[1])

    sync = {}
    for sent_position, graph_nodes in zip(generated_sent_positions, rule.dcp()[1]):
        sync[sent_position] = graph_nodes

    for i, child_idx in enumerate(derivation.child_ids(idx)):
        child_graph, child_sync = dog_evaluation_rec(derivation, child_idx)
        renaming = host_graph.replace_by(i, child_graph)

        for sent_position in child_sync:
            assert sent_position not in sync
            sync[sent_position] = [renaming.get(node, node) for node in child_sync[sent_position]]

    return host_graph, sync


def simple_labeling(nodes, dsg, edge_label=lambda e: e.label):
    return top_bot_labeling(nodes, dsg, top_label=edge_label, bot_label=edge_label)


def top_bot_labeling(nodes, dsg, top_label=lambda e: e.label, bot_label=lambda e: e.label):
    assert isinstance(dsg, DeepSyntaxGraph)
    ordered_nodes = dsg.dog.ordered_nodes()
    top_label = [top_label(dsg.dog.incoming_edge(node)) for node in dsg.dog.top(nodes, ordered_nodes)]
    bot_label = [bot_label(dsg.dog.incoming_edge(node)) for node in dsg.dog.bottom(nodes, ordered_nodes)]
    fanout = consecutive_spans(dsg.covered_sentence_positions(nodes))
    return '[' + ','.join(bot_label) + ';' + ','.join(top_label) + '; f' + str(fanout) + ']'


def missing_child_labeling(nodes, dsg, edge_label=lambda e: e.label, child_label=lambda e, i: e.label):
    assert isinstance(dsg, DeepSyntaxGraph)
    ordered_nodes = dsg.dog.ordered_nodes()
    top_label = [edge_label(dsg.dog.incoming_edge(node)) for node in dsg.dog.top(nodes, ordered_nodes)]
    bot_label = ['-'.join([child_label(dsg.dog.incoming_edge(node), i) for node, i in nodes2])
                 for nodes2 in dsg.dog.missing_children(nodes, ordered_nodes)]
    fanout = consecutive_spans(dsg.covered_sentence_positions(nodes))
    return '[' + ','.join(bot_label) + ';' + ','.join(top_label) + '; f' + str(fanout) + ']'


__all__ = ["induction_on_a_corpus", "compute_decomposition", "upward_closure", "calc_fanout", "calc_rank",
           "consecutive_spans", "dog_evaluation", "induce_grammar_from", "missing_child_labeling", "simple_labeling",
           "top_bot_labeling"]