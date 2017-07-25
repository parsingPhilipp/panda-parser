from graphs.dog import DirectedOrderedGraph, DeepSyntaxGraph

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