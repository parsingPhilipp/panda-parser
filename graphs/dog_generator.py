from graphs.dog import DirectedOrderedGraph, DeepSyntaxGraph
from random import shuffle, randint, choice, random


def generate(n_nodes, maximum_inputs=4, upward_closed=True, new_output=0.2, multiple_output=0.1):
    dog = DirectedOrderedGraph()
    for i in range(n_nodes):
        dog.add_node(i)

    nodes = range(n_nodes)
    connected = set()
    targeted = set()
    if upward_closed:
        output = choice(nodes)
        dog.add_to_outputs(output)
        connected.add(output)
    while len(targeted) != n_nodes:
        if dog.outputs and random() <= multiple_output:
            output = choice(dog.outputs)
            dog.add_to_outputs(output)
        if random() <= new_output or (upward_closed and targeted == connected):
            output = choice([node for node in nodes if node not in targeted])
            dog.add_to_outputs(output)
            connected.add(output)
        if upward_closed:
            goal = choice(list(connected.difference(targeted)))
            assert goal not in targeted
        else:
            goal = choice([x for x in range(n_nodes) if x not in targeted])
            assert goal not in targeted
        n_inputs = randint(0, maximum_inputs)
        inputs = []
        for _ in range(n_inputs):
            node = choice(nodes)
            inputs.append(node)
            connected.add(node)
        edge = dog.add_terminal_edge(inputs, randint(0, n_nodes), goal)
        targeted.add(goal)
    return dog


def generate_sdg(n_nodes, maximum_inputs=4, upward_closed=False, new_output=0.1, multiple_output=0.0):
    dog = generate(n_nodes, maximum_inputs, upward_closed, new_output, multiple_output)
    sentence = [dog.incoming_edge(node) for node in dog.nodes]
    sync = [[node] for node in dog.nodes]
    dsg = DeepSyntaxGraph(sentence, dog, sync)
    return dsg


__all__ = ["generate", "generate_sdg"]
