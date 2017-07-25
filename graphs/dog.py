from copy import deepcopy

class Edge:
    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __init__(self, outputs, inputs, label=None):
        self._label = label
        self._outputs = outputs
        self._inputs = []
        self._primary_inputs = []
        for i, elem in enumerate(inputs):
            if isinstance(elem, tuple):
                if elem[1] == 'p':
                    self._primary_inputs.append(i)
                self._inputs.append(elem[0])
            else:
                self._inputs.append(elem)

    @property
    def label(self):
        return self._label

    @property
    def terminal(self):
        return self._label is not None

    def set_primary(self, i):
        assert i < len(self.inputs)
        self._primary_inputs.append(i)

    @property
    def primary_inputs(self):
        return self._primary_inputs

class DirectedOrderedGraph:
    def __init__(self):
        self._nodes = []
        self._inputs = []
        self._outputs = []
        self._terminal_edges = []
        self._nonterminal_edges = []
        self._parents = {}
        self._incoming_edge = {}

    @property
    def outputs(self):
        return self._outputs

    def incoming_edge(self, node):
        return self._incoming_edge[node]

    def children(self, node):
        edge = self._incoming_edge[node]
        if edge is None:
            return []
        else:
            return edge.inputs

    def type(self):
        _type = []
        for edge in self._nonterminal_edges:
            _type.append((len(edge.outputs), len(edge.inputs)))
        _type.append((len(self._inputs), len(self._outputs)))
        return _type

    @property
    def terminal_edges(self):
        return self._terminal_edges

    def _node_closure(self, function, reflexive=False):
        closure = {}

        if reflexive:
            for node in self._nodes:
                closure[node] = [node]
        else:
            for node in self._nodes:
                closure[node] = deepcopy(function(node))

        changed = True
        while changed:
            changed = False
            for node in self._nodes:
                for _node in closure[node]:
                    for __node in function(_node):
                        if __node not in closure[node]:
                            changed = True
                            closure[node].append(__node)
        return closure

    def cyclic(self):
        downward_closure = self._node_closure(lambda x: self.children(x))

        for node in self._nodes:
            if node in downward_closure[node]:
                return True

        return False

    def output_connected(self):
        upward_closure = self._node_closure(lambda n: self._parents[n], reflexive=True)

        for node in self._nodes:
            if not any([True for x in upward_closure[node] if x in self._outputs]):
                return False

        return True

    def add_node(self, node):
        assert node not in self._nodes
        self._nodes.append(node)
        self._parents[node] = []
        self._incoming_edge[node] = None

    def add_edge(self, edge):
        assert len(edge.outputs) > 0
        assert all([node in self._nodes for node in edge.inputs])
        assert all([node in self._nodes for node in edge.outputs])
        assert all([self._incoming_edge[output] is None for output in edge.outputs])
        for output in edge.outputs:
            self._incoming_edge[output] = edge
            for node in edge.inputs:
                if output not in self._parents[node]:
                    self._parents[node].append(output)
        if edge.terminal:
            self._terminal_edges.append(edge)
        else:
            self._nonterminal_edges.append(edge)

    def add_terminal_edge(self, inputs, label, output):
        self.add_edge(Edge([output], deepcopy(inputs), label))

    def add_nonterminal_edge(self, inputs, outputs):
        self.add_edge(Edge(deepcopy(outputs), deepcopy(inputs)))

    def add_to_inputs(self, node):
        assert self._incoming_edge[node] is None
        self._inputs.append(node)

    def add_to_outputs(self, node):
        self._outputs.append(node)

    @staticmethod
    def __replace_inplace(the_list, old, new):
        for i, elem in enumerate(the_list):
            if elem == old:
                the_list[i] = new

    def rename_node(self, node, node_new):
        if node == node_new:
            return
        assert node_new not in self._nodes
        self.__replace_inplace(self._nodes, node, node_new)
        self.__replace_inplace(self._inputs, node, node_new)
        self.__replace_inplace(self._outputs, node, node_new)

        self._incoming_edge[node_new] = self._incoming_edge[node]
        del self._incoming_edge[node]

        if node in self._parents:
            self._parents[node_new] = self._parents[node]
            del self._parents[node]
            for key in self._parents:
                self.__replace_inplace(self._parents[key], node, node_new)

        for edge in self._nonterminal_edges + self._terminal_edges:
            if edge is not None:
                self.__replace_inplace(edge.inputs, node, node_new)
                self.__replace_inplace(edge.outputs, node, node_new)

    def replace_by(self, i, dog):
        """
        :param i:
        :type i: int
        :param dog:
        :type dog: DirectedOrderedGraph
        :return:
        :rtype:
        Hyperedge replacement
        We assume that nodes are integers to make the renaming easier.
        The node that is inserted is assumed to have no nonterminal edges.
        """
        assert 0 <= i < len(self._nonterminal_edges)
        nt_edge = self._nonterminal_edges[i]
        assert isinstance(nt_edge, Edge)
        assert len(dog._inputs) == len(nt_edge.inputs)
        assert len(dog._outputs) == len(nt_edge.outputs)
        # print(len(dog._nonterminal_edges), all([edge is None for edge in dog._nonterminal_edges]))
        assert (len(dog._nonterminal_edges) == 0) or all([edge is None for edge in dog._nonterminal_edges])

        max_node = max(self._nodes)
        for j, node in enumerate(dog._nodes):
            dog.rename_node(node, max_node + 1 + j)

        for host_node, replace_node in zip(nt_edge.inputs, dog._inputs):
            dog.rename_node(replace_node, host_node)
        for host_node, replace_node in zip(nt_edge.outputs, dog._outputs):
            dog.rename_node(replace_node, host_node)

        # clean up old parent/ child database entries
        for node in nt_edge.inputs:
            self._parents[node] = [parent for parent in self._parents[node] if parent not in nt_edge.outputs]
        for node in nt_edge.outputs:
            self._incoming_edge[node] = None

        # add all new nodes
        for node in dog._nodes:
            if node > max_node:
                self.add_node(node)

        # add new edges
        for edge in dog._terminal_edges:
            self.add_edge(edge)

        # remove nonterminal_edge (replace by None!)
        self._nonterminal_edges[i] = None

    def ordered_nodes(self):
        ordered = []
        for root_node in self._outputs:
            self.__ordered_nodes_rec(ordered, root_node)
        return ordered

    def __ordered_nodes_rec(self, ordered, v):
        if v not in ordered:
            ordered.append(v)
            for v_2 in self.children(v):
                self.__ordered_nodes_rec(ordered, v_2)


    def compress_node_names(self):
        if self.ordered_nodes() == [i for i in range(len(self._nodes))]:
            return
        max_node = max(self._nodes) + 1
        for i, node in enumerate(self._nodes):
            self.rename_node(node, max_node + i)
        ordered_nodes = self.ordered_nodes()
        for i, node in enumerate(ordered_nodes):
            self.rename_node(node, i)

    def top(self, nodes):
        tops = [node for node in nodes
                if node in self._outputs
                   or any([node2 not in nodes for node2 in self._parents[node]])]
        return [node for node in self.ordered_nodes() if node in tops]

    def bottom(self, nodes):
        bottoms = [node for node in self._nodes
                   if node not in nodes
                    and any([node2 in nodes for node2 in self._parents[node]])]
        return [node for node in self.ordered_nodes() if node in bottoms]

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        assert self.output_connected()
        assert other.output_connected()
        s_visited = []
        o_visited = []
        if len(self._outputs) != len(other.outputs):
            return False
        for so, oo in zip(self._outputs, other.outputs):
            if not self.__compare_rec(other, so, oo, s_visited, o_visited):
                return False
        return True

    def __compare_rec(self, other, sn, on, s_visited, o_visited):
        if sn in s_visited:
            return on in o_visited
        if on in o_visited:
            return False
        s_visited.append(sn)
        o_visited.append(on)
        se = self._incoming_edge[sn]
        oe = other.incoming_edge(on)
        if se is None and oe is None:
            return True
        if se is None or oe is None:
            return False
        if se.label != oe.label:
            return False
        if len(se.inputs) != len(oe.inputs) or len(se.outputs) != len(oe.outputs):
            return False
        for sn2, on2 in zip(se.inputs, oe.inputs):
            if not self.__compare_rec(other, sn2, on2, s_visited, o_visited):
                return False
        return True

    def extract_dog(self, lhs, rhs):
        assert all([pairwise_disjoint_elem(list) for list in [lhs] + rhs])
        assert all([elem in lhs for list in rhs for elem in list])
        assert pairwise_disjoint(rhs)
        assert all([elem in self._nodes for elem in lhs])
        dog = DirectedOrderedGraph()

        top_lhs = self.top(lhs)
        bot_lhs = self.bottom(lhs)

        bot_rhs = [self.bottom(rhs_i) for rhs_i in rhs]
        top_rhs = [self.top(rhs_i) for rhs_i in rhs]

        # lhs
        for node in top_lhs:
            if node not in dog._nodes:
                dog.add_node(node)
            dog.add_to_outputs(node)
        for node in bot_lhs:
            if node not in dog._nodes:
                dog.add_node(node)
            dog.add_to_inputs(node)

        # rhs
        for i in range(len(rhs)):
            for node in bot_rhs[i] + top_rhs[i]:
                if node not in dog._nodes:
                    dog.add_node(node)
            dog.add_nonterminal_edge(bot_rhs[i], top_rhs[i])

        # fill recursively
        visited = []
        for node in top_lhs:
            self.__fill_rec(node, dog, visited, lhs, top_rhs, bot_rhs)
        return dog

    def __fill_rec(self, node, dog, visited, lhs, top_rhs, bot_rhs):
        if node not in lhs or node in visited:
            return
        visited.append(node)
        for i, tops in enumerate(top_rhs):
            if node in tops:
                for node2 in bot_rhs[i]:
                    self.__fill_rec(node2, dog, visited, lhs, top_rhs, bot_rhs)
                return
        edge = self.incoming_edge(node)
        assert edge is not None
        for node in edge.inputs:
            if node not in dog._nodes:
                dog.add_node(node)
        dog.add_edge(deepcopy(edge))
        for node2 in edge.inputs:
            self.__fill_rec(node2, dog, visited, lhs, top_rhs, bot_rhs)

    def primary_is_tree(self):
        outgoing = {}
        for edge in self._terminal_edges + self._nonterminal_edges:
            if edge is None:
                continue
            if len(edge.inputs) > 0 and len(edge.primary_inputs) == 0:
                return False
            for i in edge.primary_inputs:
                node = edge.inputs[i]
                if node in outgoing:
                    return False
                outgoing[node] = (edge, i)
        return True


class DeepSyntaxGraph:
    def __init__(self, sentence, dog, synchronization, label=None):
        self.__dog = dog
        self.__sentence = sentence
        self.__label = label
        self.__synchronization = synchronization

    def get_graph_position(self, sentence_position):
        return self.__synchronization[sentence_position]

    @property
    def label(self):
        return self.__label

    @property
    def dog(self):
        return self.__dog

    @property
    def sentence(self):
        return self.__sentence

    def extract_recursive_partitioning(self):
        assert self.dog.primary_is_tree()
        assert len(self.dog.outputs) == 1
        return self.__extract_recursive_partitioning_rec(self.dog.outputs[0])

    def __extract_recursive_partitioning_rec(self, node):
        covered = [sent_pos for sent_pos in range(len(self.sentence))
                   if node in self.get_graph_position(sent_pos)]
        edge = self.dog.incoming_edge(node)
        if edge is None:
            return (covered, [])
        children = []
        for i in edge.primary_inputs:
            child_node = edge.inputs[i]
            child_rec_par = self.__extract_recursive_partitioning_rec(child_node)
            assert child_rec_par != ([], [])
            children += [child_rec_par]
            for sent_pos in child_rec_par[0]:
                assert sent_pos not in covered
                covered.append(sent_pos)
        covered.sort()
        if len(children) == 1 and covered == children[0][0]:
            return children[0]
        else:
            return covered, children


def pairwise_disjoint_elem(list):
    for i, elem in enumerate(list):
        if elem in list[i+1:]:
            return False
    return True


def pairwise_disjoint(lists):
    for i, l1 in enumerate(lists):
        for l2 in lists[i+1:]:
            for elem in l1:
                if elem in l2:
                    return False
    return True