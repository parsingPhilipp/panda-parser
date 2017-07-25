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
        self._inputs = inputs

    @property
    def label(self):
        return self._label

    @property
    def terminal(self):
        return self._label is not None


class DirectedOrderedGraph:
    def __init__(self):
        self._nodes = []
        self._inputs = []
        self._outputs = []
        self._terminal_edges = []
        self._nonterminal_edges = []
        self._parents = {}
        self._children = {}
        self._incoming_edge = {}

    @property
    def outputs(self):
        return self._outputs

    def incoming_edge(self, node):
        return self._incoming_edge[node]

    def type(self):
        _type = []
        for edge in self._nonterminal_edges:
            _type.append((len(edge.outputs), len(edge.inputs)))
        _type.append((len(self._inputs), len(self._outputs)))
        return _type

    def _node_closure(self, the_dict, reflexive=False):
        closure = {}

        if reflexive:
            for node in self._nodes:
                closure[node] = [node]
        else:
            for node in self._nodes:
                closure[node] = list(the_dict[node])

        changed = True
        while changed:
            changed = False
            for node in self._nodes:
                for _node in closure[node]:
                    for __node in the_dict[_node]:
                        if __node not in closure[node]:
                            changed = True
                            closure[node].append(__node)
        return closure

    def cyclic(self):
        downward_closure = self._node_closure(self._children)

        for node in self._nodes:
            if node in downward_closure[node]:
                return True

        return False

    def output_connected(self):
        upward_closure = self._node_closure(self._parents, reflexive=True)

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
        assert all([node not in self._children for node in edge.outputs])
        assert all([self._incoming_edge[output] is None for output in edge.outputs])
        for output in edge.outputs:
            self._incoming_edge[output] = edge
            self._children[output] = edge.inputs
            for node in edge.inputs:
                if output not in self._parents[node]:
                    self._parents[node].append(output)
        if edge.terminal:
            self._terminal_edges.append(edge)
        else:
            self._nonterminal_edges.append(edge)

    def add_terminal_edge(self, inputs, label, output):
        _inputs = list(inputs)
        self.add_edge(Edge([output], _inputs, label))

    def add_nonterminal_edge(self, inputs, outputs):
        _inputs = list(inputs)
        _outputs = list(outputs)
        self.add_edge(Edge(_outputs, _inputs))

    def add_to_inputs(self, node):
        assert node not in self._children
        self._inputs.append(node)
        self._children[node] = []

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

        if node in self._children:
            self._children[node_new] = self._children[node]
            del self._children[node]
            for key in self._children:
                self.__replace_inplace(self._children[key], node, node_new)

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
        print(len(dog._nonterminal_edges), all([edge is None for edge in dog._nonterminal_edges]))
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
            del self._children[node]
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
            for v_2 in self._children[v]:
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

    # def top(self, nodes):
    # def __top_rec(self, nodes):

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
