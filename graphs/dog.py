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
        self._functions = []
        for i, elem in enumerate(inputs):
            if isinstance(elem, tuple):
                if elem[1] == 'p':
                    self._primary_inputs.append(i)
                self._inputs.append(elem[0])
            else:
                self._inputs.append(elem)
            self._functions.append("--")

    @property
    def label(self):
        return self._label

    @property
    def terminal(self):
        return self._label is not None

    def set_primary(self, i):
        assert i < len(self.inputs)
        self._primary_inputs.append(i)
        return self

    def set_function(self, i, gr_function):
        assert i < len(self.inputs)
        self._functions[i] = gr_function
        return self

    def get_function(self, i):
        assert i < len(self.inputs)
        return self._functions[i]

    @property
    def primary_inputs(self):
        return self._primary_inputs

    def __str__(self):
        try:
            s = "[" + ", ".join(map(self.__function_strings, enumerate(self.inputs))) + "] -" \
                + (str(self.label) if self.label is not None else "") \
                + "-> [" + ", ".join(map(str, self.outputs)) + "]"
        except UnicodeEncodeError:
            # TODO: proper handling of encoding problems
            s = "foo"
        return s

    def __function_strings(self, pair):
        i, node = pair
        if self._functions[i] == '--':
            return str(node)
        else:
            return str(self._functions[i]) + ':' + str(node)

    def compare_labels(self, other):
        return all(map(lambda x, y: x == y, [self.label] + self._functions, [other.label] + other._functions))


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

    @property
    def nodes(self):
        return self._nodes

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

    def __str__(self):
        return "G[inputs=[" + ", ".join(map(str, self._inputs)) + "], " \
               + "outputs=[" + ", ".join(map(str, self._outputs)) + "], " \
               + "nont_edges=[" + ", ".join(map(str, self._nonterminal_edges)) + "], " \
               + "term_edges=[" + ", ".join(map(str, self._terminal_edges)) + "]] "

    def node_closure(self, function, reflexive=False):
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
        downward_closure = self.node_closure(lambda x: self.children(x))

        for node in self._nodes:
            if node in downward_closure[node]:
                return True

        return False

    def output_connected(self):
        upward_closure = self.node_closure(lambda n: self._parents[n], reflexive=True)

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
        return edge

    def add_terminal_edge(self, inputs, label, output):
        return self.add_edge(Edge([output], deepcopy(inputs), label))

    def add_nonterminal_edge(self, inputs, outputs):
        return self.add_edge(Edge(deepcopy(outputs), deepcopy(inputs)))

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

    def rename_node(self, node, node_new, trace=None):
        if node == node_new:
            return
        assert node_new not in self._nodes

        if trace is not None:
            trace[node] = node_new

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

        dog_node_renaming = {}
        max_node = max(self._nodes + dog._nodes)
        for j, node in enumerate(dog._nodes):
            dog.rename_node(node, max_node + 1 + j, dog_node_renaming)

        dog_node_renaming2 = {}
        for host_node, replace_node in zip(nt_edge.inputs, dog._inputs):
            dog.rename_node(replace_node, host_node, dog_node_renaming2)
        for host_node, replace_node in zip(nt_edge.outputs, dog._outputs):
            dog.rename_node(replace_node, host_node, dog_node_renaming2)

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
        return self.compose_node_renaming(dog_node_renaming, dog_node_renaming2)

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

    def compose_node_renaming(self, renaming1, renaming2):
        renaming = {}
        for node in renaming1:
            if renaming1[node] in renaming2:
                renaming[node] = renaming2[renaming1[node]]
            else:
                renaming[node] = renaming1[node]
        for node2 in renaming2:
            if node2 not in renaming:
                renaming[node2] = renaming2[node2]
        return renaming

    def compress_node_names(self):
        if self.ordered_nodes() == [i for i in range(len(self._nodes))]:
            return {}
        max_node = max(self._nodes) + 1

        renaming1 = {}
        for i, node in enumerate(self._nodes):
            self.rename_node(node, max_node + i, renaming1)
        ordered_nodes = self.ordered_nodes()
        renaming2 = {}
        for i, node in enumerate(ordered_nodes):
            self.rename_node(node, i, renaming2)
        return self.compose_node_renaming(renaming1, renaming2)

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

    def missing_children(self, nodes):
        relevant_parents = {node: [(parent, self.children(parent).index(node))
                                   for parent in self._parents[node] if parent in nodes]
                            for node in self._nodes
                            if node not in nodes
                            and any([node2 in nodes for node2 in self._parents[node]])}
        return [relevant_parents[node] for node in self.ordered_nodes() if node in relevant_parents]

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        assert self.output_connected()
        assert other.output_connected()
        morphism = {}
        inverse_morphism = {}
        if len(self._outputs) != len(other.outputs):
            return False
        for so, oo in zip(self._outputs, other.outputs):
            if not self.__compare_rec(other, so, oo, morphism, inverse_morphism):
                return False
        return True

    def __compare_rec(self, other, sn, on, morphism, inverse_morphism):
        if sn in morphism:
            return on in inverse_morphism and morphism[sn] == on and inverse_morphism[on] == sn
        if on in inverse_morphism:
            return False
        morphism[sn] = on
        inverse_morphism[on] = sn
        se = self._incoming_edge[sn]
        oe = other.incoming_edge(on)
        if se is None and oe is None:
            return True
        if se is None or oe is None:
            return False
        if not se.compare_labels(oe):
            return False
        if len(se.inputs) != len(oe.inputs) or len(se.outputs) != len(oe.outputs):
            return False
        for sn2, on2 in zip(se.inputs, oe.inputs):
            if not self.__compare_rec(other, sn2, on2, morphism, inverse_morphism):
                return False
        return True

    def compute_isomorphism(self, other):
        assert self.output_connected()
        assert other.output_connected()
        morphism = {}
        inverse_morphism = {}
        if len(self._outputs) != len(other.outputs):
            return None
        for so, oo in zip(self._outputs, other.outputs):
            if not self.__compare_rec(other, so, oo, morphism, inverse_morphism):
                return None
        return morphism, inverse_morphism

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

    def primary_is_tree(self, weak=False):
        """
        :param weak: if weak == True, then an "internal edge" of the DOG is allowed to be a leave of the tree
                     otherwise, each internal edge needs have at least one primary child
        :type weak: bool
        """
        outgoing = {}
        for edge in self._terminal_edges + self._nonterminal_edges:
            if edge is None:
                continue
            if (not weak) and len(edge.inputs) > 0 and len(edge.primary_inputs) == 0:
                return False
            for i in edge.primary_inputs:
                node = edge.inputs[i]
                if node in outgoing:
                    return False
                outgoing[node] = (edge, i)

        upward_closure = self.node_closure(lambda n: outgoing[n][0].outputs if n in outgoing else [], reflexive=True)

        for node in self._nodes:
            if not any([True for x in upward_closure[node] if x in self._outputs]):
                return False

        return True

    def internal_edges_without_primary_input(self):
        return [edge for edge in self._terminal_edges
                if len(edge.inputs) > 0 and len(edge.primary_inputs) == 0]

    def project_labels(self, proj):
        for edge in self._terminal_edges:
            edge.label = proj(edge.label)

    def export_dot(self, title):

        def node_line(node):
            s = "\t" + str(node) + " [shape=plaintext"
            inputs = ['i' + str(i) for i, n in enumerate(self._inputs) if n == node]
            outputs = ['o' + str(i) for i, n in enumerate(self._outputs) if n == node]
            if len(inputs) + len(outputs) > 0:
                label = str(node) + '[' + ','.join(inputs + outputs) + ']'
                s += ' , label=\"' + label + '\"'
            s += '];'
            return s

        def edge_line(edge, idx, label):
            return "\t" + idx + "[ shape=box, label=\"" + str(label) + "\"];"

        def tentacles(edge, idx):
            inputs = ["\t" + str(inp) + "->" + idx + "[label = \""
                      + (str(edge.get_function(i)) + ':' if edge.get_function(i) != '--' else "")
                      + str(i) + "\"];" for i, inp in enumerate(edge.inputs)]
            outputs = ["\t" + idx + "->" + str(out) for out in edge.outputs]
            return inputs + outputs

        node_lines = [node_line(node) for node in self._nodes]
        term_edge_lines = [line for i, edge in enumerate(self._terminal_edges) for line in
                           [edge_line(edge, 't' + str(i), edge.label)] + tentacles(edge, 't' + str(i))]
        nont_edge_lines = [line for i, edge in enumerate(self._nonterminal_edges) if edge is not None for line in
                           [edge_line(edge, 'n' + str(i), 'e' + str(i))] + tentacles(edge, 'n' + str(i))]
        return 'digraph G {\n\trankdir=BT;\n' \
               + '\tlabelloc=top;\n\tlabel=\"' + title + '\";\n' \
               + '\n'.join(node_lines + term_edge_lines + nont_edge_lines) \
               + '\n}'

    def export_graph_json(self, terminal_encoding, tentacle_labels=True, terminal_labeling=str):
        def label_edge(edge):
            label = str(terminal_labeling(edge.label))
            if tentacle_labels:
                label += '_' + '_'.join([edge.get_function(i) for i in range(len(edge.inputs))])
            return label

        data = {"type": "hypergraph"}
        data['nodes'] = [node for node in self._nodes]
        data['edges'] = []
        idx = 0
        for edge in self._nonterminal_edges:
            data['edges'].append({
                'id': idx
                , 'label': terminal_encoding.object_index(label_edge(edge))
                , 'attachment': edge.inputs + edge.outputs
                , 'terminal': False
            })
            idx += 1
        for edge in self._terminal_edges:
            data['edges'].append({
                'id': idx
                , 'label': terminal_encoding.object_index(label_edge(edge))
                , 'attachment': edge.inputs + edge.outputs
                , 'terminal': True
            })
            idx += 1
        data['ports'] = self._inputs + self._outputs
        return data

    def binarize(self, bin_modifier=lambda x: x + '-BAR', bin_func='--'):
        bin_dog = DirectedOrderedGraph()
        for node in self.nodes:
            bin_dog.add_node(node)
        for node in self._inputs:
            bin_dog.add_to_inputs(node)
        for node in self._outputs:
            bin_dog.add_to_outputs(node)
        for edge in self._nonterminal_edges:
            bin_dog.add_nonterminal_edge(edge.inputs, edge.outputs)
        next_node = max(self.nodes) + 1
        for edge in self._terminal_edges:
            if len(edge.inputs) <= 2:
                new_edge = bin_dog.add_terminal_edge(edge.inputs, edge.label, edge.outputs[0])
                for i, _ in enumerate(edge.inputs):
                    new_edge.set_function(i, edge.get_function(i))
                    if i in edge.primary_inputs:
                        new_edge.set_primary(i)
            else:
                new_nodes = []
                for i in range(len(edge.inputs) - 2):
                    new_nodes.append(next_node)
                    bin_dog.add_node(next_node)
                    next_node += 1
                new_nodes.append(edge.inputs[-1])
                right_functions = [bin_func] * (len(edge.inputs) - 2) + [edge.get_function(len(edge.inputs) - 1)]
                outputs = [edge.outputs[0]] + new_nodes
                for (i, left), right, right_function, output \
                        in zip(enumerate(edge.inputs), new_nodes, right_functions, outputs):
                    label = edge.label if i == 0 else bin_modifier(edge.label)
                    primary_l = 'p' if i in edge.primary_inputs else 's'
                    primary_r = 'p' if i < len(edge.inputs) - 2 or (i == len(edge.inputs) - 2 and (i + 1) in edge.primary_inputs) else 's'
                    bin_dog.add_terminal_edge([(left, primary_l), (right, primary_r)], label, output)\
                        .set_function(0, edge.get_function(i)).set_function(1, right_function)
        return bin_dog

    def debinarize(self, is_bin=lambda x: x.endswith("-BAR")):
        dog = DirectedOrderedGraph()
        nodes = []
        bin_nodes = []
        for node in self.nodes:
            if node in self._incoming_edge:
                incoming_edge = self.incoming_edge(node)
                if is_bin(incoming_edge.label):
                    bin_nodes.append(node)
                else:
                    nodes.append(node)
            else:
                nodes.append(node)
        for node in nodes:
            dog.add_node(node)
        for node in self._inputs:
            assert node not in bin_nodes
            dog.add_to_inputs(node)
        for node in self._outputs:
            assert node not in bin_nodes
            dog.add_to_outputs(node)
        if any([edge is not None for edge in self._nonterminal_edges]):
            for edge in self._nonterminal_edges:
                assert edge is not None
                assert not any([node in bin_nodes for node in edge.inputs])
                assert not any([node in bin_nodes for node in edge.outputs])
                dog.add_nonterminal_edge(edge.inputs, edge.outputs)

        closest_non_bin_node = {node: None for node in bin_nodes}
        left_of = {node: [] for node in bin_nodes}
        changed = True
        while changed:
            changed = False
            for node in bin_nodes:
                assert len(self._parents[node]) == 1
                parent = self._parents[node][0]
                if parent in nodes and closest_non_bin_node[node] != parent:
                    closest_non_bin_node[node] = parent
                    changed = True
                elif parent in bin_nodes and closest_non_bin_node[parent] != closest_non_bin_node[node]:
                    left_of[node] = left_of[parent] + [parent]
                    closest_non_bin_node[node] = closest_non_bin_node[parent]
                    changed = True
        conflation = {}
        for node in closest_non_bin_node:
            parent = closest_non_bin_node[node]
            assert parent is not None
            if parent in conflation:
                conflation[parent] += [node]
            else:
                conflation[parent] = [node]
        for parent in conflation:
            conflation[parent] = sorted(conflation[parent], key=lambda x: left_of[x])

        for edge in self.terminal_edges:
            if edge.outputs[0] not in bin_nodes:
                if not any([node in bin_nodes for node in edge.inputs]):
                    new_edge = dog.add_terminal_edge(edge.inputs, edge.label, edge.outputs[0])
                    for i, _ in enumerate(edge.inputs):
                        new_edge.set_function(i, edge.get_function(i))
                        if i in edge.primary_inputs:
                            new_edge.set_primary(i)
                else:
                    assert edge.inputs[0] not in bin_nodes
                    inputs = [(edge.inputs[0], 'p' if 0 in edge.primary_inputs else 's')]
                    functions = [edge.get_function(0)]

                    for node in conflation[edge.outputs[0]][:-1]:
                        bin_edge = self.incoming_edge(node)
                        inputs += [(bin_edge.inputs[0], 'p' if 0 in bin_edge.primary_inputs else 's')]
                        functions += [bin_edge.get_function(0)]

                    bin_edge = self.incoming_edge(conflation[edge.outputs[0]][-1])
                    inputs += [(node, 'p' if i in bin_edge.primary_inputs else 's')
                               for i, node in enumerate(bin_edge.inputs)]
                    functions += [bin_edge.get_function(0), bin_edge.get_function(1)]
                    new_edge = dog.add_terminal_edge(inputs, edge.label, edge.outputs[0])
                    for i, func in enumerate(functions):
                        new_edge.set_function(i, func)

        return dog


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

    @property
    def synchronization(self):
        return self.__synchronization

    def recursive_partitioning(self, subgrouping=False, weak=False):
        assert self.dog.primary_is_tree(weak=weak)
        assert len(self.dog.outputs) == 1
        return self.__extract_recursive_partitioning_rec(self.dog.outputs[0], subgrouping)

    def covered_sentence_positions(self, dog_positions):
        return [sent_pos for sent_pos in range(len(self.sentence))
                if any([dog_pos in self.get_graph_position(sent_pos)
                        for dog_pos in dog_positions])]

    def __extract_recursive_partitioning_rec(self, node, subgrouping):
        covered = self.covered_sentence_positions([node])
        edge = self.dog.incoming_edge(node)
        if edge is None:
            return set(covered), []
        children = []
        for i in edge.primary_inputs:
            child_node = edge.inputs[i]
            child_rec_par = self.__extract_recursive_partitioning_rec(child_node, subgrouping)
            assert child_rec_par != ([], [])
            children += [child_rec_par]
            for sent_pos in child_rec_par[0]:
                assert sent_pos not in covered
                covered.append(sent_pos)
        covered = set(covered)
        if len(children) == 1 and covered == children[0][0]:
            return children[0]
        elif subgrouping and len(children) > 2:
            children_new = {}
            for i, child_rec_par in enumerate(children):
                func = edge.get_function(i)
                if func in children_new:
                    children_new[func] += [child_rec_par]
                else:
                    children_new[func] = [child_rec_par]
            new_child_list = []
            for func in children_new:
                if len(children_new[func]) > 1:
                    new_child_list.append((set([sent_pos for child in children_new[func] for sent_pos in child[0]])
                                           , children_new[func]))
                else:
                    new_child_list += children_new[func]
            if len(new_child_list) == 1 and covered == new_child_list[0][0]:
                return new_child_list[0]
            else:
                return covered, new_child_list
        else:
            return covered, children

    def id_yield(self):
        return map(lambda x: self.get_graph_position(x), [i for i in range(len(self.sentence))])

    def export_bihypergraph_json(self, terminal_encoding, tentacle_labels=True, terminal_labeling=str):
        data = {"type": "bihypergraph"}
        data["G2"] = self.dog.export_graph_json(terminal_encoding, tentacle_labels, terminal_labeling=terminal_labeling)
        max_node = max(data["G2"]['nodes'])
        max_edge = max(map(lambda x: x['id'], data["G2"]['edges']))
        data["G1"] = self.string_to_graph_json(self.sentence, terminal_encoding, terminal_labeling=terminal_labeling,
                                               start_node=max_node + 1, start_edge=max_edge + 1)
        max_edge = max(map(lambda x: x['id'], data["G1"]['edges']))
        data["alignment"] = [{'id': idx + max_edge + 1
                                 , 'label': terminal_encoding.object_index(None)
                                 , 'attachment': [max_node + 1 + idx] + self.__synchronization[idx]
                              } for idx in range(len(self.__synchronization)) if self.__synchronization[idx] != []
                             ]
        return data

    @staticmethod
    def string_to_graph_json(string, terminal_encoding, terminal_labeling=id, start_node=0, start_edge=0):
        data = {'type': 'hypergraph'
            , 'nodes': [i for i in range(start_node, start_node + len(string) + 1)]
            , 'edges': [{'id': idx + start_edge
                            , 'label': terminal_encoding.object_index(terminal_labeling(symbol))
                            , 'attachment': [start_node + idx, start_node + idx + 1]
                            , 'terminal': True
                         } for idx, symbol in enumerate(string)]
            , 'ports': [start_node, start_node + len(string)]
                }
        return data

    def labeled_frames(self, replace_nodes_by_string_positions=True, guard=lambda x: True):
        frames = []

        descendants = self.dog.node_closure(self.dog.children)

        for node in self.dog.nodes:
            edge = self.dog.incoming_edge(node)

            if replace_nodes_by_string_positions:
                predicate = [i for i, sync in enumerate(self.synchronization) if node in sync]
                if predicate == []:
                    predicate = edge.label
                else:
                    predicate = tuple(sorted(predicate))
            else:
                predicate = edge.label

            arg_label_list = []
            for i, child in enumerate(edge.inputs):
                func = edge.get_function(i)

                if replace_nodes_by_string_positions:
                    arg = [i for i, sync in enumerate(self.synchronization) if child in sync]
                    if arg == []:
                        arg = [i for i, sync in enumerate(self.synchronization)
                               if any([desc in descendants[child] for desc in sync])]
                    arg = tuple(sorted(arg))
                else:
                    arg = self.dog.incoming_edge(child).label

                arg_label_list.append((arg, func))

            frame = predicate, tuple(arg_label_list)
            if guard(frame):
                frames.append(frame)

        return frames

    def binarize(self, bin_modifier=lambda x: x + "-BAR", bin_func="--"):
        bin_dog = self.dog.binarize(bin_modifier=bin_modifier, bin_func=bin_func)
        return DeepSyntaxGraph(self.sentence, bin_dog, self.synchronization, self.label)

    def debinarize(self, is_bin=lambda x: x.endswith("-BAR")):
        dog = self.dog.debinarize(is_bin=is_bin)
        assert all([all([node in dog.nodes for node in sync]) for sync in self.synchronization])
        return DeepSyntaxGraph(self.sentence, dog, self.synchronization)


def pairwise_disjoint_elem(list):
    for i, elem in enumerate(list):
        if elem in list[i + 1:]:
            return False
    return True


def pairwise_disjoint(lists):
    for i, l1 in enumerate(lists):
        for l2 in lists[i + 1:]:
            for elem in l1:
                if elem in l2:
                    return False
    return True
