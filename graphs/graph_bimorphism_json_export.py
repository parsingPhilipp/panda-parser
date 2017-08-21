from grammar.lcfrs import LCFRS_rule, LCFRS_var
def export_dog_grammar_to_json(grammar, terminals):
    data = {}
    data["rules"] = []
    for rule in grammar.rules():
        rule_obj = {"idx": rule.get_idx(), "lhs_nont": rule.lhs().nont(), "rhs_nonts": rule.rhs(),
                    "weight": rule.weight(), "dog": rule.dcp()[0].export_graph_json(terminals)}
        next_node = max(rule.dcp()[0].nodes) + 1
        next_edge = max(map(lambda x: x['id'], rule_obj['dog']['edges'])) + 1
        rule_obj["lcfrs"], terminal_nodes = convert_lcfrs_part(rule, terminals, next_node=next_node, first_edge=next_edge)
        rule_obj["alignment"] = []
        idx = max(map(lambda x: x['id'], rule_obj['lcfrs']['edges'])) + 1
        for sent_node, sync in zip(terminal_nodes, rule.dcp()[1]):
            if len(sync) > 0:
                edge = {'attachment': [sent_node] + sync, 'label': terminals.object_index(None), 'id': idx}
                idx += 1
                rule_obj["alignment"].append(edge)
        data["rules"].append(rule_obj)
    data["start"] = grammar.start()
    data["alignmentLabel"] = terminals.object_index(None)
    data["nonterminalEdgeLabel"] = terminals.object_index(None)
    return data


def convert_lcfrs_part(rule, terminals, next_node=0, first_edge=0):
    assert isinstance(rule, LCFRS_rule)
    nodes = []
    ports = []
    edges = []
    rhs_nont_connections = {}
    for idx, _ in enumerate(rule.rhs()):
        rhs_nont_connections[idx] = []
    term_edges = first_edge
    terminal_nodes = []
    for arg in rule.lhs().args():
        current_node = next_node
        next_node += 1
        nodes.append(current_node)
        ports.append(current_node)
        for elem in arg:
            if isinstance(elem, LCFRS_var):
                while len(rhs_nont_connections[elem.mem]) < elem.arg * 2 + 2:
                    rhs_nont_connections[elem.mem].append(None)
                rhs_nont_connections[elem.mem][elem.arg * 2] = current_node
                rhs_nont_connections[elem.mem][elem.arg * 2 + 1] = next_node
            else:
                edges.append({ "terminal": True
                          , "id": term_edges + len(rule.rhs())
                          , "attachment": [current_node, next_node]
                          , "label": terminals.object_index(elem)})
                terminal_nodes.append(current_node)

            nodes.append(next_node)
            current_node = next_node
            next_node += 1
        ports.append(current_node)

    for idx, _ in enumerate(rule.rhs()):
        assert(all([node is not None for node in rhs_nont_connections[idx]]))
        edges.append({ "terminal": False
                     , "id": idx + first_edge
                     , "attachment": rhs_nont_connections[idx]
                     , "label": terminals.object_index(None)})

    return {"nodes": nodes, "ports": ports, "edges": edges}, terminal_nodes


def export_corpus_to_json(corpus, terminals, terminal_labeling=str):
    data = {  "corpus": []
            , "alignmentLabel": terminals.object_index(None)
            , "nonterminalEdgeLabel": terminals.object_index(None)
            }
    for dsg in corpus:
        data["corpus"].append(dsg.export_bihypergraph_json(terminals, terminal_labeling=terminal_labeling))
    return data