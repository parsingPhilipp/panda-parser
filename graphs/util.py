from __future__ import print_function
import subprocess
from collections import defaultdict
from decomposition import n_spans

def render_and_view_dog(dog, name, path="/tmp/"):
    dot = dog.export_dot(name)
    dot_path = path + name + '.dot'
    pdf_path = path + name + '.pdf'
    with open(dot_path, 'w') as dot_file:
        dot_file.write(dot)

    command = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
    p = subprocess.Popen(command)
    p.communicate()
    # print(command, p.returncode, p.stderr, p.stdout)

    q = subprocess.Popen(["zathura", pdf_path])
    return q


def extract_recursive_partitioning(dsg, alpha=0.9, beta=0.2):
    assert not dsg.dog.cyclic()
    VROOT = max(dsg.dog.nodes) + 1

    # add all natural roots, i.e., nodes without predecessors but children or outputs of the dog

    natural_roots = [node for node in dsg.dog.nodes
                     if (node not in dsg.dog.parents or dsg.dog.parents[node] == [])
                     and (node in dsg.dog.outputs or dsg.dog.children(node))]


    tree_map = defaultdict(list)
    distance = {VROOT: 1}
    queue = []
    for root in natural_roots:
        if root not in distance and (root not in dsg.dog.parents or dsg.dog.parents[root] == []):
            tree_map[VROOT].append(root)
            distance[root] = distance[VROOT] + 1
            queue.append(root)

    def add_unambiguous_children(node):
        for child in dsg.dog.children(node):
            if child not in distance and len(dsg.dog.parents[child]) == 1:
                tree_map[node].append(child)
                distance[child] = distance[node] + 1
                queue.append(child)

    while queue:
        first = queue[0]
        queue = queue[1:]
        add_unambiguous_children(first)

    def unambiguous_children(node, visited=set()):
        children = set()
        if node in visited:
            return children
        visited.add(node)
        for child in dsg.dog.children(node):
            if len(dsg.dog.parents[child]) == 1:
                children.add(child)
                children = children.union(unambiguous_children(child, visited))
        return children

    def ambiguous_children(node):
        def ambiguous_children_rec(node, children):
            for child in dsg.dog.children(node):
                if child not in children:
                    children.add(child)
                    children = children.union(ambiguous_children_rec(child))
        return ambiguous_children_rec(node, set())

    def current_children(node):
        children = set()
        for child in tree_map[node]:
            children.add(node)
            children = children.union(current_children(child))
        return children

    # add all remaining outputs to either VROOT or in another position
    queue2 = [root for root in dsg.dog.outputs if root not in distance]
    while queue2:
        head = queue2[0]
        queue2 = queue2[1:]
        if head in distance:
            continue
        best_parent = VROOT
        penalty = n_spans(dsg.covered_sentence_positions(unambiguous_children(head))) * beta \
                  + (1.0 - beta)
        for parent in dsg.dog.parents[head]:
            if parent not in distance:
                continue
            penalty_2 = n_spans(unambiguous_children(head).union(current_children(parent))) * beta \
                        + (1.0 / distance[parent] * (1-beta))
            if penalty_2 < penalty:
                best_parent = parent
                penalty = penalty_2

        tree_map[best_parent].append(head)
        distance[head] = distance[best_parent] + 1
        add_unambiguous_children(head)

        while queue:
            first = queue[0]
            queue = queue[1:]
            add_unambiguous_children(first)

    # resolve all ambiguous attachments
    queue2 = [child for node in distance if node != VROOT
             for child in dsg.dog.children(node) if child not in distance]
    while queue2:
        while queue2:
            head = queue2[0]
            queue2 = queue2[1:]
            if head in distance:
                continue
            parent = sorted([parent for parent in dsg.dog.parents if parent in distance],
                            key=lambda x: distance[x] * alpha + n_spans(current_children(x).union(unambiguous_children(head))))[0]
            distance[head] = distance[parent] + 1
            tree_map[parent].append(head)
            add_unambiguous_children(head)
            while queue:
                first = queue[0]
                queue = queue[1:]
                add_unambiguous_children(first)
        queue2 = [child for node in distance if node != VROOT
                      for child in dsg.dog.children(node) if child not in distance]

    # locally attach all remaining nodes
    for node in dsg.dog.nodes:
        if node not in distance:
            assert dsg.dog.parents[node] == []
            parent = sorted([p for p in distance], key=lambda x: n_spans(dsg.covered_sentence_positions({node}.union({y for y in [x] if y != VROOT}).union(current_children(x)))) * alpha + (1-alpha) * (1.0) / distance[x])[0]
            # parents = sorted([p for p in distance if p != VROOT], key=lambda x: n_spans(dsg.covered_sentence_positions({node, x}.union(current_children(x)))) * alpha + (1-alpha) * (1.0) / distance[x])
            # print(node, [(x, dsg.covered_sentence_positions({node, x}.union(current_children(x))), n_spans(dsg.covered_sentence_positions({node, x}.union(current_children(x)))),  (1.0) / distance[x]) for x in parents])

            distance[node] = distance[parent] + 1
            tree_map[parent].append(node)

    def tree_map_to_recursive_partitioning_rec(node):
        covered = set()
        children = []
        for child in tree_map[node]:
            child_rec_par = tree_map_to_recursive_partitioning_rec(child)
            children.append(child_rec_par)
            if not covered.isdisjoint(child_rec_par[0]):
                print(tree_map)
                assert covered.isdisjoint(child_rec_par[0])
            covered = covered.union(child_rec_par[0])
        if node != VROOT:
            node_covered = set(dsg.covered_sentence_positions([node]))
            if node_covered:
                assert covered.isdisjoint(node_covered)
                covered = covered.union(node_covered)
                children.append((node_covered, []))
        if len(children) > 1:
            return covered, children
        else:
            return children[0]

    # print(tree_map)

    return tree_map_to_recursive_partitioning_rec(VROOT)


def pretty_print_rec_partitioning(rec_par):
    def print_rec(rec_par, indent):
        print('\t' * indent, rec_par[0])
        for child in rec_par[1]:
            print_rec(child, indent+1)

    print_rec(rec_par, 0)
