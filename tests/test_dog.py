from __future__ import print_function
import unittest
from graphs.dog import *
from graphs.util import render_and_view_dog
from graphs.graph_bimorphism_json_export import export_dog_grammar_to_json, export_corpus_to_json
from graphs.graph_decomposition import *
from corpora.tiger_parse import sentence_names_to_deep_syntax_graphs
from corpora.negra_parse import acyclic_syntax_graph_to_sentence_name, acyclic_graphs_to_sentence_names
from hybridtree.monadic_tokens import ConstituentTerminal, ConstituentCategory
from grammar.lcfrs_derivation import LCFRSDerivationWrapper
from parser.naive.parsing import LCFRS_parser
from parser.cpp_cfg_parser.parser_wrapper import CFGParser
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory, fanout_limited_partitioning, \
    fanout_limited_partitioning_left_to_right
from grammar.induction.terminal_labeling import PosTerminals
import subprocess
import json
from util.enumerator import Enumerator
import shutil
import os
import sys
from graphs.schick_parser_rtg_import import read_rtg
from parser.supervised_trainer.trainer import PyDerivationManager
from graphs.parse_accuracy import PredicateArgumentScoring
from graphs.dog_generator import generate
from random import randint
from copy import deepcopy

# from setup import schick_executable
SCHICK_PARSER_JAR = 'HypergraphReduct-1.0-SNAPSHOT.jar'


class GraphTests(unittest.TestCase):
    def test_acyclic_dog(self):
        for dog in [build_acyclic_dog(), dog_s1(), dog_s2(), dog_s3(), dog_s11(), dog_s12(), dog_s13(), dog_s131()]:
            self.assertFalse(dog.cyclic())
            self.assertTrue(dog.output_connected())

    def test_cyclic_dog(self):
        dog = dog_se()
        self.assertTrue(dog.cyclic())
        self.assertTrue(dog.output_connected())

    def test_substitution(self):
        dog_13 = dog_s13()
        dog_13.replace_by(0, dog_s131())
        dog_13.replace_by(1, dog_s132())
        dog_13.replace_by(2, dog_s133())

        dog_1 = dog_s1()
        dog_1.replace_by(0, dog_s11())
        dog_1.replace_by(1, dog_s12())

        dog_3 = dog_s3()
        dog_3.replace_by(0, dog_s32())
        dog_3.replace_by(1, dog_13)

        dog_host = dog_se()
        dog_host.replace_by(0, dog_1)
        dog_host.replace_by(1, dog_s2())
        dog_host.replace_by(2, dog_3)

        self.assertEqual(build_acyclic_dog(), dog_host)

    def test_eq(self):
        self.assertEqual(build_acyclic_dog(), build_acyclic_dog())
        self.assertEqual(dog_s132(), dog_s132())
        self.assertNotEqual(build_acyclic_dog(), build_acyclic_dog_permuted())
        self.assertNotEqual(dog_s1(), dog_s3())
        self.assertEqual(dog_s1() == dog_s13(), dog_s13() == dog_s1())
        self.assertNotEqual(dog_s13(), dog_s2())
        self.assertNotEqual(dog_s1(), dog_s2())
        self.assertNotEqual(dog_s131(), dog_s132())

    def test_order(self):
        self.assertListEqual(build_acyclic_dog().ordered_nodes(), [0, 1, 4, 5, 6, 8, 9, 10, 2, 3, 7])
        self.assertListEqual(build_acyclic_dog_permuted().ordered_nodes(), [0, 1, 4, 5, 6, 8, 9, 10, 2, 3, 7])
        self.assertListEqual(dog_se().ordered_nodes(), [0, 1, 5, 4, 2, 3])
        dog = dog_se()
        dog.compress_node_names()
        self.assertListEqual(dog.ordered_nodes(), [i for i in range(6)])

    def test_top_bottom(self):
        dog = build_acyclic_dog()

        nodes = [1, 4, 5]
        self.assertListEqual(dog.top(nodes), [1, 4])
        self.assertListEqual(dog.bottom(nodes), [6])

        nodes = [3, 7, 6, 8, 9, 10]
        self.assertListEqual(dog.top(nodes), [6, 3])
        self.assertListEqual(dog.bottom(nodes), [4])

        nodes = [i for i in range(11)]
        self.assertListEqual(dog.top(nodes), [0])
        self.assertListEqual(dog.bottom(nodes), [])

        nodes = [2, 4, 5, 7, 8, 9, 10]
        for node in nodes:
            self.assertListEqual(dog.top([node]), [node])
            self.assertListEqual(dog.bottom([node]), [])

    def test_extraction(self):
        dog = build_acyclic_dog()

        self.assertEqual(dog_se(), dog.extract_dog([i for i in range(11)], [[1, 4, 5], [2], [3, 6, 7, 8, 9, 10]]))

        for (lab, i) in [
            ('und', 2), ('Sie', 4), ('entwickelt', 5), ('druckt', 7),
            ('Verpackungen', 8), ('und', 9), ('Etiketten', 10)
        ]:
            self.assertEqual(build_terminal_dog(lab), dog.extract_dog([i], []))

        self.assertEqual(dog_s1(), dog.extract_dog([1, 4, 5], [[4], [5]]))
        self.assertEqual(dog_s3(), dog.extract_dog([3, 7, 6, 8, 9, 10], [[7], [6, 8, 9, 10]]))

    def test_copy_and_substitution_order_invariance(self):
        dog = dog_s13()
        dog.replace_by(0, dog_s131())
        dog_2 = deepcopy(dog)
        self.assertEqual(dog, dog_2)
        dog.replace_by(1, dog_s132())
        self.assertNotEqual(dog, dog_2)
        dog_2.replace_by(2, dog_s133())
        dog_2.replace_by(1, dog_s132())
        self.assertNotEqual(dog, dog_2)
        dog.replace_by(2, dog_s133())
        self.assertEqual(dog, dog_2)

    def test_primary(self):
        dog = build_acyclic_dog()
        self.assertTrue(dog.primary_is_tree())
        dog2 = build_acyclic_dog_permuted()
        self.assertFalse(dog2.primary_is_tree())

    def test_upward_closure(self):
        dog = build_acyclic_dog()
        for (lab, i) in [
            ('und', 2), ('Sie', 4), ('entwickelt', 5), ('druckt', 7),
            ('Verpackungen', 8), ('und', 9), ('Etiketten', 10)
        ]:
            self.assertListEqual(upward_closure(dog, [i]), [i])

        self.assertSetEqual(set(upward_closure(dog, [4, 5])), {1, 4, 5})
        self.assertSetEqual(set(upward_closure(dog, [8, 9, 10])), {6, 8, 9, 10})
        self.assertSetEqual(set(upward_closure(dog, [4, 5, 2, 7, 8, 9, 10])), set([i for i in range(11)]))

    def test_dsg(self):
        dsg = build_dsg()
        rec_part = dsg.recursive_partitioning()
        self.assertEqual(rec_part, ({0, 1, 2, 3, 4, 5, 6}, [({0, 1}, [({0}, []), ({1}, [])]), ({2}, []), (
            {3, 4, 5, 6}, [({3}, []), ({4, 5, 6}, [({4}, []), ({5}, []), ({6}, [])])])]))
        dcmp = compute_decomposition(dsg, rec_part)
        self.assertEqual(dcmp, ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [([1, 4, 5], [([4], []), ([5], [])]), ([2], []), (
            [3, 6, 7, 8, 9, 10], [([7], []), ([6, 8, 9, 10], [([8], []), ([9], []), ([10], [])])])]))
        self.__structurally_equal(rec_part, dcmp)

        grammar = induce_grammar_from(dsg, rec_part, dcmp, terminal_labeling=str)
        # print(grammar)

        for nont, label in zip(["[4]", "[5]", "[2]", "[7]", "[8]", "[9]", "[10]"],
                               ["Sie", "entwickelt", "und", "druckt", "Verpackungen", "und", "Etiketten"]):
            for rule in grammar.lhs_nont_to_rules(nont):
                self.assertEqual(rule.dcp()[0], build_terminal_dog(label))

        for nont, graph in zip(
                ["[1, 4, 5]", "[6, 8, 9, 10]", "[3, 6, 7, 8, 9, 10]", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"],
                [dog_s1(), dog_s13(), dog_s3(), dog_se()]):
            for rule in grammar.lhs_nont_to_rules(nont):
                self.assertEqual(rule.dcp()[0], graph)

        parser = LCFRS_parser(grammar)
        parser.set_input(dsg.sentence)  # ["Sie", "entwickelt", "und", "druckt", "Verpackungen", "und", "Etiketten"]
        parser.parse()
        self.assertTrue(parser.recognized())

        derivation = parser.best_derivation_tree()
        self.assertNotEqual(derivation, None)

        dog, sync_list = dog_evaluation(derivation)
        self.assertEqual(dog, dsg.dog)

        self.assertEqual(len(sync_list), len(dsg.sentence))
        # print(dog)
        # print(sync)
        # print(sync_list)

        morphism, _ = dsg.dog.compute_isomorphism(dog)

        for i in range(len(dsg.sentence)):
            self.assertListEqual(list(map(lambda x: morphism[x], dsg.get_graph_position(i))), sync_list[i])

    def test_induction_with_labeling_strategies(self):
        dsg = build_dsg()
        rec_part_strategy = the_recursive_partitioning_factory().get_partitioning('right-branching')[0]
        rec_part = rec_part_strategy(dsg)
        dcmp = compute_decomposition(dsg, rec_part)

        grammar = induce_grammar_from(dsg, rec_part, dcmp, labeling=simple_labeling, terminal_labeling=str)
        print(grammar)

        parser = LCFRS_parser(grammar)
        parser.set_input(dsg.sentence)  # ["Sie", "entwickelt", "und", "druckt", "Verpackungen", "und", "Etiketten"]
        parser.parse()
        self.assertTrue(parser.recognized())

        derivation = parser.best_derivation_tree()
        self.assertNotEqual(derivation, None)

        dog, sync_list = dog_evaluation(derivation)
        self.assertEqual(dog, dsg.dog)

        self.assertEqual(len(sync_list), len(dsg.sentence))
        # print(dog)
        # print(sync)
        # print(sync_list)

        morphism, _ = dsg.dog.compute_isomorphism(dog)

        for i in range(len(dsg.sentence)):
            self.assertListEqual(list(map(lambda x: morphism[x], dsg.get_graph_position(i))), sync_list[i])

    def __structurally_equal(self, rec_part, decomp):
        self.assertEqual(len(rec_part[1]), len(decomp[1]))
        for rec_part_child, decomp_child in zip(rec_part[1], decomp[1]):
            self.__structurally_equal(rec_part_child, decomp_child)

    def test_tiger_parse_to_dsg(self):
        dsg = sentence_names_to_deep_syntax_graphs(["s26954"], "res/tiger/tiger_s26954.xml", hold=False)[0]

        f = lambda token: token.form() if isinstance(token, ConstituentTerminal) else token
        dsg.dog.project_labels(f)

        print(map(str, dsg.sentence))
        print(dsg.label)
        print(dsg.dog)
        print([dsg.get_graph_position(i) for i in range(len(dsg.sentence))])

        # strip "VROOT"
        sub_dog = dsg.dog.extract_dog([i for i in range(11)], [])

        self.assertEqual(sub_dog, build_acyclic_dog())

    def test_tiger_to_export_conversion(self):
        for s in ["s26954", "s22084"]:
            dsg = sentence_names_to_deep_syntax_graphs([s], "res/tiger/tiger_%s.xml" % s, hold=False, ignore_puntcuation=False)[0]
            lines = acyclic_syntax_graph_to_sentence_name(dsg)
            print(''.join(lines))

    def test_corpus_conversion(self):
        dsgs = sentence_names_to_deep_syntax_graphs(["s" + str(i) for i in range(1, 50474 + 1)],
                                                    "res/tiger/tiger_release_aug07.corrected.16012013.utf8.xml")
        lines = acyclic_graphs_to_sentence_names(dsgs, 1, 500)
        with open('/tmp/tiger_full_with_sec_edges.export', 'w') as fd:
            fd.write(''.join(lines))

    def test_induction_from_corpus_tree(self):
        dsg = sentence_names_to_deep_syntax_graphs(["s26954"], "res/tiger/tiger_s26954.xml", hold=False)[0]

        def label_edge(edge):
            if isinstance(edge.label, ConstituentTerminal):
                return edge.label.pos()
            else:
                return edge.label
        labeling = lambda nodes, dsg: simple_labeling(nodes, dsg, label_edge)

        term_labeling_token = PosTerminals()

        def term_labeling(token):
            if isinstance(token, ConstituentTerminal):
                return term_labeling_token.token_label(token)
            else:
                return token

        rec_part_strategy = the_recursive_partitioning_factory().get_partitioning('cfg')[0]
        rec_part = rec_part_strategy(dsg)
        dcmp = compute_decomposition(dsg, rec_part)

        grammar = induce_grammar_from(dsg, rec_part, dcmp, labeling=labeling, terminal_labeling=term_labeling)

        print(grammar)

        parser = LCFRS_parser(grammar)
        parser.set_input(term_labeling_token.prepare_parser_input(dsg.sentence))
        parser.parse()
        self.assertTrue(parser.recognized())

        derivation = parser.best_derivation_tree()
        self.assertNotEqual(derivation, None)

    def test_induction_on_a_corpus(self):
        interactive = False
        start = 1
        stop = 50
        path = "res/tiger/tiger_release_aug07.corrected.16012013.utf8.xml"
        # path = "res/tiger/tiger_8000.xml"
        exclude = []
        dsgs = sentence_names_to_deep_syntax_graphs(
            ['s' + str(i) for i in range(start, stop + 1) if i not in exclude]
            , path
            , hold=False)

        rec_part_strategy = the_recursive_partitioning_factory().get_partitioning('cfg')[0]

        def label_edge(edge):
            if isinstance(edge.label, ConstituentTerminal):
                return edge.label.pos()
            else:
                return edge.label
        nonterminal_labeling = lambda nodes, dsg: simple_labeling(nodes, dsg, label_edge)

        term_labeling_token = PosTerminals()
        def term_labeling(token):
            if isinstance(token, ConstituentTerminal):
                return term_labeling_token.token_label(token)
            else:
                return token

        grammar = induction_on_a_corpus(dsgs, rec_part_strategy, nonterminal_labeling, term_labeling, normalize=True)
        grammar.make_proper()

        parser = CFGParser(grammar)

        scorer = PredicateArgumentScoring()

        for dsg in dsgs:
            parser.set_input(term_labeling_token.prepare_parser_input(dsg.sentence))
            parser.parse()
            self.assertTrue(parser.recognized())
            derivation = parser.best_derivation_tree()
            dog, sync = dog_evaluation(derivation)
            dsg2 = DeepSyntaxGraph(dsg.sentence, dog, sync)

            f = lambda token: token.pos() if isinstance(token, ConstituentTerminal) else token
            dsg.dog.project_labels(f)
            parser.clear()

            scorer.add_accuracy_frames(
                dsg.labeled_frames(guard=lambda x: len(x[1]) > 0),
                dsg2.labeled_frames(guard=lambda x: len(x[1]) > 0)
            )

            # print('dsg: ', dsg.dog, '\n', [dsg.get_graph_position(i) for i in range(len(dsg.sentence))],
            # '\n\n parsed: ', dsg2.dog, '\n', [dsg2.get_graph_position(i+1) for i in range(len(dsg2.sentence))])
            # print()
            if interactive:
                if dsg.label == 's50':
                    pass
                if dsg.dog != dog:
                    z1 = render_and_view_dog(dsg.dog, "corpus_" + dsg.label)
                    z2 = render_and_view_dog(dog, "parsed_" + dsg.label)
                    z1.communicate()
                    z2.communicate()

        print("Labeled frames:")
        print("P", scorer.labeled_frame_scorer.precision(), "R", scorer.labeled_frame_scorer.recall(),
              "F1", scorer.labeled_frame_scorer.fmeasure())
        print("Labeled dependencies:")
        print("P", scorer.labeled_dependency_scorer.precision(), "R", scorer.labeled_dependency_scorer.recall(),
              "F1", scorer.labeled_dependency_scorer.fmeasure())

    def test_dot_export(self):
        dsg = sentence_names_to_deep_syntax_graphs(["s26954"], "res/tiger/tiger_s26954.xml", hold=False)[0]

        f = lambda token: token.form() if isinstance(token, ConstituentTerminal) else token
        dsg.dog.project_labels(f)

        dot = dsg.dog.export_dot("s26954")
        print(dot)

        render_and_view_dog(dsg.dog, "foo")

    def test_json_export(self):
        dog = build_acyclic_dog()
        terminals = Enumerator()
        data = dog.export_graph_json(terminals)
        with open('/tmp/json_graph_1.json', 'w') as file:
            json.dump(data, file)

        dsg = build_dsg()
        data = dsg.export_bihypergraph_json(terminals)
        with open('/tmp/json_bigraph_1.json', 'w') as file:
            json.dump(data, file)

        rule_dog = dog_se()
        data2 = rule_dog.export_graph_json(terminals)
        with open('/tmp/json_nonterminal_graph_1.json', 'w') as file:
            json.dump(data2, file)

        terminals.print_index()

    def test_json_grammar_export(self):
        dog = build_acyclic_dog()
        terminals = Enumerator()
        data = dog.export_graph_json(terminals)
        with open('/tmp/json_graph_1.json', 'w') as file:
            json.dump(data, file)

        dsg = build_dsg()
        data = dsg.export_bihypergraph_json(terminals)
        with open('/tmp/json_bigraph_1.json', 'w') as file:
            json.dump(data, file)

        rule_dog = dog_se()
        data2 = rule_dog.export_graph_json(terminals)
        with open('/tmp/json_nonterminal_graph_1.json', 'w') as file:
            json.dump(data2, file)

        terminals.print_index()

        dsg = build_dsg()
        rec_part_strategy = the_recursive_partitioning_factory().get_partitioning('right-branching')[0]
        rec_part = rec_part_strategy(dsg)
        dcmp = compute_decomposition(dsg, rec_part)

        grammar = induce_grammar_from(dsg, rec_part, dcmp, labeling=simple_labeling, terminal_labeling=str)

        print(grammar)
        data = export_dog_grammar_to_json(grammar, terminals)
        with open('/tmp/json_grammar.json', 'w') as file:
            json.dump(data, file)

        with open('/tmp/json_corpus.json', 'w') as file:
            json.dump(export_corpus_to_json([dsg], terminals), file)

    def test_json_corpus_grammar_export(self):
        start = 1
        stop = 50
        # path = "res/tiger/tiger_release_aug07.corrected.16012013.utf8.xml"
        path = "res/tiger/tiger_8000.xml"
        exclude = []
        dsgs = sentence_names_to_deep_syntax_graphs(
            ['s' + str(i) for i in range(start, stop + 1) if i not in exclude]
            , path
            , hold=False)

        rec_part_strategy = the_recursive_partitioning_factory().get_partitioning('cfg')[0]

        def label_edge(edge):
            if isinstance(edge.label, ConstituentTerminal):
                return edge.label.pos()
            else:
                return edge.label

        nonterminal_labeling = lambda nodes, dsg: simple_labeling(nodes, dsg, label_edge)

        term_labeling_token = PosTerminals()

        def term_labeling(token):
            if isinstance(token, ConstituentTerminal):
                return term_labeling_token.token_label(token)
            else:
                return token

        grammar = induction_on_a_corpus(dsgs, rec_part_strategy, nonterminal_labeling, term_labeling)
        grammar.make_proper()

        terminals = Enumerator()

        data = export_dog_grammar_to_json(grammar, terminals)
        grammar_path = '/tmp/json_grammar.json'
        with open(grammar_path, 'w') as file:
            json.dump(data, file)

        corpus_path = '/tmp/json_corpus.json'
        with open(corpus_path, 'w') as file:
            json.dump(export_corpus_to_json(dsgs, terminals, terminal_labeling=term_labeling), file)

        with open('/tmp/enumerator.enum', 'w') as file:
            terminals.print_index(file)

        reduct_dir = '/tmp/reduct_grammars'
        if os.path.isdir(reduct_dir):
            shutil.rmtree(reduct_dir)
        os.makedirs(reduct_dir)
        p = subprocess.Popen([' '.join(
            ["java", "-jar", os.path.join("util", SCHICK_PARSER_JAR), 'dog-reduct', '-g', grammar_path, '-t',
             corpus_path, "-o", reduct_dir])], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        print("stdout", p.stdout.name)

        while True:
            nextline = p.stdout.readline()
            if nextline == b'' and p.poll() is not None:
                break
            print(nextline.decode('unicode_escape'), end='')
            # sys.stdout.write(nextline)
            # sys.stdout.flush()

        p.wait()
        p.stdout.close()
        self.assertEqual(0, p.returncode)

        rtgs = []
        for i in range(1, len(dsgs) + 1):
            rtgs.append(read_rtg('/tmp/reduct_grammars/' + str(i) + '.gra'))

        derivation_manager = PyDerivationManager(grammar)
        derivation_manager.convert_rtgs_to_hypergraphs(rtgs)
        derivation_manager.serialize(bytes('/tmp/reduct_manager.trace', encoding='utf8'))

        f = lambda token: token.pos() if isinstance(token, ConstituentTerminal) else token

        for i, (rtg, dsg) in enumerate(zip(rtgs, dsgs)):
            derivations = [LCFRSDerivationWrapper(der) for der in derivation_manager.enumerate_derivations(i, grammar)]
            self.assertGreaterEqual(len(derivations), 1)
            if len(derivations) > 1:
                print("Sentence", i)
                for der in derivations:
                    print(der)

            for der in derivations:
                dog, sync = dog_evaluation(der)
                dsg2 = DeepSyntaxGraph(der.compute_yield(), dog, sync)
                dsg.dog.project_labels(f)
                dsg.sentence = list(map(f, dsg.sentence))
                self.assertEqual(dsg.sentence, dsg2.sentence)
                morphs = dsg.dog.compute_isomorphism(dsg2.dog)
                self.assertFalse(morphs is None)
                self.assertListEqual([[morphs[0].get(node, node) for node in syncs]
                                      for syncs in dsg.synchronization], dsg2.synchronization)
        pass
                # print("i", i)
                # print("dsg", dsg.sentence, dsg.dog, dsg.synchronization)
                # print("dsg_reduct", dsg2.sentence, dsg2.dog, dsg2.synchronization)
                # print("morph", morphs[0])

    def test_frames(self):
        dsg = build_dsg()
        # print(dsg.labeled_frames())
        # print(dsg.labeled_frames(guard=lambda x: len(x[1]) > 0))
        # print(dsg.labeled_frames(replace_nodes_by_string_positions=False))
        scorer = PredicateArgumentScoring()

        def setify(frames):
            return set([(label, frozenset(args)) for label, args in frames])

        # print(scorer.extract_dependencies_from_frames(dsg.labeled_frames(), include_label=True))

        self.assertSetEqual(dsg.labeled_frames(),
                            setify([('CS', (((0, 1, 4, 5, 6), 'CJ'), ((2,), 'CD'), ((0, 3, 4, 5, 6), 'CJ'))),
                                    ('S', (((0,), 'SB'), ((1,), 'HD'), ((4, 5, 6), 'OA'))), ((2,), ()),
                                    ('S', (((0,), 'SB'), ((3,), 'HD'), ((4, 5, 6), 'OA'))), ((0,), ()), ((1,), ()),
                                    ('CNP', (((4,), 'CJ'), ((5,), 'CD'), ((6,), 'CJ'))), ((3,), ()), ((4,), ()),
                                    ((5,), ()),
                                    ((6,), ())])
                            )
        self.assertSetEqual(dsg.labeled_frames(guard=lambda x: len(x[1]) > 0),
                            setify(
                                [('CS', (((0, 1, 4, 5, 6), 'CJ'), ((2,), 'CD'), ((0, 3, 4, 5, 6), 'CJ'))),
                                 ('S', (((0,), 'SB'), ((1,), 'HD'), ((4, 5, 6), 'OA'))),
                                 ('S', (((0,), 'SB'), ((3,), 'HD'), ((4, 5, 6), 'OA'))),
                                 ('CNP', (((4,), 'CJ'), ((5,), 'CD'), ((6,), 'CJ')))]))
        self.assertSetEqual(dsg.labeled_frames(replace_nodes_by_string_positions=False),
                            setify(
                                [('CS', (('S', 'CJ'), ('und', 'CD'), ('S', 'CJ'))),
                                 ('S', (('Sie', 'SB'), ('entwickelt', 'HD'), ('CNP', 'OA'))), ('und', ()),
                                 ('S', (('Sie', 'SB'), ('druckt', 'HD'), ('CNP', 'OA'))), ('Sie', ()),
                                 ('entwickelt', ()),
                                 ('CNP', (('Verpackungen', 'CJ'), ('und', 'CD'), ('Etiketten', 'CJ'))), ('druckt', ()),
                                 ('Verpackungen', ()), ('und', ()), ('Etiketten', ())]))
        self.assertSetEqual(scorer.extract_dependencies_from_frames(dsg.labeled_frames(), include_label=True),
                            {('CS', (0, 1, 4, 5, 6), 'CJ'), ('CS', (2,), 'CD'), ('CS', (0, 3, 4, 5, 6), 'CJ'),
                             ('S', (0,), 'SB'), ('S', (1,), 'HD'), ('S', (4, 5, 6), 'OA'), ('S', (0,), 'SB'),
                             ('S', (3,), 'HD'), ('S', (4, 5, 6), 'OA'), ('CNP', (4,), 'CJ'), ('CNP', (5,), 'CD'),
                             ('CNP', (6,), 'CJ')})
        self.assertSetEqual(scorer.extract_dependencies_from_frames(dsg.labeled_frames(), include_label=True),
                            scorer.extract_dependencies_from_frames(dsg.labeled_frames(guard=lambda x: len(x[1]) > 0),
                                                                    include_label=True))

        self.assertSetEqual(scorer.extract_unlabeled_frames(dsg.labeled_frames()),
                            {((4,), frozenset([])), ((0,), frozenset([])), ((5,), frozenset([])),
                             ('CS', frozenset([(2,), (0, 3, 4, 5, 6), (0, 1, 4, 5, 6)])), ((3,), frozenset([])),
                             ('S', frozenset([(3,), (0,), (4, 5, 6)])), ('S', frozenset([(0,), (4, 5, 6), (1,)])),
                             ((1,), frozenset([])), ('CNP', frozenset([(5,), (6,), (4,)])), ((6,), frozenset([])),
                             ((2,), frozenset([]))}
                            )

    def test_subgrouping(self):
        start = 4
        stop = 4
        exclude = []
        path = "res/tiger/tiger_8000.xml"
        dsgs = sentence_names_to_deep_syntax_graphs(
            ['s' + str(i) for i in range(start, stop + 1) if i not in exclude]
            , path
            , hold=False
            , reorder_children=True)
        f = lambda token: token.pos() if isinstance(token, ConstituentTerminal) else token
        for dsg in dsgs:
            dsg.dog.project_labels(f)
            render_and_view_dog(dsg.dog, "tigerdsg4", "/tmp/")
            print(list(map(lambda x: x.form(), dsg.sentence)))
            print(dsg.synchronization)
            print(dsg.recursive_partitioning())
            print(fanout_limited_partitioning(dsg.recursive_partitioning(), 1))
            print(dsg.recursive_partitioning(subgrouping=True))
            print(fanout_limited_partitioning(dsg.recursive_partitioning(subgrouping=True), 1))
            self.assertTupleEqual(({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
                                   , [({0, 1, 2, 3, 4},
                                       [({0}, []), ({1}, []), ({2}, []), ({3, 4}, [({3}, []), ({4}, [])])]),
                                      ({5}, []),
                                      ({6, 7, 8, 9, 10, 11},
                                       [({8, 9, 10, 6, 7}, [({8, 6, 7}, [({6}, []), ({7}, []), ({8}, [])]),
                                           ({9, 10}, [({9}, []), ({10}, [])])]), ({11}, [])])
                                      ]),
                                  dsg.recursive_partitioning(subgrouping=True))
            self.assertTupleEqual(({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, [
                ({0, 1, 2, 3, 4, 5}, [({0, 1, 2, 3, 4}, [({0, 1, 2}, [({0, 1}, [({0}, []), ({1}, [])]), ({2}, [])]),
                                                         ({3, 4}, [({3}, []), ({4}, [])])]), ({5}, [])]), (
                    {6, 7, 8, 9, 10, 11}, [({8, 9, 10, 6, 7}, [({8, 6, 7}, [({6, 7}, [({6}, []), ({7}, [])]), (
                        {8}, [])]), ({9, 10}, [({9}, []), ({10}, [])])]), ({11}, [])])]),
                                  fanout_limited_partitioning(dsg.recursive_partitioning(subgrouping=True), 1))

    def test_fanout_marking(self):
        label = 's1813'
        path = "res/tiger/tiger_8000.xml"
        dsgs = sentence_names_to_deep_syntax_graphs([label], path, hold=False, reorder_children=True)

        term_labeling_token = PosTerminals()

        def term_labeling(token):
            if isinstance(token, ConstituentTerminal):
                return term_labeling_token.token_label(token)
            else:
                return token

        def rec_part_strategy(direction, subgrouping, fanout):
            if direction == "right-to-left":
                return lambda dsg: fanout_limited_partitioning(dsg.recursive_partitioning(subgrouping), fanout)
            else:
                return lambda dsg: fanout_limited_partitioning_left_to_right(dsg.recursive_partitioning(subgrouping),
                                                                             fanout)

        def label_edge(edge):
            if isinstance(edge.label, ConstituentTerminal):
                return edge.label.pos()
            else:
                return edge.label

        def stupid_edge(edge):
            return "X"

        def label_child(edge, j):
            return edge.get_function(j)

        def simple_nonterminal_labeling(nodes, dsg):
            return simple_labeling(nodes, dsg, label_edge)

        def bot_stupid_nonterminal_labeling(nodes, dsg):
            return top_bot_labeling(nodes, dsg, label_edge, stupid_edge)

        def missing_child_nonterminal_labeling(nodes, dsg):
            return missing_child_labeling(nodes, dsg, label_edge, label_child)

        rec_part = rec_part_strategy("left-to-right", True, 2)(dsgs[0])
        print(rec_part)

        dcmp = compute_decomposition(dsgs[0], rec_part)
        print(dcmp)

        grammar = induce_grammar_from(dsgs[0], rec_part, dcmp, labeling=missing_child_nonterminal_labeling,
                                      terminal_labeling=term_labeling)

        print(grammar)

        for rule in grammar.rules():
            print(rule)

    def test_binarization(self):
        dog = build_acyclic_dog()

        bin_dog_control = build_acyclic_dog_binarized()
        # render_and_view_dog(bin_dog_control, 'binarized_dog')
        self.assertTrue(dog.primary_is_tree())

        bin_dog = dog.binarize()
        #render_and_view_dog(bin_dog, 'binarized_auto')
        self.assertEqual(bin_dog, bin_dog_control)
        self.assertTrue(bin_dog.primary_is_tree())

        debin_dog = bin_dog.debinarize()
        # render_and_view_dog(debin_dog, 'debinerized')
        self.assertEqual(dog, debin_dog)

    def test_primary_tree_violation_workaround(self):
        label = 's150'
        label2 = 's6516'
        path = "res/tiger/tiger_8000.xml"
        train_dsgs = sentence_names_to_deep_syntax_graphs([label, label2], path, hold=False, reorder_children=True)
        binarize = True

        # Grammar induction
        term_labeling_token = PosTerminals()

        def label_edge(edge):
            if isinstance(edge.label, ConstituentTerminal):
                return edge.label.pos()
            else:
                return edge.label

        def term_labeling(token):
            if isinstance(token, ConstituentTerminal):
                return term_labeling_token.token_label(token)
            else:
                return token

        if binarize:
            def modify_token(token):
                if isinstance(token, ConstituentCategory):
                    token_new = deepcopy(token)
                    token_new.set_category(token.category() + '-BAR')
                    return token_new
                elif isinstance(token, str):
                    return token + '-BAR'
                else:
                    assert False

            train_dsgs = [dsg.binarize(bin_modifier=modify_token) for dsg in train_dsgs]

            def is_bin(token):
                if isinstance(token, ConstituentCategory):
                    if token.category().endswith('-BAR'):
                        return True
                elif isinstance(token, str):
                    if token.endswith('-BAR'):
                        return True
                return False

            def debinarize(dsg):
                return dsg.debinarize(is_bin=is_bin)

        else:
            debinarize = id

        def rec_part_strategy(direction, subgrouping, fanout):
            if direction == "right-to-left":
                return lambda dsg: fanout_limited_partitioning(dsg.recursive_partitioning(subgrouping), fanout)
            else:
                return lambda dsg: fanout_limited_partitioning_left_to_right(
                    dsg.recursive_partitioning(subgrouping, weak=True),
                    fanout)
        the_rec_part_strategy = rec_part_strategy("left-to-right", True, 1)

        def simple_nonterminal_labeling(nodes, dsg):
            return simple_labeling(nodes, dsg, label_edge)
        # render_and_view_dog(train_dsgs[0].dog, 'train_dsg_tmp')
        grammar = induction_on_a_corpus(train_dsgs, the_rec_part_strategy, simple_nonterminal_labeling, term_labeling)

    def test_dog_generation(self):
        for i in range(10):
            dog = generate(randint(2, 12), maximum_inputs=4, new_output=0.4, upward_closed=True)
            render_and_view_dog(dog, 'random_dog_' + str(i))


def build_acyclic_dog():
    dog = DirectedOrderedGraph()
    for i in range(11):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([(1, 'p'), (2, 'p'), (3, 'p')], 'CS', 0) \
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    dog.add_terminal_edge([(4, 'p'), (5, 'p'), 6], 'S', 1) \
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_terminal_edge([4, (7, 'p'), (6, 'p')], 'S', 3) \
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_terminal_edge([(8, 'p'), (9, 'p'), (10, 'p')], 'CNP', 6) \
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    for (lab, i) in [
        ('und', 2), ('Sie', 4), ('entwickelt', 5), ('druckt', 7),
        ('Verpackungen', 8), ('und', 9), ('Etiketten', 10)
    ]:
        dog.add_terminal_edge([], lab, i)

    return dog


def build_acyclic_dog_permuted():
    dog = DirectedOrderedGraph()
    for i in range(11):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([1, 2, 3], 'CS', 0) \
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    dog.add_terminal_edge([4, 5, 6], 'S', 1) \
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_terminal_edge([4, 7, 6], 'S', 3) \
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_terminal_edge([8, 9, 10], 'CNP', 6) \
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    for (lab, i) in [
        ('und', 2), ('Sie', 4), ('entwickelt', 7), ('druckt', 5),
        ('Verpackungen', 8), ('und', 9), ('Etiketten', 10)
    ]:
        dog.add_terminal_edge([], lab, i)

    return dog


def build_acyclic_dog_binarized():
    dog = DirectedOrderedGraph()
    for i in range(15):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([(1, 'p'), (11, 'p')], 'CS', 0) \
        .set_function(0, "CJ")
    dog.add_terminal_edge([(2, 'p'), (3, 'p')], 'CS-BAR', 11) \
        .set_function(0, "CD").set_function(1, "CJ")
    dog.add_terminal_edge([(4, 'p'), (12, 'p')], 'S', 1) \
        .set_function(0, "SB")
    dog.add_terminal_edge([(5, 'p'), 6], 'S-BAR', 12) \
        .set_function(0, "HD").set_function(1, "OA")
    dog.add_terminal_edge([4, (13, 'p')], 'S', 3) \
        .set_function(0, "SB")
    dog.add_terminal_edge([(7, 'p'), (6, 'p')], 'S-BAR', 13) \
        .set_function(0, "HD").set_function(1, "OA")
    dog.add_terminal_edge([(8, 'p'), (14, 'p')], 'CNP', 6) \
        .set_function(0, "CJ")
    dog.add_terminal_edge([(9, 'p'), (10, 'p')], 'CNP-BAR', 14) \
        .set_function(0, "CD").set_function(1, "CJ")
    for (lab, i) in [
        ('und', 2), ('Sie', 4), ('entwickelt', 5), ('druckt', 7),
        ('Verpackungen', 8), ('und', 9), ('Etiketten', 10)
    ]:
        dog.add_terminal_edge([], lab, i)

    return dog


def build_terminal_dog(terminal):
    dog = DirectedOrderedGraph()
    dog.add_node(0)
    dog.add_to_outputs(0)
    dog.add_terminal_edge([], terminal, 0)
    return dog


def dog_se():
    dog = DirectedOrderedGraph()
    for i in range(6):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([1, 2, 3], 'CS', 0) \
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    dog.add_nonterminal_edge([5], [1, 4])
    dog.add_nonterminal_edge([], [2])
    dog.add_nonterminal_edge([4], [5, 3])
    return dog


def dog_s1():
    dog = DirectedOrderedGraph()
    for i in range(4):
        dog.add_node(i)
    dog.add_to_outputs(0)
    dog.add_to_outputs(1)
    dog.add_to_inputs(3)

    dog.add_terminal_edge([1, 2, 3], 'S', 0) \
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_nonterminal_edge([], [1])
    dog.add_nonterminal_edge([], [2])

    return dog


def dog_s2():
    return build_terminal_dog('und')


def dog_s3():
    dog = DirectedOrderedGraph()
    for i in range(4):
        dog.add_node(i)
    dog.add_to_outputs(3)
    dog.add_to_outputs(0)
    dog.add_to_inputs(1)

    dog.add_terminal_edge([1, 2, 3], 'S', 0) \
        .set_function(0, "SB").set_function(1, "HD").set_function(2, "OA")
    dog.add_nonterminal_edge([], [2])
    dog.add_nonterminal_edge([], [3])

    return dog


def dog_s11():
    return build_terminal_dog('Sie')


def dog_s12():
    return build_terminal_dog('entwickelt')


def dog_s32():
    return build_terminal_dog('druckt')


def dog_s13():
    dog = DirectedOrderedGraph()
    for i in range(4):
        dog.add_node(i)
    dog.add_to_outputs(0)

    dog.add_terminal_edge([1, 2, 3], 'CNP', 0) \
        .set_function(0, "CJ").set_function(1, "CD").set_function(2, "CJ")
    for i in range(1, 4):
        dog.add_nonterminal_edge([], [i])
    return dog


def dog_s131():
    return build_terminal_dog('Verpackungen')


def dog_s132():
    return build_terminal_dog('und')


def dog_s133():
    return build_terminal_dog('Etiketten')


def build_dsg():
    dog = build_acyclic_dog()
    sentence = ["Sie", "entwickelt", "und", "druckt", "Verpackungen", "und", "Etiketten"]
    synchronization = [[4], [5], [2], [7], [8], [9], [10]]
    return DeepSyntaxGraph(sentence, dog, synchronization)


if __name__ == '__main__':
    unittest.main()
