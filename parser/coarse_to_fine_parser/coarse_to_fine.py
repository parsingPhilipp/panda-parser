from parser.parser_interface import AbstractParser
from parser.coarse_to_fine_parser.ranker import rank_derivations, build_ranker
from parser.trace_manager.sm_trainer import PyLatentAnnotation
import math

class Coarse_to_fine_parser(AbstractParser):
    def best_derivation_tree(self):
        for _, der in self.k_best_derivation_trees():
            return der

    def parse(self):
        self.base_parser.parse()

    def best(self):
        for weight, _ in self.k_best_derivation_trees():
            return weight

    def set_input(self, input):
        self.base_parser.set_input(input)

    def clear(self):
        self.base_parser.clear()
        self.k_best_list = None
        self.ranking = None
        self.ranker = None

    def all_derivation_trees(self):
        pass

    def recognized(self):
        return self.base_parser.recognized()

    def __init__(self, grammar, base_parser_type, la, grammarInfo, nontMap, input=None, save_preprocess=None, load_preprocess=None, k=50):
        self.grammar = grammar
        self.base_parser = base_parser_type(grammar, input=input, save_preprocess=save_preprocess, load_preprocess=load_preprocess, k=k)
        self.la = [la] if isinstance(la, PyLatentAnnotation) else la
        self.grammarInfo = grammarInfo
        self.nontMap = nontMap
        self.ranking = None
        self.ranker = None
        self.k_best_list = None

    def k_best_derivation_trees(self):
        if self.k_best_list is None:

            self.k_best_list = []
            self.ranking = []
            for i, (weight, der) in enumerate(self.base_parser.k_best_derivation_trees()):
                self.k_best_list.append(der)
                self.ranking.append((i, weight))

            length = len(self.ranking)
            for ref, la in enumerate(self.la):
                if self.ranker is None:
                    self.ranker = build_ranker(self.k_best_list, self.grammar, self.grammarInfo, self.nontMap)

                new_ranking = self.ranker.rank(la)
                self.ranker.clean_up()

                # a simple heuristic, one could also back-off between those lists
                i = 0
                if new_ranking[-1][1] == 0.0:
                    # todo: some binary search could be used here for efficency
                    while i < len(new_ranking) and new_ranking[i][1] > 0.0:
                        i += 1
                    if i < length * 0.1:
                        break
                length = i
                self.ranking = new_ranking
                # print "Ranking: ", ref, self.ranking

            # print

        for idx, weight in self.ranking:
            if not (math.isinf(weight) or math.isnan(weight)):
                yield weight, self.k_best_list[idx]

    def resolve_path(self, preprocess_path):
        return self.base_parser.resolve_path(preprocess_path)





