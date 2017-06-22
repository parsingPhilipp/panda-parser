import re
from decomposition import left_branching_partitioning, right_branching_partitioning, fanout_limited_partitioning, fanout_limited_partitioning_left_to_right, fanout_limited_partitioning_argmax, fanout_limited_partitioning_random_choice, fanout_limited_partitioning_no_new_nont
from random import seed

# Recursive partitioning strategies

def left_branching(tree):
    return left_branching_partitioning(len(tree.id_yield()))


def right_branching(tree):
    return right_branching_partitioning(len(tree.id_yield()))


def direct_extraction(tree):
    return tree.recursive_partitioning()


def cfg(tree):
    return fanout_k(tree, 1)


fanout_k = lambda tree, k: fanout_limited_partitioning(tree.recursive_partitioning(), k)
fanout_k_left_to_right = lambda tree, k: fanout_limited_partitioning_left_to_right(tree.recursive_partitioning(), k)
fanout_k_argmax = lambda tree, k: fanout_limited_partitioning_argmax(tree.recursive_partitioning(), k)
fanout_k_random = lambda tree, k: fanout_limited_partitioning_random_choice(tree.recursive_partitioning(), k)
fanout_k_no_new_nont = lambda tree, nonts, nont_labelling, fallback, k: fanout_limited_partitioning_no_new_nont(tree.recursive_partitioning(), k, tree, nonts, nont_labelling, fallback)


class RecursivePartitioningFactory:
    def __init__(self):
        self.__partitionings = {}

    def registerPartitioning(self, name, partitioning):
        self.__partitionings[name] = partitioning

    def getPartitioning(self, name):
        partitioning_names = name.split(',')
        partitionings = []
        for name in partitioning_names:
            match = re.search(r'fanout-(\d+)([-\w]*)', name)
            if match:
                k = int(match.group(1))
                trans = match.group(2)
                if trans == '': #right-to-left bfs
                    rec_par = lambda tree: fanout_k(tree, k)
                    rec_par.__name__ = 'fanout_' + str(k)
                    partitionings.append(rec_par)
                if trans == '-left-to-right':
                    rec_par = lambda tree: fanout_k_left_to_right(tree, k)
                    rec_par.__name__ = 'fanout_' + str(k) + '_left_to_right'
                    partitionings.append(rec_par)
                if trans == '-argmax':
                    rec_par = lambda tree: fanout_k_argmax(tree, k)
                    rec_par.__name__ = 'fanout_' + str(k) + '_argmax'
                    partitionings.append(rec_par)
                #set seed, if random strategy is chosen
                randMatch = re.search(r'-random-(\d*)', trans)
                if randMatch:
                    s = int(randMatch.group(1))
                    seed(s)
                    rec_par = lambda tree: fanout_k_random(tree, k)
                    rec_par.__name__ = 'fanout_' + str(k) + '_random'
                    partitionings.append(rec_par)
                #set fallback strategy if no position corresponds to an existing nonterminal
                noNewMatch = re.search(r'-no-new-nont([-\w]*)', trans)
                if noNewMatch:
                    fallback = noNewMatch.group(1)
                    randMatch = re.search(r'-random-(\d*)', fallback)
                    if randMatch:
                        s = int(randMatch.group(1))
                        seed(s)
                        fallback = '-random'
                    rec_par = lambda tree, nonts, nont_labelling: fanout_k_no_new_nont(tree, nonts, nont_labelling, k, fallback)
                    rec_par.__name__ = 'fanout_' + str(k) + '_no_new_nont'
                    partitionings.append(rec_par)
            else:
                rec_par = self.__partitionings[name]
                if rec_par:
                    partitionings.append(rec_par)
                else:
                    return None
        if partitionings:
            return partitionings
        else:
            return None


def the_recursive_partitioning_factory():
    factory = RecursivePartitioningFactory()
    factory.registerPartitioning('left-branching', left_branching)
    factory.registerPartitioning('right-branching', right_branching)
    factory.registerPartitioning('direct-extraction', direct_extraction)
    factory.registerPartitioning('cfg', cfg)
    return factory