import re

from decomposition import left_branching_partitioning, right_branching_partitioning, fanout_limited_partitioning

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


class RecursivePartitioningFactory:
    def __init__(self):
        self.__partitionings = {}

    def registerPartitioning(self, name, partitioning):
        self.__partitionings[name] = partitioning

    def getPartitioning(self, name):
        partitioning_names = name.split(',')
        partitionings = []
        for name in partitioning_names:
            match = re.search(r'fanout-(\d+)', name)
            if match:
                k = int(match.group(1))
                rec_par = lambda tree: fanout_k(tree, k)
                rec_par.__name__ = 'fanout_' + str(k)
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