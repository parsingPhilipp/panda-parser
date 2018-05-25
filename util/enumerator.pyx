from __future__ import print_function
import sys
cdef class Enumerator:
    def __init__(self, unsigned_long first_index=0):
        self.first_index = first_index
        self.counter = first_index
        self.obj_to_ind = {}
        self.ind_to_obj = {}

    def index_object(self, int i):
        """
        :type i: int
        :return:
        """
        return self.ind_to_obj[i]

    cpdef unsigned_long object_index(self, obj):
        if obj in self.obj_to_ind:
            return self.obj_to_ind[obj]
        else:
            self.obj_to_ind[obj] = self.counter
            self.ind_to_obj[self.counter] = obj
            self.counter += 1
            return self.counter - 1

    cpdef vector[unsigned_long] objects_indices(self, objects):
        result = vector[unsigned_long]()
        for obj in objects:
            result += [self.object_index(obj)]
        return result

    def get_first_index(self):
        return self.first_index

    def get_counter(self):
        return self.counter

# cdef class Enumerator:
#     cdef int first_index
#     cdef int counter
#     cdef dict obj_to_ind
#     cdef dict ind_to_obj
#
#     def __init__(self, first_index=1):
#         self.first_index = first_index
#         self.counter = first_index - 1
#         self.obj_to_ind = {}
#         self.ind_to_obj = {}
#
#     def index_object(self, i):
#         """
#         :type i: int
#         :return:
#         """
#         return self.ind_to_obj[i]
#
#     def object_index(self, obj):
#         i = self.obj_to_ind.get(obj, None)
#         if i:
#             return i
#         else:
#             self.counter += 1
#             self.obj_to_ind[obj] = self.counter
#             self.ind_to_obj[self.counter] = obj
#             return self.counter


    def print_index(self, to_file=sys.stdout):
        for i in range(self.first_index, self.counter):
            print(str(i) + " " + str(self.index_object(i)), file=to_file)
'''
    def print_index_and_stats(self, grammar, inh, syn):
        fanouts = defaultdict(lambda: 0)
        inherits = defaultdict(lambda: 0)
        synths = defaultdict(lambda: 0)
        args = defaultdict(lambda: 0)
        max_fanout = 0
        max_inh = 0
        max_syn = 0
        max_args = 0
        for i in range (self.first_index, self.counter + 1):
            fanout = grammar.fanout(self.index_object(i))
            fanouts[fanout] += 1
            max_fanout = max(max_fanout, fanout)
            inherits[inh[i]] += 1
            max_inh = max(max_inh, inh[i])
            synths[syn[i]] += 1
            max_syn = max(max_syn, syn[i])
            args[inh[i] + syn[i]] += 1
            max_args = max(max_args, inh[i] + syn[i])
            print >>self.file, i, self.index_object(i), fanout, inh[i], syn[i]
        return max_fanout, max_inh, max_syn, max_args, fanouts, inherits, synths, args
'''