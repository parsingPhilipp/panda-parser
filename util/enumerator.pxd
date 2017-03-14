from libcpp.vector cimport vector

ctypedef unsigned long unsigned_long

cdef class Enumerator:
    cdef unsigned_long counter
    cdef dict obj_to_ind
    cdef dict ind_to_obj
    cdef unsigned_long first_index

    cdef unsigned_long object_index(self, obj)
    cdef objects_indices(self, objects)