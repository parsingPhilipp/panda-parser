from libcpp.vector cimport vector
ctypedef unsigned long unsigned_long

cdef class Enumerator:
    cpdef unsigned_long counter
    cpdef dict obj_to_ind
    cpdef dict ind_to_obj
    cpdef unsigned_long first_index

    cpdef unsigned_long object_index(self, obj)
    cpdef vector[unsigned_long] objects_indices(self, objects)