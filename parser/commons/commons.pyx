cpdef void output_helper_utf8(str s):
    output_helper(bytes(s, encoding="utf-8"))