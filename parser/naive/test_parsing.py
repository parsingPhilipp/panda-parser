__author__ = 'kilian'

import unittest
from parsing import *


class MyTestCase(unittest.TestCase):
    def test_LHS_instance(self):
        lhs = LHS_instance("A")
        lhs.add_arg()
        for mem in [Span(0, 2), "foo", Span(3, 4)]:
            lhs.add_mem(mem)
        self.assertEqual(lhs.consistent(), True)

        lhs = LHS_instance("A")
        lhs.add_arg()
        for mem in [Span(0, 2), Span(3, 4)]:
            lhs.add_mem(mem)
        self.assertEqual(lhs.consistent(), False)
        self.assertEqual(lhs.next_member_bounds(0, 3), (0, 3))

        lhs = LHS_instance("A")
        lhs.add_arg()
        for mem in [Span(0, 2), Span(2, 4)]:
            lhs.add_mem(mem)
        lhs.add_arg()

        for mem in [Span(6, 9), Span(9, 12)]:
            lhs.add_mem(mem)

        self.assertEqual(lhs.consistent(), True)

        self.assertEqual(str(lhs), "A([0-2] [2-4]; [6-9] [9-12])")

        lhs.collapse()

        self.assertEqual(str(lhs), "A([0-4]; [6-12])")


if __name__ == '__main__':
    unittest.main()
