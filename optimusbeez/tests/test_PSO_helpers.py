from unittest import TestCase

import optimusbeez as ob

class TestPoint(TestCase):

	def test_number_evaluations(self):
		n = ob.determine_n_evaluations(3, 4, 5)
		self.assertTrue(n == 75)