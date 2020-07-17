from unittest import TestCase

import optimusbeez as ob

class TestPoint(TestCase):
	def test_is_float(self):
		f = ob.evaluate((1,1), "Rosenbrock")
		self.assertTrue(f==0)