from unittest import TestCase

import optimusbeez as ob
import numpy as np

class TestPoint(TestCase):

	def test_Rosenbrock_zero(self):
		f = ob.evaluate((1,1), "Rosenbrock")
		self.assertTrue(f==0)
	def test_Alpine_zero(self):
		f = ob.evaluate((0,0), "Alpine")
		self.assertTrue(f==0)
	def test_Griewank_zero(self):
		f = ob.evaluate((0,0), "Griewank")
		self.assertTrue(f==0)
	def test_Ackley_zero(self):
		f = ob.evaluate((0,0), "Ackley")
		self.assertTrue(f==0)

	def test_Rosenbrock_point(self):
		f = ob.evaluate((4,7), "Rosenbrock")
		self.assertTrue(f==8109)
	def test_Alpine_point(self):
		f = ob.evaluate((15,4), "Alpine")
		self.assertTrue(np.isclose(f,13.88152758,))
	def test_Griewank_point(self):
		f = ob.evaluate((-3,6), "Griewank")
		self.assertTrue(np.isclose(f,0.563118157))
	def test_Ackley_point(self):
		f = ob.evaluate((-2,-2), "Ackley")
		self.assertTrue(np.isclose(f,6.593599079))

	def test_invalid_fn_name(self):
		f = ob.evaluate((1,1), "doesnotexist")
		self.assertTrue(f==None)