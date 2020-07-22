from unittest import TestCase

import optimusbeez as ob
import numpy as np

class TestPoint(TestCase):
	# Check zero points
	def test_Rosenbrock_zero(self):
		f = ob.Rosenbrock((1,1,1,1,1))
		self.assertTrue(f==0)
	def test_Alpine_zero(self):
		f = ob.Alpine((0,0,0,0))
		self.assertTrue(f==0)
	def test_Griewank_zero(self):
		f = ob.Griewank((0,0,0))
		self.assertTrue(f==0)

	# Check non-zero points
	def test_Rosenbrock_point(self):
		f = ob.Rosenbrock((4,7))
		self.assertTrue(f==8109)
	def test_Alpine_point(self):
		f = ob.Alpine((15,4))
		self.assertTrue(np.isclose(f,13.88152758,))
	def test_Griewank_point(self):
		f = ob.Griewank((-3,6))
		self.assertTrue(np.isclose(f,0.563118157))