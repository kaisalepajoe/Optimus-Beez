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
<<<<<<< HEAD
		f = ob.Griewank((-3,6))
		self.assertTrue(np.isclose(f,0.563118157))
=======
		f = ob.evaluate((-3,6), "Griewank")
		self.assertTrue(np.isclose(f,0.563118157))

	# Check non-zero multidimensional points
	def test_Rosenbrock_5dim_point(self):
		f = ob.evaluate((1,2,3,4,5), "Rosenbrock")
		self.assertTrue(f==14814)

	# Test input types and dimensions
	def test_invalid_fn_name_type(self):
		self.assertRaises(TypeError, ob.evaluate, (1,1), ["Griewank"])
	def test_undefined_fn_name(self):
		self.assertRaises(ValueError, ob.evaluate, (1,1), "Undefined function")
	def test_invalid_pos_type(self):
		self.assertRaises(TypeError, ob.evaluate, "[1,1]", "Alpine")

	# Test different input types for position
	def test_tuple_position(self):
		f = ob.evaluate((4,7), "Rosenbrock")
		self.assertTrue(f==8109)
	def test_list_position(self):
		f = ob.evaluate([4,7], "Rosenbrock")
		self.assertTrue(f==8109)
	def test_np_array_position(self):
		f = ob.evaluate(np.array([4,7]), "Rosenbrock")
		self.assertTrue(f==8109)
>>>>>>> c7f69704df42b03c5b1813dcd6edbfd3c71ea5bc
