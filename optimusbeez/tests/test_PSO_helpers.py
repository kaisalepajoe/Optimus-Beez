from unittest import TestCase

import optimusbeez as ob
import numpy as np

class TestPoint(TestCase):

	# Test determine error function
	def test_determine_error_for_positives(self):
		error = ob.determine_error(2,3)
		self.assertTrue(error == 1)
	def test_determine_error_for_1_negative(self):
		error = ob.determine_error(-2,3)
		self.assertTrue(error == 5)
	def test_determine_error_for_2_negative(self):
		error = ob.determine_error(2,-4)
		self.assertTrue(error == 6)
	def test_determine_error_for_negatives(self):
		error = ob.determine_error(-2,-1)
		self.assertTrue(error == 1)

	# Test dictionary functions
	def test_read_undefined_file(self):
		self.assertRaises(NameError, ob.read_dictionary_from_file, "undefined.txt")
	def test_write_non_dictionary(self):
		self.assertRaises(TypeError, ob.write_dictionary_to_file, "string", "/pathdoesnotexist/undefined.txt")
	def test_write_to_invalid_path(self):
		self.assertRaises(NameError, ob.write_dictionary_to_file, {}, "/pathdoesnotexist/undefined.txt")

	def test_generate_random_constants_time_less_than_deviation(self):
		self.assertRaises(ValueError, ob.generate_random_constants, 10, 100)
	def test_generate_random_constants_negative_allowed_evaluations(self):
		self.assertRaises(ValueError, ob.generate_random_constants, -100, 10)
	def test_generate_random_constants_negative_allowed_deviation(self):
		self.assertRaises(ValueError, ob.generate_random_constants, 10, -10)
	def test_generate_random_constants_generic_output(self):
		np.random.seed(123)
		self.assertTrue(np.all(np.isclose(ob.generate_random_constants(100, 10), [2.69649954, 7., 7., 1., 6.])))
	def test_generate_random_constants_output_type(self):
		self.assertTrue(type(ob.generate_random_constants(100,10)) == np.ndarray)
	def test_generate_random_constants_zero_deviation_output(self):
		np.random.seed(123)
		self.assertTrue(np.all(np.isclose(ob.generate_random_constants(10, 0), [2.69649954, 1.        , 1.        , 1.        , 3.        ])))

	def test_generate_random_constants_few_allowed_evaluations_output(self):
		np.random.seed(123)
		self.assertTrue(np.all(np.isclose(ob.generate_random_constants(4,0), [2.69649954, 1.        , 1.        , 2.        , 1.        ])))
