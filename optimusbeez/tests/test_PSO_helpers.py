from unittest import TestCase

import optimusbeez as ob

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