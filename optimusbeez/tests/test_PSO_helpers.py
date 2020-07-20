from unittest import TestCase

import optimusbeez as ob

class TestPoint(TestCase):

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