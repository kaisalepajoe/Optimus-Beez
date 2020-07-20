from unittest import TestCase

import optimusbeez as ob
import numpy as np

class TestExperimentDictionaryType(TestCase):
	def test_init_invalid_dictionary_type_1(self):
		self.assertRaises(TypeError, ob.Experiment, "not dictionary", {})
	def test_init_invalid_dictionary_type_2(self):
		self.assertRaises(TypeError, ob.Experiment, {}, "not dictionary")

class TestExperiment(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 5, 'k': 3, 'time_steps': 10, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "xmin":-100, "xmax":100, "show_animation":False}
		np.random.seed(123)
		self.experiment = ob.Experiment(self.constants, self.fn_info)

	def test_init_N(self):
		self.assertTrue(self.experiment.N == 5)
	def test_init_time_steps(self):
		self.assertTrue(self.experiment.time_steps == 10)
	def test_init_repetitions(self):
		self.assertTrue(self.experiment.repetitions == 1)
	def test_init_fn_name(self):
		self.assertTrue(self.experiment.fn_name == "Rosenbrock")
	def test_init_optimal_f(self):
		self.assertTrue(self.experiment.optimal_f == 0)
	def test_init_k(self):
		self.assertTrue(self.experiment.k == 3)
	def test_init_phi(self):
		self.assertTrue(self.experiment.phi == 2.4)
	def test_init_xmin(self):
		self.assertTrue(self.experiment.xmin == -100)
	def test_init_xmax(self):
		self.assertTrue(self.experiment.xmax == 100)
	def test_init_show_animation(self):
		self.assertTrue(self.experiment.show_animation == False)
	def test_init_vmax(self):
		self.assertTrue(self.experiment.vmax == 100)
	def test_init_c1(self):
		self.assertTrue(np.isclose(self.experiment.c1, 0.420204102))
	def test_init_cmax(self):
		self.assertTrue(np.isclose(self.experiment.cmax, 1.008489847))


	def test_constants_invalid_dictionary_type(self):
		self.assertRaises(TypeError, self.experiment.constants, "not dictionary")
	def test_constants_returns_same_dictionary(self):
		self.assertTrue(self.experiment.constants() == self.constants)
	def test_constants_assigns_correct_variables(self):
		new_constants = {'phi': 0, 'N': 0, 'k': 0, 'time_steps': 0, 'repetitions': 0}
		self.assertTrue(self.experiment.constants(new_constants) == new_constants)

	def test_fn_info_invalid_dictionary_type(self):
		self.assertRaises(TypeError, self.experiment.fn_info(), "not dictionary")
	def test_fn_info_returns_same_dictionary(self):
		self.assertTrue(self.experiment.fn_info() == self.fn_info)
	def test_fn_info_assigns_correct_variables(self):
		new_fn_info = {"fn_name":"Alpine", "optimal_f":5, "xmin":-300, "xmax":300, "show_animation":True}
		self.assertTrue(self.experiment.fn_info(new_fn_info) == new_fn_info)