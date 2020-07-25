from unittest import TestCase

import optimusbeez as ob
import numpy as np

class TestRegularExperiment(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 5, 'k': 3, 'time_steps': 10, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "dim":2, "xmin":[-100,-100], "xmax":[100,100],\
		"param_is_integer":[False, False], "special_constraints":False, "constraints_extra_arguments":"an argument",\
		"show_animation":False}
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
		self.assertTrue(np.all(self.experiment.xmin == np.array([-100,-100])))
	def test_init_xmax(self):
		self.assertTrue(np.all(self.experiment.xmax == np.array([100,100])))
	def test_param_is_integer(self):
		self.assertTrue(np.all(self.experiment.param_is_integer == np.array([False, False])))
	def test_special_constraints(self):
		self.assertTrue(self.experiment.special_constraints == False)
	def test_constraints_extra_arguments(self):
		self.assertTrue(self.experiment.constraints_extra_arguments == 'an argument')
	def test_init_show_animation(self):
		self.assertTrue(self.experiment.show_animation == False)
	def test_init_vmax(self):
		self.assertTrue(np.all(self.experiment.vmax == np.array([100,100])))
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
		new_fn_info = {"fn_name":"Alpine", "optimal_f":5, "dim":3, "xmin":[-300,100], "xmax":[300,200], \
			"param_is_integer":[True, True], "special_constraints":[True, True], "show_animation":True}
		self.assertTrue(self.experiment.fn_info(new_fn_info) == new_fn_info)


	def test_n_evaluations_invalid_input_type(self):
		self.assertRaises(TypeError, self.experiment.n_evaluations(), 10, 10, 10.5)
	def test_n_evaluations_without_input(self):
		self.assertTrue(self.experiment.n_evaluations() == 55)
	def test_n_evaluations_with_input(self):
		self.assertTrue(self.experiment.n_evaluations(10, 10, 10) == 1100)


	def test_generate_random_constants_non_integer_input(self):
		self.assertRaises(TypeError, self.experiment.generate_random_constants, 100, 10.5)
	def test_generate_random_constants_time_less_than_deviation(self):
		self.assertRaises(ValueError, self.experiment.generate_random_constants, 10, 100)
	def test_generate_random_constants_negative_allowed_evaluations(self):
		self.assertRaises(ValueError, self.experiment.generate_random_constants, -100, 10)
	def test_generate_random_constants_negative_allowed_deviation(self):
		self.assertRaises(ValueError, self.experiment.generate_random_constants, 10, -10)
	def test_generate_random_constants_generic_output(self):
		self.assertTrue(self.experiment.generate_random_constants(100, 10) == {'phi': 2.2205303944854657, 'N': 2, 'k': 2, 'time_steps': 4, 'repetitions': 9})
	def test_generate_random_constants_output_type(self):
		self.assertTrue(type(self.experiment.generate_random_constants(100,10)) == dict)
	def test_generate_random_constants_zero_deviation_output(self):
		self.assertTrue(self.experiment.generate_random_constants(10, 0) == {'phi': 2.0907483129111455, 'N': 1, 'k': 1, 'time_steps': 9, 'repetitions': 1})
	def test_generate_random_constants_few_allowed_evaluations_output(self):
		self.assertTrue(self.experiment.generate_random_constants(2,0) == {'phi': 2.278590709547289, 'N': 1, 'k': 1, 'time_steps': 1, 'repetitions': 1})
	def test_run_n_evaluations_too_small(self):
		self.assertRaises(ValueError, self.experiment.run, 2)


class TestSwarmIndividualFunctions(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 5, 'k': 3, 'time_steps': 10, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "dim":4, "xmin":[-10,-10,-10,-10], "xmax":[10,10,10,10],\
			"param_is_integer":[False, False, True, True], "special_constraints":[False,False,False,False], "show_animation":False}
		np.random.seed(123)
		self.swarm = ob.Swarm(self.constants, self.fn_info)

	def test_random_initial_positions(self):
		initial_positions = self.swarm.random_initial_positions()
		expected_result = np.array([[  3.92938371,  -1.5378708 ,   4.        ,  -6.        ],
       [ -4.2772133 ,   9.61528397, -10.        , -10.        ],
       [ -5.46297093,   3.69659477,   5.        ,   6.        ],
       [  1.02629538,  -0.38136197,   9.        ,  -6.        ],
       [  4.3893794 ,  -2.15764964,   4.        ,   7.        ]])
		self.assertTrue(np.all(np.isclose(initial_positions, expected_result)))

	def test_random_initial_velocities(self):
		initial_velocities = self.swarm.random_initial_velocities()
		expected_result = np.array([[ 3.92938371, -4.2772133 , -5.46297093,  1.02629538],
       [ 4.3893794 , -1.5378708 ,  9.61528397,  3.69659477],
       [-0.38136197, -2.15764964, -3.13643968,  4.58099415],
       [-1.22855511, -8.80644207, -2.03911489,  4.75990811],
       [-6.35016539, -6.49096488,  0.63102748,  0.63655174]])
		self.assertTrue(np.all(np.isclose(initial_velocities, expected_result)))
		self.assertTrue(np.all(initial_velocities >= -self.swarm.vmax))
		self.assertTrue(np.all(initial_velocities < self.swarm.vmax))



'''
	def test_random_informants_number_of_informants(self):
		self.swarm.random_informants()
		lengths = []
		for particle in self.swarm.particles:
			self.assertTrue(len(particle.informants) == 3)
'''
'''
bad test
	def test_get_parameters_output(self):
		parameters = self.swarm.get_parameters()
		comparison = np.isclose(parameters, np.array([-3.81361970e+00, -2.15764964e+01,  1.30489995e+05]))
		self.assertTrue(np.all(comparison))
'''
'''
bad test
	def test_run_algorithm_all_positions(self):
		expected_result = np.array([[[[ 46.7991893 , -87.19275988],
         [-53.49409115,   1.96922873],
         [ -6.22460003,   0.39400196],
         [  5.46429236,  -5.81931768],
         [ -2.95360666,  13.99349076]],

        [[ 14.58608934, -80.68934346],
         [-32.08856757,  28.48613077],
         [ 26.95631477, -34.03201005],
         [-14.72586541, -24.91144592],
         [ -5.183328  ,  22.8765549 ]],

        [[ -2.12735447, -18.10587412],
         [  2.29895146,  18.04251819],
         [-12.47866874, -34.12836263],
         [-11.28348576, -20.71524746],
         [  0.929358  ,  12.75981039]],

        [[ -5.13901806,  13.05869337],
         [ 16.79018072,  11.49726951],
         [-13.18657921,   2.80659838],
         [ 10.73708493,  -6.57741672],
         [ -0.62065283,  15.20950349]],

        [[ -6.18301226,  25.4980874 ],
         [  5.10938824,  -0.65200085],
         [ -2.77621843,   8.18044838],
         [ 16.07130018,  -0.48097431],
         [  0.87787585,  10.64818449]],

        [[ -2.39070364,  26.35909118],
         [ -1.38328562,  -6.95367481],
         [  1.59825788,  10.43856219],
         [  2.13562728,   8.30326266],
         [ -1.03406004,   1.62375638]],

        [[ -2.7435565 ,  21.52356464],
         [ -4.41216889,  -6.05341586],
         [ -0.68866038,   9.92197179],
         [ -5.53180636,  11.97436829],
         [ -2.35251729,  -1.99512403]],

        [[ -3.16658587,  21.13084371],
         [ -3.85445596,   1.81687177],
         [ -3.01128549,   8.50938161],
         [ -2.71403666,   8.46501967],
         [ -1.7403355 ,  -1.82105768]],

        [[ -3.3289108 ,  14.15040143],
         [ -1.26548909,   7.87043598],
         [ -3.79994892,   7.74786704],
         [ -1.57063861,   6.73428692],
         [ -1.43719973,   3.90102459]],

        [[ -3.01982347,   9.54331179],
         [ -1.03038617,  10.58393088],
         [ -2.4731585 ,   7.78272422],
         [ -2.89682232,   7.55356923],
         [ -2.17378244,   7.19540922]]]])


		self.swarm.run_algorithm()
		comparison = np.isclose(self.swarm.all_positions, expected_result)
		self.assertTrue(np.all(comparison))
'''
'''
bad test
	def test_run_algorithm_best_results(self):
		self.swarm.run_algorithm()
		self.assertTrue(np.all(np.isclose(self.swarm.best_position, np.array([-2.77621843,  8.18044838]))))
		self.assertTrue(self.swarm.best_f == 36.638364725123814)
		self.assertTrue(self.swarm.error == 36.638364725123814)
'''
'''
class TestParticle(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 5, 'k': 3, 'time_steps': 10, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "dim":2, "xmin":[-100,-100], "xmax":[100,100],\
			"param_is_integer":[False, False], "show_animation":False}
		np.random.seed(123)
		self.swarm = ob.Swarm(self.constants, self.fn_info)
		self.swarm.distribute_swarm()
		self.particle = self.swarm.particles[0]

	def test_set_initial_state(self):
		self.particle.set_initial_state([0,0], [1,1], [2,2])
		self.assertTrue(self.particle.pos == [0,0])
		self.assertTrue(self.particle.vel == [1,1])
		self.assertTrue(self.particle.p == [2,2])
		self.assertTrue(self.particle.g == [2,2])
		self.assertTrue(self.particle.informants == [])
'''
'''
bad test
	def test_communicate(self):
		self.particle.communicate()
		self.assertTrue(np.all(np.isclose(self.particle.g, np.array([ 3.92938371e+01, -4.27721330e+01,  2.51787835e+08]))))

	def test_random_confidence(self):
		c1c2 = self.particle.random_confidence()
		print(c1c2)
		c1_comparison = np.isclose(c1c2[0], np.array([0.78360109, 0.17538296]))
		c2_comparison = np.isclose(c1c2[1], np.array([0.02821952, 0.15539038]))
		self.assertTrue(np.all(c1_comparison))
		self.assertTrue(np.all(c2_comparison))
'''
'''
bad test
	def test_step(self):
		self.particle.step()
		self.assertTrue(np.all(np.isclose(self.particle.pos, [ 26.11438891, -23.52260765])))
		self.assertTrue(np.all(np.isclose(self.particle.vel, [-13.17944821,  19.24952536])))
		self.assertTrue(np.all(np.isclose(self.particle.p, [ 3.92938371e+01, -4.27721330e+01,  2.51787835e+08])))
		self.assertTrue(np.all(np.isclose(self.particle.g, [ 3.92938371e+01, -4.27721330e+01,  2.51787835e+08])))
'''