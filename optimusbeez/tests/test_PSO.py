from unittest import TestCase

import optimusbeez as ob
import numpy as np

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


	def test_optimize_constants_save_config_wrong_input_type(self):
		self.assertRaises(TypeError, self.experiment.optimize_constants, tests=10, tests_with_each_constants=2, \
			allowed_evaluations=10, allowed_deviation=5, \
			save_configuration=0, create_file="no")
	def test_optimize_constants_save_config_wrong_input_string(self):
		self.assertRaises(ValueError, self.experiment.optimize_constants, tests=10, tests_with_each_constants=2, \
			allowed_evaluations=10, allowed_deviation=5, \
			save_configuration="nooo", create_file="no")
	def test_optimize_constants_save_config_assigned_constants(self):
		self.experiment.optimize_constants(tests=10, tests_with_each_constants=2, allowed_evaluations=10,\
			allowed_deviation=5, save_configuration="yes", create_file="no")
		self.assertTrue(self.experiment.constants() == {'phi': 2.0907483129111455, 'N': 1, 'k': 1, 'time_steps': 1, 'repetitions': 5})

	def test_optimize_constants_create_file_wrong_input_type(self):
		self.assertRaises(TypeError, self.experiment.optimize_constants, tests=10, tests_with_each_constants=2, \
			allowed_evaluations=10, allowed_deviation=5, \
			save_configuration="no", create_file=0)
	def test_optimize_constants_create_file_wrong_input_string(self):
		self.assertRaises(ValueError, self.experiment.optimize_constants, tests=10, tests_with_each_constants=2, \
			allowed_evaluations=10, allowed_deviation=5, \
			save_configuration="no", create_file="nooo")


	def test_run_n_evaluations_too_small(self):
		self.assertRaises(ValueError, self.experiment.run, 2)
	def test_run_best_position(self):
		self.experiment.run(1000)
		comparison_best_position = np.isclose(self.experiment.best_position, [-3.41548961, 11.66760793])
		self.assertTrue(comparison_best_position.all())
		self.assertTrue(self.experiment.best_f == 19.496964077915194)
		self.assertTrue(self.experiment.error == 19.496964077915194)

class TestSwarm(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 5, 'k': 3, 'time_steps': 10, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "xmin":-100, "xmax":100, "show_animation":False}
		np.random.seed(123)
		self.swarm = ob.Swarm(self.constants, self.fn_info)
		self.swarm.distribute_swarm()

	def test_distribute_swarm_particles_number(self):
		self.swarm.distribute_swarm()
		self.assertTrue(len(self.swarm.particles) == 5)
	def test_distribute_swarm_particles_type(self):
		self.swarm.distribute_swarm()
		for particle in self.swarm.particles:
			self.assertTrue(isinstance(particle, ob.PSO.Particle))

	def test_random_informants_number_of_informants(self):
		self.swarm.random_informants()
		lengths = []
		for particle in self.swarm.particles:
			self.assertTrue(len(particle.informants) == 3)

	def test_get_parameters_output(self):
		parameters = self.swarm.get_parameters()
		comparison = np.isclose(parameters, np.array([-3.81361970e+00, -2.15764964e+01,  1.30489995e+05]))
		self.assertTrue(np.all(comparison))

	def test_run_algorithm_all_positions(self):
		expected_result = np.array([[[[ 1.63707200e+01, -8.71927599e+01],
         [-4.39814667e+01,  1.96922873e+00],
         [-3.09500989e+01,  3.94001959e-01],
         [ 1.84345570e+00, -5.81931768e+00],
         [-2.95360666e+00,  1.39934908e+01]],

        [[-8.44889795e-01, -8.06893435e+01],
         [ 5.84739069e-01,  2.84861308e+01],
         [ 1.01272114e+01, -3.40320100e+01],
         [-1.84239485e+01, -2.45183861e+01],
         [-5.18332800e+00,  2.28765549e+01]],

        [[-6.25906448e+00, -1.81058741e+01],
         [ 1.95161031e+01,  1.88464558e+01],
         [ 2.67250373e+01, -3.63105333e+01],
         [-1.15636491e+01, -2.21178509e+01],
         [-2.08805524e+00,  1.33719822e+01]],

        [[-2.84434813e+00,  1.36480008e+01],
         [ 9.02345269e+00,  2.01217110e+01],
         [ 1.54883857e+01, -6.87251990e+00],
         [ 1.16323932e+01, -1.03723395e+01],
         [ 1.37504130e-01,  1.47628985e+01]],

        [[-1.40947031e+00,  2.69911093e+01],
         [-3.18117522e+00,  2.11041127e+01],
         [-8.01034857e+00,  3.52421989e+00],
         [ 5.49781180e+00,  1.10507490e+01],
         [-7.50629940e-01,  2.13515582e+01]],

        [[-2.44935404e+00,  2.41993872e+01],
         [-8.29570516e+00,  1.92572126e+01],
         [-1.61134165e+01,  1.61890940e+01],
         [-1.26623024e-02,  1.26855140e+01],
         [-3.45046548e+00,  2.01911919e+01]],

        [[-3.48188858e+00,  1.97923871e+01],
         [-7.93456748e+00,  1.75646526e+01],
         [-1.32984091e+01,  1.24791830e+01],
         [-3.84169416e+00,  6.70468445e+00],
         [-5.24265697e+00,  2.03246408e+01]],

        [[-3.46088361e+00,  1.13342014e+01],
         [-5.65930991e+00,  1.77171616e+01],
         [-7.86497594e+00,  6.18068325e+00],
         [-4.73125421e+00,  8.59090824e+00],
         [-5.34190607e+00,  2.05830107e+01]],

        [[-3.45205724e+00,  7.78003701e+00],
         [-2.47730128e+00,  1.47743006e+01],
         [-3.79517298e+00,  4.07450123e+00],
         [-4.30566987e+00,  1.06194534e+01],
         [-4.51310897e+00,  1.74654840e+01]],

        [[-3.45196978e+00,  9.70935313e+00],
         [-1.47234342e+00,  1.16539046e+01],
         [-1.88925073e+00,  4.89304958e+00],
         [-3.95610401e+00,  1.20633395e+01],
         [-3.91284298e+00,  1.40990308e+01]]]])

		self.swarm.run_algorithm()
		comparison = np.isclose(self.swarm.all_positions, expected_result)
		self.assertTrue(np.all(comparison))

	def test_run_algorithm_best_results(self):
		self.swarm.run_algorithm()
		self.assertTrue(np.all(np.isclose(self.swarm.best_position, np.array([-3.46088361, 11.33420137]))))
		self.assertTrue(self.swarm.best_f == 61.31051109394555)
		self.assertTrue(self.swarm.error == 61.31051109394555)


class TestParticle(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 5, 'k': 3, 'time_steps': 10, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "xmin":-100, "xmax":100, "show_animation":False}
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

	def test_communicate(self):
		self.particle.communicate()
		self.assertTrue(np.all(np.isclose(self.particle.g, np.array([ 3.92938371e+01, -4.27721330e+01,  2.51787835e+08]))))

	def test_random_confidence(self):
		c1c2 = self.particle.random_confidence()
		print(c1c2)
		c1_comparison = np.isclose(c1c2[0], np.array([0.78360109, 0.02821952]))
		c2_comparison = np.isclose(c1c2[1], np.array([0.17538296, 0.15539038]))
		self.assertTrue(np.all(c1_comparison))
		self.assertTrue(np.all(c2_comparison))

	def test_step(self):
		self.particle.step()
		self.assertTrue(np.all(np.isclose(self.particle.pos, [ 26.11438891, -23.52260765])))
		self.assertTrue(np.all(np.isclose(self.particle.vel, [-13.17944821,  19.24952536])))
		self.assertTrue(np.all(np.isclose(self.particle.p, [ 3.92938371e+01, -4.27721330e+01,  2.51787835e+08])))
		self.assertTrue(np.all(np.isclose(self.particle.g, [ 3.92938371e+01, -4.27721330e+01,  2.51787835e+08])))