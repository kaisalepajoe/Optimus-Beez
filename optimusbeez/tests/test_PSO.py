from unittest import TestCase

import optimusbeez as ob
import numpy as np

class TestRegularExperiment(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 5, 'k': 3, 'time_steps': 10, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "dim":2, "xmin":[-100,-100], "xmax":[100,100],\
		"param_is_integer":[False, False], "special_constraints":False, "constraints_extra_arguments":"an argument",\
		"constraints_function":None, "disable_progress_bar":False, "show_animation":False, "get_parameters_from":'g-values'}
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
		self.assertRaises(TypeError, self.experiment.n_evaluations())
	def test_n_evaluations_without_input(self):
		self.assertTrue(self.experiment.n_evaluations() == 55)
	
	def test_run_n_evaluations_too_small(self):
		self.assertRaises(ValueError, self.experiment.run, 2)


class TestSwarmSetup(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 5, 'k': 3, 'time_steps': 10, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "dim":4, "xmin":[-10,-10,-10,-10], "xmax":[10,10,10,10],\
			"param_is_integer":[False, False, True, True], "special_constraints":False,
			"constraints_function":None,"constraints_extra_arguments":[], "show_animation":True,
			"disable_progress_bar":False,"show_animation":False, "get_parameters_from":'g-values'}
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

	def test_random_informants_number_of_informants(self):
		particles = []
		for i in range(6):
			particles.append(ob.Particle())
		self.swarm.particles = particles
		self.swarm.random_informants()
		for particle in self.swarm.particles:
			self.assertTrue(len(particle.informants) == 3)

class TestSwarmEvolution(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 2, 'k': 2, 'time_steps': 3, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "dim":4, "xmin":[-10,-10,-10,-10], "xmax":[10,10,10,10],\
			"param_is_integer":[False, False, False, False], "special_constraints":False,
			"constraints_function":None,"constraints_extra_arguments":[], "show_animation":True,
			"disable_progress_bar":False,"show_animation":False, "get_parameters_from":'g-values'}
		np.random.seed(123)
		self.swarm = ob.Swarm(self.constants, self.fn_info)
		initial_positions = np.array([[1,1,1,1], [2,2,2,2]])
		initial_velocities = np.array([[1,1,1,1], [2,2,2,2]])
		self.swarm.create_particles(initial_positions, initial_velocities)

	def test_create_particles_N(self):
			self.assertTrue(len(self.swarm.particles) == 2)

	def test_get_parameters_output_g_values(self):
		for particle in self.swarm.particles:
			particle.step()
		result = self.swarm.get_parameters()
		self.assertTrue(result[-1] == 0)
		self.assertTrue(np.all(result[0:4] == [1,1,1,1]))

	def test_get_parameters_output_average_p_values(self):
		for particle in self.swarm.particles:
			particle.step()
		self.swarm.get_parameters_from = 'average p-values'
		result = self.swarm.get_parameters()
		self.assertTrue(result[-1] == 601.5)
		self.assertTrue(np.all(result[0:4] == [1.5, 1.5, 1.5, 1.5]))

class TestParticle(TestCase):
	def setUp(self):
		self.constants = {'phi': 2.4, 'N': 2, 'k': 2, 'time_steps': 3, 'repetitions': 1}
		self.fn_info = {"fn_name":"Rosenbrock", "optimal_f":0, "dim":2, "xmin":[-10,-10], "xmax":[10,10],\
			"param_is_integer":[False, False], "special_constraints":False,
			"constraints_function":None,"constraints_extra_arguments":[], "show_animation":True,
			"disable_progress_bar":False,"show_animation":False, "get_parameters_from":'g-values'}
		np.random.seed(123)
		self.particle1 = ob.Particle(self.constants, self.fn_info)
		self.particle2 = ob.Particle(self.constants, self.fn_info)
		self.particle1_informants = [self.particle1, self.particle2]

	def test_set_initial_state(self):
		self.particle1.set_initial_state([0,0], [1,1], [2,2])
		self.assertTrue(self.particle1.pos == [0,0])
		self.assertTrue(self.particle1.vel == [1,1])
		self.assertTrue(self.particle1.p == [2,2])
		self.assertTrue(self.particle1.g == [2,2])
		self.assertTrue(self.particle1.informants == [])

	def test_communicate(self):
		self.particle1.set_initial_state([1,1], [1,1], [3,3,300])
		self.particle2.set_initial_state([2,2], [2,2], [4,4,400])
		self.particle1.g = [5,5,500]
		self.particle2.g = [0,0,0]
		self.particle1.informants = [self.particle1, self.particle2]
		self.particle1.communicate()
		self.assertTrue(np.all(self.particle1.g == [0,0,0]))
		self.particle2.communicate()
		self.assertTrue(np.all(self.particle2.g == [0,0,0]))
	
	def test_random_confidence(self):
		c2c3 = self.particle1.random_confidence()
		print(c2c3)
		c2_comparison = np.isclose(c2c3[0], np.array([0.7023821 , 0.22877739]))
		c3_comparison = np.isclose(c2c3[1], np.array([0.28856861, 0.55599535]))
		self.assertTrue(np.all(c2_comparison))
		self.assertTrue(np.all(c3_comparison))

	def test_vel(self):
		self.particle1.set_initial_state(np.array([1,1]), np.array([1,1]), np.array([3,3,300]))
		self.particle1.g = np.array([2,2,200])
		self.particle1.find_vel()
		print(self.particle1.c1)
		print(self.particle1.vel)
		self.assertTrue(np.all(np.isclose(self.particle1.vel, [2.1135369128867287, 1.4337542328867288])))

	def test_vel_constraining_positive(self):
		self.particle1.set_initial_state(np.array([1,1]), np.array([100,1]), np.array([3,3,300]))		
		self.particle1.g = np.array([2,2,200])
		self.particle1.find_vel()
		print(self.particle1.vmax)
		print(self.particle1.c1)
		print(self.particle1.vel)
		self.assertTrue(np.all(np.isclose(self.particle1.vel, [10, 1.4337542328867288])))	

	def test_vel_constraining_negative(self):
		self.particle1.set_initial_state(np.array([1,1]), np.array([1,-100]), np.array([3,3,300]))		
		self.particle1.g = np.array([2,2,200])
		self.particle1.find_vel()
		print(self.particle1.vmax)
		print(self.particle1.c1)
		print(self.particle1.vel)
		self.assertTrue(np.all(np.isclose(self.particle1.vel, [2.1135369128867287, -10])))	


	def test_set_pos_no_constraints(self):
		self.particle1.pos = np.array([1,1])
		self.particle1.vel = np.array([1,1])
		self.particle1.set_pos()
		self.assertTrue(np.all(self.particle1.pos == [2,2]))

	def test_set_pos_with_constraints(self):
		self.particle1.special_constraints = True
		self.particle1.constraints_function = "Ntr_constrain_position"
		self.particle1.dim = 5
		self.particle1.xmin = [-10,-10,-10,-10,-10]
		self.particle1.xmax = [10,10,10,10,10]
		self.particle1.pos = np.array([2.4, 2, 2, 1, 1])
		self.particle1.vel = np.array([0.2,2,2,2,0])
		self.particle1.constraints_extra_arguments = [True, 10, 8]
		self.particle1.set_pos()
		print(self.particle1.pos)
		self.assertTrue(np.all(self.particle1.pos == [2.6, 4, 4, 3, 1]))
		self.assertTrue(np.all(self.particle1.vel == [0.2,2,2,2,0]))

	def test_set_pos_outside_search_area(self):
		self.particle1.pos = np.array([1,1])
		self.particle1.vel = np.array([100,-100])
		self.particle1.set_pos()
		self.assertTrue(np.all(self.particle1.pos == [1,1]))

	def test_is_in_search_area(self):
		self.particle1.xmin = np.array([3,3,3])
		self.particle1.xmax = np.array([6,6,6])
		booleans = self.particle1.is_in_search_area(np.array([1,4,9]))
		self.assertTrue(np.all(booleans == [False, True, False]))

