# Particle Swarm Optimization
# This script finds the global MINIMUM of the
# selected function.

# This is the simplest version, PSO(0) from
# the book "Particle Swarm Optimization" by
# Maurice Clerc.

###################################################################

# Import required modules
import numpy as np 
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pkgutil
import io
import sys
from tqdm import tqdm
from .evaluate import Rosenbrock
from .evaluate import Alpine
from .evaluate import Griewank

# Set random seed
# np.random.seed(123)

###################################################################

# Helper functions

# Read and return dictionary from txt file
def read_dictionary_from_file(filename):
	try:
		data = pkgutil.get_data('optimusbeez', filename)
	except:
		raise NameError(f"File '{filename}' does not exist in 'optimusbeez' directory")
	dictionary = eval(data)
	if type(dictionary) == dict:
			return dictionary
	else:
		raise TypeError(f"{dictionary} is not a dictionary.")

def write_dictionary_to_file(dictionary, filepath):
	if type(dictionary) != dict:
		raise TypeError(f"Invalid type {type(dictionary)} for dictionary")
	try:
		file = open(filepath, "w")
		file.write(str(dictionary))
		file.close()
	except:
		raise NameError(f"Invalid path: {filepath}. Example path: /home/username/.../filename.txt")

# Determine error
def determine_error(found_value, minimum_value=0):
	return abs(found_value - minimum_value)

def n_evaluations(N, time_steps, repetitions):
	n_evaluations = N*time_steps*repetitions + repetitions*N
	if type(N) == np.ndarray:
		return (n_evaluations).astype(int)
	else:
		return n_evaluations

def generate_random_constants(allowed_evaluations, allowed_deviation):
	if type(allowed_evaluations) != int and type(allowed_deviation) != int:
		raise TypeError(f"Invalid types {type(allowed_evaluations)} and {type(allowed_deviation)} for allowed_evaluations and allowed_deviation.")
	if allowed_evaluations <= allowed_deviation:
		raise ValueError(f"Allowed deviation cannot be larger than or equal to allowed evaluations")
	if allowed_evaluations < 2:
		raise ValueError(f"Allowed evaluations cannot be less than 2")
	if allowed_deviation < 0:
		raise ValueError(f"Allowed deviation cannot be less than 0")

	max_n_evaluations = allowed_evaluations + allowed_deviation

	# Set minimum and maximum values for search
	constants_min = np.array([2.0001,1,1,1,1])
	constants_max = np.array([3,None,max_n_evaluations,max_n_evaluations,max_n_evaluations])

	# Initiate empty constants array
	constants = np.inf*np.ones(5)
	# Set phi, which does not depend on other values
	constants[0] = np.random.uniform(constants_min[0], constants_max[0])
	# n_evaluations = Ntr + Nr = Nr(t+1)
	# Choose N and r randomly from a geometric distribution
	while True:
		Nr = np.random.geometric(0.05, 2)
		# Check that it is possible to be below max_n_evaluations with this Nr
		if max_n_evaluations/(Nr[0]*Nr[1]) > 2:
			constants[2] = Nr[0]
			constants[4] = Nr[1]
			break
		else:
			continue
	# Choose t uniformly
	max_time_steps = max_n_evaluations/(constants[2]*constants[4])
	constants[3] = np.random.randint(1,max_time_steps)

	# Set k, which cannot be greater than N
	constants[1] = np.random.randint(constants_min[1], constants[2]+1)

	# print(f"Generated random constants: {constants}")
	return constants

###################################################################

# The training session class is inherited by particles and swarms
# This class sets the required parameters from constants and fn_info
class Experiment:
	def __init__(self, constants=None, fn_info=None):
		if np.all(constants == None):
			constants = read_dictionary_from_file('optimal_constants.txt')
		if np.all(fn_info == None):
			fn_info = read_dictionary_from_file('fn_info.txt')

		if type(constants) == dict and type(fn_info) == dict:
			self.N = constants["N"]
			self.time_steps = constants["time_steps"]
			self.repetitions = constants["repetitions"]
			self.fn_name = fn_info["fn_name"]
			self.optimal_f = fn_info["optimal_f"]
			self.dim = fn_info["dim"]
			self.k = constants["k"]
			self.phi = constants["phi"]
			self.xmin = np.array(fn_info["xmin"])
			self.xmax = np.array(fn_info["xmax"])
			self.param_is_integer = np.array(fn_info["param_is_integer"])
			self.show_animation = fn_info["show_animation"]
			self.special_constraints = fn_info["special_constraints"]
			self.constraints_function = fn_info["constraints_function"]
			self.constraints_extra_arguments = fn_info["constraints_extra_arguments"]
			self.disable_progress_bar = fn_info["disable_progress_bar"]
			self.get_parameters_from = fn_info["get_parameters_from"]

			# Calculate maximum velocity
			self.vmax = np.absolute(self.xmax - self.xmin)/2

			# Calculate confidence parameters using phi
			self.c1 = 1/(self.phi-1+np.sqrt(self.phi**2-2*self.phi))
			self.cmax = self.c1*self.phi

		else:
			raise TypeError(f"Invalid types {type(constants)} and {type(fn_info)} for constants and fn_info.")

	# Return dictionary of current constants if argument 'dictionary' is not given
	# Update current constants if 'dictionary' is given and return the given dictionary
	def constants(self, dictionary=None):
		if dictionary == None:
			constants = {'phi': self.phi, 'N': self.N, 'k': self.k, 
				'time_steps': self.time_steps, 'repetitions': self.repetitions}
			return constants
		elif type(dictionary) == dict:
			constants = dictionary
			self.phi = constants["phi"]
			self.N = constants["N"]
			self.k = constants["k"]
			self.time_steps = constants["time_steps"]
			self.repetitions = constants["repetitions"]
			return constants
		else:
			raise TypeError(f"Invalid type {type(dictionary)} for dictionary")

	def fn_info(self, dictionary=None):
		if dictionary == None:
			fn_info = {"fn_name":self.fn_name, "optimal_f":self.optimal_f, "dim":self.dim,
				"xmin":self.xmin.tolist(), "xmax":self.xmax.tolist(), 
				"param_is_integer":self.param_is_integer.tolist(),
				"special_constraints":self.special_constraints,
				"constraints_function":self.constraints_function,
				"constraints_extra_arguments":self.constraints_extra_arguments,
				"show_animation":self.show_animation,
				"disable_progress_bar":self.disable_progress_bar,
				"get_parameters_from": self.get_parameters_from}
			return fn_info
		elif type(dictionary) == dict:
			fn_info = dictionary
			return fn_info
		else:
			raise TypeError(f"Invalid type {type(dictionary)} for dictionary")

	def n_evaluations(self):
		return n_evaluations(self.N, self.time_steps, self.repetitions)

	def run(self, max_n_evaluations=None):
		if max_n_evaluations == None:
			max_n_evaluations = n_evaluations(self.N, self.time_steps, self.repetitions)
		else:
			if max_n_evaluations <= 2*self.N*self.repetitions:
				raise ValueError(f"Number of evaluations must be greater than 2Nr. In this case >= {2*self.N*self.repetitions}")
		
		self.time_steps = math.ceil((max_n_evaluations - self.repetitions*self.N)/(self.repetitions*self.N))
		print("Running algorithm...")

		constants = self.constants()
		fn_info = self.fn_info()

		self.swarm = Swarm(constants, fn_info)
		self.swarm.distribute_swarm()
		self.swarm.run_algorithm()
		true_n_evaluations = n_evaluations(self.swarm.N, self.swarm.time_steps, self.swarm.repetitions)

		self.best_position = self.swarm.best_position
		self.best_f = self.swarm.best_f
		self.error = self.swarm.error

		print(f"{true_n_evaluations} evaluations made.")
		print(f"The best position is {repr(self.best_position.tolist())}.")
		print(f"The value at this position is {self.best_f}")
		print(f"Error in value: {self.error}")

		if self.show_animation == False:
			pass
		else:
			self.swarm.simulate_swarm()

###################################################################

class Swarm(Experiment):
	def random_initial_positions(self):
		initial_positions = np.inf*np.ones((self.N, self.dim))
		# Check if there are any special constraints
		if self.special_constraints == False:
			# Create array of initial positions and velocities
			# taking into account that some parameters must be integers
			for d in range(self.dim):
				if self.param_is_integer[d] == True:
					initial_positions[:,d] = np.random.randint(self.xmin[d], self.xmax[d], self.N)
				elif self.param_is_integer[d] == False:
					initial_positions[:,d] = np.random.uniform(self.xmin[d], self.xmax[d], self.N)
			# Note that these positions are all of type np.float64 even though randint is called
		else:
			# This should be modifiable automatically
			for particle in range(self.N):
				initial_positions[particle] = generate_random_constants(
					self.constraints_extra_arguments[0],
					self.constraints_extra_arguments[1])
		return initial_positions

	def random_initial_velocities(self):
		return np.random.uniform(-self.vmax, self.vmax, (self.N, self.dim))

	def create_particles(self, initial_positions, initial_velocities):
		# Create array of initial p-values by evaluating initial positions
		p_values = np.inf*np.ones((self.N, self.dim+1))
		for i, pos in enumerate(initial_positions):
			p_values[i,self.dim] = eval(self.fn_name)(pos)
			p_values[i,0:self.dim] = pos

		constants = self.constants()
		fn_info = self.fn_info()

		# Create list of particle objects
		self.particles = []
		for i in range(self.N):
			pos = initial_positions[i]
			vel = initial_velocities[i]
			p = p_values[i]
			particle = Particle(constants, fn_info)
			particle.set_initial_state(pos, vel, p)
			self.particles.append(particle)

	# Choose k informants randomly
	def random_informants(self):
		for particle in self.particles:
			particle.informants = np.random.choice(self.particles, particle.k)

	def distribute_swarm(self):
		# Distribute swarm over search space

		# Create array of initial positions and velocities
		initial_positions = self.random_initial_positions()
		initial_velocities = self.random_initial_velocities()

		self.create_particles(initial_positions, initial_velocities)

		# Initiate k informants randomly
		self.random_informants()

		# Initialise array of positions for animation
		self.positions = np.inf*np.ones((self.time_steps, self.N, self.dim))
		self.positions[0,:,:] = initial_positions

	# Update positions of particles for all time steps
	def evolve(self):
		# With progress bar
		# Evolve swarm for all time steps
		for time_step in tqdm(range(self.time_steps),
			desc=f"Repetition {self.current_repetition}/{self.repetitions}: Evolving swarm",
			disable=self.disable_progress_bar):
			for i, particle in enumerate(self.particles):
				particle.step()
				# Update positions for animation
				self.positions[time_step,i,:] = particle.pos
			# Select informants for next time step
			self.random_informants()


	# Extract optimal parameters
	def get_parameters(self):
		if self.get_parameters_from == "g-values":
			final_g = np.inf*np.ones((self.N, self.dim+1))
			for i,particle in enumerate(self.particles):
				final_g[i,:] = particle.g
			optimal_i = np.argmin(final_g[:,self.dim])
			result = final_g[optimal_i]
		if self.get_parameters_from == "average p-values":
			final_p = np.inf*np.ones((self.N, self.dim+1))
			for i,particle in enumerate(self.particles):
				final_p[i,:] = particle.p 
			result = np.average(final_p, axis=0)
		return result

	# Run the algorithm for required number of repetitions
	# Return best found position, value, and error
	def run_algorithm(self):

		# results contains the best found positions and values for each repetition
		results = np.inf*np.ones((self.repetitions, self.dim+1))
		# all_positions contains all the visited positions for each repetition
		# all_positions is used to create an animation of the swarm
		self.all_positions = np.inf*np.ones((self.repetitions, self.time_steps, self.N, self.dim))

		for r in range(self.repetitions):
			self.current_repetition = r+1
			self.distribute_swarm()
			self.evolve()
			result = self.get_parameters()
			results[r] = result
			self.all_positions[r] = self.positions

		self.best_value_index = np.argmin(results[:,self.dim])

		self.best_position = results[self.best_value_index][0:self.dim]
		self.best_f = results[self.best_value_index][self.dim]
		self.error = determine_error(self.best_f, self.optimal_f)

	def simulate_swarm(self):
		# If dim > 2, then only first 2 axes will be silmulated
		# Plot initial positions of particles
		#Testing: changed axes to plot N and r values
		fig, ax = plt.subplots()
		ax.set_xlim(self.xmin[2], self.xmax[2])
		ax.set_ylim(self.xmin[4], self.xmax[4])
		scat = ax.scatter(self.all_positions[self.best_value_index,0,:,2], 
			self.all_positions[self.best_value_index,0,:,4], color="Black", s=2)

		# Create animation
		interval = 200_000 / (self.N * self.time_steps * self.repetitions)
		self.animation = FuncAnimation(fig, func=self.update_frames, interval=interval, 
			fargs=[scat, self.all_positions, self.best_value_index])
		plt.show()

	# Required update function for simulation
	def update_frames(self, j, *fargs):
		scat, all_positions, best_value_index = fargs
		try:
			scat.set_offsets(all_positions[best_value_index,j,:,2:5:2])
		except:
			print("Simulation finished")
			self.animation.event_source.stop()

###################################################################

class Particle(Experiment):

	def set_initial_state(self, pos, vel, p):
		# Initializes particle with assigned
		# position and velocity
		self.pos = pos
		self.vel = vel
		# Set initial best found value by particle
		# format: np array of shape (1, 3) - x, y, value
		self.p = p

		# Best found position and value by informants
		# format: np array of shape (1, 3) - x, y, value
		self.g = p

		# Empty list of informants
		self.informants = []

	def communicate(self):
		# Communicate with informants
		# Receive best positions with values from informants
		received = np.zeros((self.k, self.dim+1))
		for i, informant in enumerate(self.informants):
			received[i, :] = informant.g
		# Find best g from communicated values
		i = np.argmin(received[:,self.dim])
		best_received_g = received[i]
		# Set g to LOWEST value
		if best_received_g[-1] < self.g[-1]:
			self.g = best_received_g

	# Randomly assign confidence parameters
	# c2 and c3 in the interval [0, cmax)
	def random_confidence(self):
		c2 = np.inf*np.ones(self.dim)
		c3 = np.inf*np.ones(self.dim)

		for d in range(self.dim):
			c2[d] = np.random.uniform(0, self.cmax)
			c3[d] = np.random.uniform(0, self.cmax)
		return (c2, c3)

	def update_p(self, value):
		# Update p if current position is LOWER
		if value < self.p[self.dim]:
			self.p[self.dim] = value
			self.p[0:self.dim] = self.pos

	def update_g(self, value):
		if value < self.g[self.dim]:
			self.g[self.dim] = value
			self.g[0:self.dim] = self.pos
		self.communicate()

	def find_vel(self):
		c2, c3 = self.random_confidence()		
		possible_vel = self.c1*self.vel + \
			c2*(self.p[0:self.dim] - self.pos) + \
			c3*(self.g[0:self.dim] - self.pos)	

		# Constrain velocity
		smaller_than_vmax = possible_vel < self.vmax
		greater_than_neg_vmax = possible_vel > -self.vmax
		vel_comparison = np.zeros((len(self.vmax), 2))
		vel_comparison[:,0] = smaller_than_vmax
		vel_comparison[:,1] = greater_than_neg_vmax
		vel_comparison = np.all(vel_comparison, axis=1)
		self.vel = np.where(vel_comparison, possible_vel, self.vel)

	def set_pos(self):
		if self.special_constraints == True:
			next_pos, vel = eval(self.constraints_function)(self, self.constraints_extra_arguments)
			self.pos = next_pos
			self.vel = vel
		else:
			possible_pos = self.pos + self.vel
			in_search_area = self.is_in_search_area(possible_pos)
			self.pos = np.where(in_search_area, possible_pos, self.pos)
			self.vel = np.where(in_search_area, self.vel, 0)

	def is_in_search_area(self, possible_pos):
		smaller_than_xmax = possible_pos <= self.xmax
		greater_than_xmin = possible_pos >= self.xmin
		is_allowed = np.zeros((len(self.xmax), 2))
		is_allowed[:,0] = smaller_than_xmax
		is_allowed[:,1] = greater_than_xmin
		is_allowed = np.all(is_allowed, axis=1)

		return is_allowed

	def step(self):
		value = eval(self.fn_name)(self.pos)
		self.update_p(value)
		self.update_g(value)
		self.find_vel()
		self.set_pos()





###################################################################
def optimize_constants(allowed_evaluations=500, allowed_deviation=20, optimization_time_steps=100,
	optimization_repetitions=5):
	# Note that this function optimizes for the function and search space given in
	# the file fn_info.txt
	# If you change this file, make sure to set show_animation to False

	optimal_experiment_constants = {'phi': 2.4, 'k':3, 'N': 15, 'time_steps': optimization_time_steps,
		'repetitions':optimization_repetitions}

	phi_min = 2.00001
	phi_max = 3
	k_min = 1
	N_min = 1
	N_max = 30
	time_steps_min = 1
	time_steps_max = allowed_evaluations + allowed_deviation
	repetitions_min = 1
	repetitions_max = 30

	xmin = [phi_min, k_min, N_min, time_steps_min, repetitions_min]
	xmax = [phi_max, N_max, N_max, time_steps_max, repetitions_max]

	param_is_integer = [False, True, True, True, True]
	evaluation_function_name = "constant_optimization_evaluation_function"	

	optimal_experiment_fn_info = {"fn_name":evaluation_function_name, "optimal_f":0, "dim":5,\
		 "xmin":xmin, "xmax":xmax, "param_is_integer":param_is_integer,\
		 "special_constraints":True, "constraints_function":"Ntr_constrain_next_position",\
		 "constraints_extra_arguments":[allowed_evaluations,allowed_deviation], "show_animation":True,
		 "disable_progress_bar":False, "get_parameters_from":"average p-values"}

	optimal_experiment = Experiment(optimal_experiment_constants, optimal_experiment_fn_info)

	# Optimize the experiment
	optimal_experiment.run()

	optimal_constants = position_to_constants_dictionary(optimal_experiment.best_position)

	# Restore printing
	sys.stdout = sys.__stdout__

	print(f"Constants optimization finished.")
	print(f"The best found constants configuration is {optimal_constants}")
	print(f"This configuration has the error {optimal_experiment.best_f}")

	return optimal_constants

def constant_optimization_evaluation_function(new_constants):
	# Suppress printing
	text_trap = io.StringIO()
	sys.stdout = text_trap

	constants = position_to_constants_dictionary(new_constants)
	# Function info is default from function_info.txt
	experiment = Experiment(constants=constants)
	experiment.disable_progress_bar = True
	experiment.run()

	return experiment.error

def Ntr_constrain_next_position(particle, extra_arguments):
	# Constrain N, time_steps, repetitions so the number of evaluations
	# does not exceed the maximum_n_evaluations argument

	allowed_evaluations = extra_arguments[0]
	allowed_deviation = extra_arguments[1]
	max_n_evaluations = allowed_evaluations + allowed_deviation

	previous_pos = particle.pos
	previous_pos_n_evaluations = n_evaluations(previous_pos[2], previous_pos[3], previous_pos[4])
	if previous_pos_n_evaluations > max_n_evaluations:
		raise ValueError(f"The previous position is invalid given the special constraints.")
	
	possible_pos = previous_pos + particle.vel
	vel = particle.vel

	# Constrain all values to rectangular search area
	in_search_area = particle.is_in_search_area(possible_pos)
	possible_pos = np.where(in_search_area, possible_pos, particle.pos)
	vel = np.where(in_search_area, particle.vel, 0)

	# Further constrain N, time_steps and repetitions according to allowed number of evaluations
	while True:
		one_of_Ntr = np.random.randint(2,5)
		possible_pos[one_of_Ntr] = previous_pos[one_of_Ntr]
		vel[one_of_Ntr] = 0
		possible_n_evaluations = n_evaluations(math.ceil(possible_pos[2]), math.ceil(possible_pos[3]),\
			math.ceil(possible_pos[4]))
		if possible_n_evaluations <= max_n_evaluations and possible_n_evaluations >= 2:
			new_pos = possible_pos
			break
		else:
			continue

	# Further constrain k to be less than or equal to N
	if new_pos[1] > new_pos[2]:
		new_pos[1] = new_pos[2]

	# Convert required constants to integers
	for constant in range(1,5):
		new_pos[constant] = math.ceil(new_pos[constant])

	return new_pos, vel

def position_to_constants_dictionary(position):
	# This helper function turns a 5D position into the constants dictionary
	# Used for constants optimization
	if len(position) != 5:
		raise ValueError(f"The position {position} cannot be turned into a constants dictionary")
	constants = {}
	constants["phi"] = position[0]
	constants["k"] = int(position[1])
	constants["N"] = int(position[2])
	constants["time_steps"] = int(position[3])
	constants["repetitions"] = int(position[4])

	return constants

###################################################################


if __name__ == "__main__":
	experiment = Experiment()
	experiment.run()