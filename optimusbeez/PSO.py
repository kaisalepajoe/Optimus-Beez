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

	# Set minimum and maximum values for search
	phi_min = 2.00001
	phi_max = 3
	k_min = 1
	N_min = 1
	N_max = 30
	time_steps_min = 1
	time_steps_max = allowed_evaluations + allowed_deviation
	repetitions_min = 1
	repetitions_max = 30

	# Initiate empty dictionary
	constants = np.inf*np.ones(5)

	# Set N-t-r grid size
	NTR = np.ones((N_max-N_min+1, time_steps_max-time_steps_min+1, \
		repetitions_max-repetitions_min+1))
	# Populate grid with total time steps
	for n in range(len(NTR)):
		for t in range(len(NTR[n])):
			for r in range(len(NTR[n, t])):
				NTR[n,t,r] = n_evaluations(N=n+N_min,
					time_steps=t+time_steps_min,
					repetitions=r+repetitions_min)
	valid_NTR_choices = np.where((NTR >= (allowed_evaluations - allowed_deviation)) \
		& (NTR <= (allowed_evaluations + allowed_deviation)))
	valid_NTR_choices = np.array([valid_NTR_choices[0], valid_NTR_choices[1], valid_NTR_choices[2]])
	# valid_NTR_choices contains the indices that correspond to parameters
	# with the allowed total number of time steps

	# Set N, time_steps, repetitions
	n, t, r = valid_NTR_choices[:,np.random.randint(0,valid_NTR_choices.shape[1])]
	N = n + N_min
	time_steps = t + time_steps_min
	repetitions = r + repetitions_min

	# Set parameters
	constants[0] = np.random.uniform(phi_min, phi_max)
	constants[1] = np.random.randint(k_min, N+1)
	constants[2] = N
	constants[3] = time_steps
	constants[4] = repetitions

	print(f"Generated random constants: {constants}")
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
				"show_animation":self.show_animation}
			return fn_info
		elif type(dictionary) == dict:
			fn_info = dictionary
			return fn_info
		else:
			raise TypeError(f"Invalid type {type(dictionary)} for dictionary")

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
		print(f"The best position is {self.best_position}.")
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
		for time_step in range(self.time_steps):
			for i, particle in enumerate(self.particles):
				particle.step()
				# Update positions for animation
				self.positions[time_step,i,:] = particle.pos
			# Select informants for next time step
			self.random_informants()

	# Extract optimal parameters (from g-values)
	def get_parameters(self):
		final_g = np.inf*np.ones((self.N, self.dim+1))
		for i,particle in enumerate(self.particles):
			final_g[i,:] = particle.g
		optimal_i = np.argmin(final_g[:,self.dim])
		result = final_g[optimal_i]
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
			print(f"{r+1}/{self.repetitions}")
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
		fig, ax = plt.subplots()
		ax.set_xlim(self.xmin[0], self.xmax[0])
		ax.set_ylim(self.xmin[1], self.xmax[1])
		scat = ax.scatter(self.all_positions[self.best_value_index,0,:,0], 
			self.all_positions[self.best_value_index,0,:,1], color="Black", s=2)

		# Create animation
		interval = 200_000 / (self.N * self.time_steps * self.repetitions)
		self.animation = FuncAnimation(fig, func=self.update_frames, interval=interval, 
			fargs=[scat, self.all_positions, self.best_value_index])
		plt.show()

	# Required update function for simulation
	def update_frames(self, j, *fargs):
		scat, all_positions, best_value_index = fargs
		try:
			scat.set_offsets(all_positions[best_value_index,j,:,0:2])
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
		# Set g to LOWEST value
		i = np.argmin(received[:,self.dim])
		self.g = received[i]

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
def optimize_constants(allowed_evaluations=500, allowed_deviation=20, optimization_iterations=30):
	# Note that this function optimizes for the function and search space given in
	# the file fn_info.txt
	# If you change this file, make sure to set show_animation to False

	optimal_experiment_constants = {'phi': 2.4, 'k':3, 'N': 15, 'time_steps': optimization_iterations,
		'repetitions': 1}

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
		 "constraints_extra_arguments":[allowed_evaluations,allowed_deviation], "show_animation":True}

	optimal_experiment = Experiment(optimal_experiment_constants, optimal_experiment_fn_info)

	# Optimize the experiment
	optimal_experiment.run()

	optimal_constants = position_to_constants_dictionary(optimal_experiment.best_position)

	print(f"Constants optimization finished.")
	print(f"The best found constants configuration is {optimal_constants}")
	print(f"This configuration has the error {optimal_experiment.best_f}")

	return optimal_constants

def constant_optimization_evaluation_function(new_constants):
	constants = {}
	constants["phi"] = new_constants[0]
	constants["k"] = math.ceil(new_constants[1])
	constants["N"] = math.ceil(new_constants[2])
	constants["time_steps"] = math.ceil(new_constants[3])
	constants["repetitions"] = math.ceil(new_constants[4])
	# Function info is default from function_info.txt
	experiment = Experiment(constants=constants)
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
	constants["k"] = position[1]
	constants["N"] = position[2]
	constants["time_steps"] = position[3]
	constants["repetitions"] = position[4]

	return constants

###################################################################


if __name__ == "__main__":
	experiment = Experiment()
	experiment.run()