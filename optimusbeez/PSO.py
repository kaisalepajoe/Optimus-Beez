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

def read_dictionary_from_file(filename):
	"""
	Read and return a dictionary from a txt file.

	Parameters
	----------
	filename : str
	The name of the file to be read.
	This file must be located in the 'optimusbeez' directory.

	Returns
	-------
	dictionary : dict

	"""
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
	"""
	Write a dictionary to a txt file.

	Takes a dictionary and an absolute file path as arguments and 
	writes the dictionary to the specified file. Returns nothing.

	Parameters
	----------
	dictionary : dict
	filepath : str
	The filepath must be an absolute path to the file.
	Example: '/dir1/dir2/filename.txt'.

	Returns
	-------
	Nothing
	"""
	if type(dictionary) != dict:
		raise TypeError(f"Invalid type {type(dictionary)} for dictionary")
	try:
		file = open(filepath, "w")
		file.write(str(dictionary))
		file.close()
	except:
		raise NameError(f"Invalid path: {filepath}. Example path: /home/username/.../filename.txt")

def determine_error(found_value, minimum_value=0):
	'''
	Returns the difference between two values.

	Parameters
	----------
	found_value : number
	A value found in the experiment.

	minimum_value : number
	The known value.

	Returns
	-------
	error : number
	The absolute difference between the found value and known value.
	'''
	error = abs(found_value - minimum_value)
	return error

def n_evaluations(N, time_steps, repetitions):
	'''
	Returns the number of evaluations of an experiment.

	Returns the number of evaluations for a given number of particles,
	time_steps, and repetitions.

	If given an array, does calculations on rows and returns the number of
	evaluations as an array.

	Parameters
	----------
	N : number or np.ndarray
	The number of particles in the swarm.

	time_steps : number or np.ndarray
	The number of time steps in each swarm evolution.
	
	repetitions : number or np.ndarray
	The number of times swarm evolutions are repeated.

	Returns
	-------
	n_evaluations : number or np.ndarray
	'''
	n_evaluations = N*time_steps*repetitions + repetitions*N
	if type(N) == np.ndarray or type(time_steps) == np.ndarray\
		or type(repetitions) == np.ndarray:
		return (n_evaluations).astype(int)
	else:
		return math.ceil(n_evaluations)

def generate_random_constants(allowed_evaluations, allowed_deviation):
	'''
	Generates an array of randomly selected constants.

	Constants are [phi, k, N, time_steps, repetitions] - the parameters of the PSO algorithm.

	Parameters
	----------
	allowed_evaluations : int
	The number of evaluations allowed for an experiment

	allowed_deviation : int
	The allowed deviation from the allowed number of evaluations. The resulting
	constants configuration will have a number of evaluations that is
	allowed_evaluations +- allowed_deviation.

	Returns
	-------
	constants : np.ndarray
	An array of random values for the constants - [phi, k, N, time_steps, repetitions]
	'''
	if type(allowed_evaluations) != int and type(allowed_deviation) != int:
		raise TypeError(f"Invalid types {type(allowed_evaluations)} and {type(allowed_deviation)} for allowed_evaluations and allowed_deviation.")
	if allowed_evaluations <= allowed_deviation:
		raise ValueError(f"Allowed deviation cannot be larger than or equal to allowed evaluations")
	if allowed_evaluations < 2:
		raise ValueError(f"Allowed evaluations cannot be less than 2")
	if allowed_deviation < 0:
		raise ValueError(f"Allowed deviation cannot be less than 0")

	allowed_n_evaluations = allowed_evaluations + allowed_deviation

	# Set minimum and maximum values for search
	constants_min = np.array([2.0001,1,1,1,1])
	constants_max = np.array([3,None,allowed_n_evaluations,allowed_n_evaluations,allowed_n_evaluations])

	# Initiate empty constants array
	constants = np.inf*np.ones(5)
	# Set phi, which does not depend on other values
	constants[0] = np.random.uniform(constants_min[0], constants_max[0])
	# n_evaluations = Ntr + Nr = Nr(t+1)
	# Choose N and r randomly from a geometric distribution
	while True:
		Nr = np.random.geometric(0.05, 2)
		# Check that it is possible to be below allowed_n_evaluations with this Nr
		if allowed_n_evaluations/(Nr[0]*Nr[1]) > 2:
			constants[2] = Nr[0]
			constants[4] = Nr[1]
			break
		else:
			continue
	# Choose t uniformly
	max_time_steps = allowed_n_evaluations/(constants[2]*constants[4])
	constants[3] = np.random.randint(1,max_time_steps)

	# Set k, which cannot be greater than N
	constants[1] = np.random.randint(constants_min[1], constants[2]+1)

	# print(f"Generated random constants: {constants}")
	return constants

###################################################################

class Experiment:
	'''
	This class sets the parameters from the constants and fn_info dictionaries
	as attributes of the Class. The class is inherited by both Particles and Swarms.

	The main method of this script is 'run' from the Experiment class.
	Create an experiment object: experiment = ob.Experiment()
	Run the PSO algorithm: experiment.run()
	'''
	def __init__(self, constants=None, fn_info=None):
		'''
		This function sets the parameters from the dictionaries 'constants' and 'fn_info'

		'constants' contains: phi, k, N, time_steps, repetitions
		'fn_info' contains: fn_name, optimal_f, dim, xmin, xmax, param_is_integer,
		special_constraints, constraints_function, constraints_extra_arguments,
		show_animation, disable_progress_bar, get_parameters_from.

		Parameters
		----------
		constants : dict
		The dictionary containing the following hyperparameters of the PSO algorithm.

			phi : number
			Phi sets the two confidence parameters c1 and cmax as described in the PSO book.
			
			k : int
			The number of informants for each particle.
			
			N : int
			The number of particles in the swarm.
			
			time_steps : int
			The number of time steps in each evolution of the swarm.
			
			repetitions : int
			The number of times a swarm is evolved before taking the average best position and value.
		
		fn_info : dict
		The dictionary containing the following information about the function to be evaluated.
			
			fn_name : str
			The name of the function to be evaluated by the PSO. For example, 'Rosenbrock'.
			
			optimal_f : number
			The value of the best position, used to calculate the error. For Rosenbrock, this is 0.
			
			dim : int
			The number of dimensions of the problem.
			
			xmin : list
			A list of minimum values for the search space. Each element corresponds to a dimension.
			
			xmax : list
			A list of maximum values for the search space. Each element corresponds to a dimension.
			
			param_is_integer : list of bools
			A list of booleans for each dimension. True means that the parameter can only be an integer.
			
			special_constraints : bool
			True if there are any special constraints. False if the only constraints are a rectangular
			search space given by xmin and xmax.
			
			constraints_function : str
			The name of the function that applies any special constraints to the parameters visited by
			the particles. Used if special_constraints is set to True.
			
			constraints_extra_arguments : list
			A list of arguments passed to the constraints function if special_constraints is set to True.
			The first element of the list must be a boolean that indicates if initial positions must be
			generated. This boolean should initially be set to True, so the constraint function knows to 
			generate initial positions, not next positions.

			show_animation : bool
			An animation of the swarm is shown at the end of the Experiment.run function if this argument
			is set to True. 
			
			disable_progress_bar : bool
			No progress bar is printed if this argument is set to True.
			
			get_parameters_from : str
			This string is either "g-values" or "average p-values". Used in the get_parameters method of
			the Swarm class. See the doc of this method for more information.
			If not sure, set to "g-values".

		Returns
		-------
		Nothing
		'''
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
	def constants(self, constants_dictionary=None):
		'''
		Returns dictionary of current constants if argument 'constants_dictionary' is not given.
		If the argument 'constants_dictionary' is given, then sets the constants to the values
		found in 'constants_dictionary', and returns 'constants_dictionary'.

		Parameters
		----------
		constants_dictionary : dict
.

		Returns
		-------
		constants : dict
		The constants set for this Experiment object.
		'''
		if constants_dictionary == None:
			constants = {'phi': self.phi, 'N': self.N, 'k': self.k, 
				'time_steps': self.time_steps, 'repetitions': self.repetitions}
		elif type(constants_dictionary) == dict:
			constants = constants_dictionary
			self.phi = constants["phi"]
			self.N = constants["N"]
			self.k = constants["k"]
			self.time_steps = constants["time_steps"]
			self.repetitions = constants["repetitions"]
		else:
			raise TypeError(f"Invalid type {type(constants_dictionary)} for dictionary")

		return constants

	def fn_info(self, fn_info_dictionary=None):
		'''
		Returns dictionary of current fn_info if argument 'fn_info_dictionary' is not given.
		If the dictionary is given, then sets the constants to the values
		found in the dictionary, and returns the same dictionary.

		Parameters
		----------
		fn_info_dictionary : dict
		The dictionary containing the constants phi, k, N, time_steps, repetitions.

		Returns
		-------
		fn_info : dict
		The function info set for this experiment object.
		'''
		if fn_info_dictionary == None:
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
		elif type(fn_info_dictionary) == dict:
			fn_info = fn_info_dictionary
			return fn_info
		else:
			raise TypeError(f"Invalid type {type(fn_info_dictionary)} for dictionary")

	def n_evaluations(self):
		'''
		Returns the number of evaluations.

		This function uses the n_evaluations helper function to calculate the
		number of evaluations, but sets the parameters N, time_steps, repetitions
		automatically. It is useful to get the number of evaluations of an experiment
		object quickly.

		Parameters
		----------
		None

		Returns
		-------
		n_evaluations : number
		The number of evaluations of an Experiment object, given the current constants
		configuration.
		'''
		return n_evaluations(self.N, self.time_steps, self.repetitions)

	def run(self, allowed_n_evaluations=None):
		'''
		Runs the experiment from beginning to end. 
		Returns the best found position, best value and error,
		and also assigns these to the Experiment object as attributes.

		If show_animation is set to True, the swarm will be animated.
		To see the animation again, use the command experiment.swarm.animate_swarm()

		Parameters
		----------
		allowed_n_evaluations : number
		The maximum number of evaluations allowed for this run. The true number of 
		evaluations bight be a bit higher or lower than this value.

		Returns
		-------
		best_position : np.ndarray
		An array of the best performing parameters found.
		best_f : float
		The value of the function, for example 'Rosenbrock', at the best_position.
		error : float
		The difference between best_f and the known best value optimal_f from fn_info.
		'''

		# Check if user has given a new number of evaluations.
		if allowed_n_evaluations == None:
			allowed_n_evaluations = n_evaluations(self.N, self.time_steps, self.repetitions)
		else:
			if allowed_n_evaluations <= 2*self.N*self.repetitions:
				raise ValueError(f"Number of evaluations must be greater than 2Nr. In this case >= {2*self.N*self.repetitions}")
		
		# Recalculate the time_steps to achieve this maximum number of evaluations.
		self.time_steps = math.ceil((allowed_n_evaluations - self.repetitions*self.N)/(self.repetitions*self.N))
		print("Running algorithm...")

		constants = self.constants()
		fn_info = self.fn_info()

		# Create swarm and evolve the swarm for the required number of repetitions.
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
			self.swarm.animate_swarm()

		return self.best_position, self.best_f, self.error

###################################################################

class Swarm(Experiment):
	'''
	The Swarm class inherits the __init__ function from the Experiment class.
	This class creates Particle instances and evolves them through time to get a final
	best position, value, and error.
	'''
	def random_initial_positions(self):
		'''
		Returns an array of random initial positions.

		Returns an array of random initial positions for each particle
		in a swarm. Each position is within the search space given by xmin, xmax,
		and any special constraints in the fn_info dictionary.

		Parameters
		----------
		None

		Returns
		-------
		initial_positions : np.ndarray
		An array of random initial positions that are within the required search area.
		The array has dimensions (number of particles, number of dimensions of the problem).
		'''
		initial_positions = np.inf*np.ones((self.N, self.dim))
		# Check if there are any special constraints
		if self.special_constraints == False:
			# Create array of initial positions
			# taking into account that some parameters must be integers
			for d in range(self.dim):
				if self.param_is_integer[d] == True:
					initial_positions[:,d] = np.random.randint(self.xmin[d], self.xmax[d], self.N)
				elif self.param_is_integer[d] == False:
					initial_positions[:,d] = np.random.uniform(self.xmin[d], self.xmax[d], self.N)
			# Note that these positions are all of type np.float64 even though randint is called
		else:
			# This should be modifiable automatically fix
			for particle in range(self.N):
				particle1 = Particle()
				initial_positions[particle] = eval(self.constraints_function)(particle1, self.constraints_extra_arguments)
			# Set the is_initial_positions to False so the constraints function
			# can generate next positions on the next time step
			self.constraints_extra_arguments[0] = False

		return initial_positions

	def random_initial_velocities(self):
		'''
		Returns an array of random initial velocities for the Particles.

		Parameters 
		----------
		None

		Returns
		-------
		initial_velocities : np.ndarray
		An array of random initial velocities that are within [-vmax, vmax).
		The array has dimensions (number of particles, number of dimensions of the problem).
		'''
		initial_velocities = np.random.uniform(-self.vmax, self.vmax, (self.N, self.dim))

		return initial_velocities

	def create_particles(self, initial_positions, initial_velocities):
		'''
		Creates a list of Particle objects for the swarm.

		Parameters
		----------
		initial_positions : array
		An array of initial positions for all N particles
		with shape (number of particles, number of dimensions)

		initial_velocities : array
		An array of initial velocities for all N particles
		with shape (number of particles, number of dimensions)

		Returns
		-------
		None
		'''
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

	def random_informants(self):
		'''
		Chooses k informants randomly for each Particle of the Swarm.
		Sets a list of these informant Particles as attributes for each Particle.

		Parameters
		----------
		None

		Returns
		-------
		None
		'''
		for particle in self.particles:
			particle.informants = np.random.choice(self.particles, particle.k)

	def distribute_swarm(self):
		'''
		Distributes Particles in the search space and chooses informants for each particle.
		Also initializes an array of positions for animating the Swarm.

		Parameters
		----------
		None

		Returns
		-------
		None
		'''

		# Create array of initial positions and velocities
		initial_positions = self.random_initial_positions()
		initial_velocities = self.random_initial_velocities()

		self.create_particles(initial_positions, initial_velocities)

		# Initiate k informants randomly
		self.random_informants()

		# Initialise array of positions for animation
		self.positions = np.inf*np.ones((self.time_steps, self.N, self.dim))
		self.positions[0,:,:] = initial_positions

	def evolve(self):
		'''
		Updates positions of Particles for all time steps. Populates the positions array
		for animating the Swarm. Also shows a progress bar.

		Parameters
		----------
		None

		Returns
		-------
		None
		'''

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


	def get_parameters(self):
		'''
		Returns optimal parameters and lowest value found.

		If get_parameters_from is set to 'g-values' in the fn_info dictionary,
		then the optimal parameters are chosen from the global values. The g-values
		of all Particles in the Swarm are inspected and the lowest value is chosen.

		If get_parameters_from is set to 'average p-values', then the
		optimal parameters are chosen from the best visited positions of each particle.
		The p-values of all Particles in the Swarm are inspected, and the average of 
		positions and values are returned.

		Parameters
		----------
		None

		Returns
		-------
		result : np.ndarray
		An array containing the best found parameter for each dimension and the value
		of the function with these parameters. The value is the last element of the array.
		'''
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

	def run_algorithm(self):
		'''
		Evolves the swarm through time for the required number of repetitions.

		Assigns the best found position, value, and error to the Swarm, and
		returns these as a tuple.

		Parameters
		---------
		None

		Returns
		-------
		(best_position, best_f, error)

		best_position : np.ndarray
		An array of the best parameters found.

		best_f : float
		The value of the function at this position.

		error : float
		The difference between the best_f and optimal_f given in the fn_info dictionary.

		'''
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

		return self.best_position, self.best_f, self.error


	def animate_swarm(self):
		'''
		Plots an animation of the best repetition of evolving the swarm.

		Only the first 2 dimensions are plotted for a higher-dimensional problem.

		Parameters
		----------
		None

		Returns
		-------
		None
		'''

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

	def update_frames(self, j, *fargs):
		'''
		Updates the frames of the animation.

		Parameters
		----------
		j : int
		Frame number.

		*fargs contains scat, all_positions, and best_value_index

		scat : scatter plot

		all_positions : np.ndarray
		An array of all positions visited by the Swarm.

		best_value_index : int
		The index indicating the best performing repetition.

		Returns
		-------
		None
		'''
		scat, all_positions, best_value_index = fargs
		try:
			scat.set_offsets(all_positions[best_value_index,j,:,0:2])
		except:
			print("animation finished")
			self.animation.event_source.stop()

###################################################################

# Particle objects are created within the swarm class methods. 
class Particle(Experiment):
	'''
	Particle instances are created within the Swarm class methods. Particles
	inherit constants and function info from the Experiment class.
	'''

	def set_initial_state(self, pos, vel, p):
		'''
		Initializes a particle with the assigned initial position, initial velocity,
		p-value, g-value and an empty list of informants.
		
		Parameters
		----------
		pos : array
		An array containing an initial position for each dimension.

		vel : array
		An array containing an initial velocity for each dimension.

		p : array
		An array containing the best found position and value that the particle
		has visited. The initial state has p equal to the initial position and its
		value as the particle has not visited any other positions yet.

		Returns
		-------
		None

		'''
		self.pos = pos
		self.vel = vel
		# Set initial best found value by particle
		# format: np array of shape (1, 3) - x, y, value
		self.p = p

		# Best found position and value by informants or itself
		# format: np array of shape (1, 3) - x, y, value
		self.g = p

		# Empty list of informants
		self.informants = []

	def communicate(self):
		'''
		Receives g-values from informants and updates the Particle's g-value accordingly.

		If the best received g-value is smaller than the Particle's g-value, then the
		particles g-value is set to the received g-value.

		Parameters
		----------
		None

		Returns
		-------
		None
		'''

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

	def random_confidence(self):
		'''
		Randomly assigns confidence parameters c2 and c3 in the interval [0, cmax)

		Parameters
		----------
		None

		Returns
		-------
		(c2, c3)
		c2 : np.ndarray
		The confidence in the Particle's p-value. The array has the same
		dimensions as the problem.

		c3 : np.ndarray
		The confidence in the particle's g-value. The array has the same
		dimensions as the problem.
		'''
		c2 = np.inf*np.ones(self.dim)
		c3 = np.inf*np.ones(self.dim)

		for d in range(self.dim):
			c2[d] = np.random.uniform(0, self.cmax)
			c3[d] = np.random.uniform(0, self.cmax)
		return (c2, c3)

	# Compare the new position and the old p-value, and update p
	def update_p(self, value):
		'''
		Compares the new position and the old p-value, and updates p accordingly.

		If the new position is better than the old p-value, then the p-value is set
		to the new position and that position's value.

		Parameters
		----------
		value : number
		The value of the position that is compared to p

		Returns
		-------
		None
		'''

		if value < self.p[self.dim]:
			self.p[self.dim] = value
			self.p[0:self.dim] = self.pos

	def update_g(self, value):
		'''
		Updates the particle's g-value if the new position is better than the
		previously known g-value. Finishes by communicating with informants
		and updating the g-value again.

		Parameters
		----------
		value : number
		The value of the position that is compared to g

		Returns
		-------
		None
		'''
		if value < self.g[self.dim]:
			self.g[self.dim] = value
			self.g[0:self.dim] = self.pos
		self.communicate()

	def find_vel(self):
		'''
		Calculates the velocity of the Particle according to the 
		update equation from the PSO book.

		Since this is PSO(0), c1 is constant, and c2,c2 are chosen
		randomly from a rectangular probability distribution.

		Parameters
		----------
		None

		Returns
		-------
		None
		'''
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
		'''
		Uses the calculated velocity to set the next position for a Particle
		while taking into account any constraints on the search space.

		Parameters
		----------
		None

		Returns
		-------
		None
		'''
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
		'''
		Checks whether a position is inside the allowed search space given by xmin and xmax.
		Returns True or False.

		Parameters
		----------
		possible_pos : np.ndarray
		A possible position of the particle that may or may not be within the search area.

		Returns
		-------
		is_allowed : bool
		True if possible_pos is inside the search area. False if possible_pos is not inside
		the search area.
		'''
		smaller_than_xmax = possible_pos <= self.xmax
		greater_than_xmin = possible_pos >= self.xmin
		is_allowed = np.zeros((len(self.xmax), 2))
		is_allowed[:,0] = smaller_than_xmax
		is_allowed[:,1] = greater_than_xmin
		is_allowed = np.all(is_allowed, axis=1)

		return is_allowed

	def step(self):
		'''
		Moves a Particle from one position to another while taking into account
		any constraints.

		Parameters
		----------
		None

		Returns
		-------
		None
		'''
		value = eval(self.fn_name)(self.pos)
		self.update_p(value)
		self.update_g(value)
		self.find_vel()
		self.set_pos()

###################################################################

def evaluate_experiment(experiment, n_evaluations, n_iterations):
	'''
	Evaluates the experiment passed as an argument. Plots a histogram of the results,
	and finds the average result and standard deviation.

	Parameters
	----------
	experiment : Experiment object
	The Experiment that you wish to evaluate.

	n_evaluations : int
	The number of evaluations allowed to the experiment.

	n_iterations : int
	The number of times the experiment is run.

	Returns
	-------
	(avg_result, standard_deviation)
	avg_result : np.ndarray
	The average best values found by the experiment. 

	standard_deviation
	The standard deviation of the results.
	'''
	results = np.inf*np.ones(n_iterations)
	for n in tqdm(range(n_iterations), desc='Testing function'):

		# Disable printing
		experiment.disable_progress_bar = True
		text_trap = io.StringIO()
		sys.stdout = text_trap

		# Disable animation
		previous_show_animation = experiment.show_animation
		experiment.show_animation = False

		experiment.run(n_evaluations)
		results[n] = experiment.best_f

	# Restore printing
	sys.stdout = sys.__stdout__

	# Restore animation
	experiment.show_animation = previous_show_animation

	avg_result = np.average(results)
	standard_deviation = np.std(results)

	print("")
	print("Experiment evaluation finished.")
	print(f"The mean result is {avg_result}")
	print(f"with a standard deviation of {standard_deviation}")

	plt.hist(results, 100, (0, 10))
	plt.show()
	return avg_result, standard_deviation

###################################################################

def optimize_constants(allowed_evaluations=500, allowed_deviation=20,\
	optimization_time_steps=100, optimization_repetitions=1):
	'''
	This is an attempt at optimizing the constants with PSO itself.
	Returns the optimal constants dictionary.

	The result is a starting point but currently, changing the constants configuration manually
	works much better. 

	However, this function shows how special constraints and a real machine learning
	problem can be applied to this PSO algorithm.

	Note that this function optimizes for the function and search space given in the file fn_info.txt
	in the optimusbeez directory.

	Parameters
	----------
	allowed_evaluations : int
	The number of evaluations allowed for the Experiment.

	allowed_deviation : int
	The allowed deviation from the allowed evaluations.

	optimization_time_steps : int
	The number of times the Experiment is run before choosing the best constants configuration.

	optimization_repetitions : int
	The number of times the optimization algorithm is run on the Experiment.

	Returns
	-------
	optimal_constants : dict
	A dictionary of the optimal constants phi, k, N, time_steps, repetitions.
	'''

	# These are for the constant optimization Experiment
	optimal_experiment_constants = {'phi': 2.4, 'k':7, 'N': 15, 'time_steps': optimization_time_steps,
		'repetitions':optimization_repetitions}

	phi_min = 2.00001
	phi_max = 3
	k_min = 1
	N_min = 1
	N_max = 30
	time_steps_min = 1
	time_steps_max = allowed_evaluations + allowed_deviation
	repetitions_min = 3 # If repetitions_min is set to 1, the algorithm tends to favor this value
	repetitions_max = 30

	xmin = [phi_min, k_min, N_min, time_steps_min, repetitions_min]
	xmax = [phi_max, N_max, N_max, time_steps_max, repetitions_max]

	param_is_integer = [False, True, True, True, True] # Only phi is continuous

	# This evaluation function creates Experiment objects to evaluate the constants configuration
	evaluation_function_name = "constant_optimization_evaluation_function"	

	optimal_experiment_fn_info = {
	"fn_name":evaluation_function_name,
	"optimal_f":0, # We want to reduce the error to 0
	"dim":5, # There are 5 constants: phi, k, N, t, r
	"xmin":xmin, "xmax":xmax,
	"param_is_integer":param_is_integer,
	"special_constraints":True, # N,t,r are related through the number of evaluations
	"constraints_function":"Ntr_constrain_position",
	"constraints_extra_arguments":[True,allowed_evaluations,allowed_deviation],
	"show_animation":True,
	"disable_progress_bar":False,
	"get_parameters_from":"average p-values"
	}

	# Average p-values work better than g-values because the evaluation function
	# is not deterministic. The same constants configuration will not give the same
	# error every time.

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
	'''
	This is an alternative evaluation function to Rosenbrock, for example.
	The function name is given as an argument in fn_info.

	Unlike Rosenbrock, this function creates Experiment objects. It has the constants
	configuration as a position, and returns the error of the Experiment with that
	constants configuration.

	Parameters
	----------
	new_constants : array
	An array of constants of the form [phi, k, N, time_steps, repetitions].

	Returns
	-------
	experiment.error : float
	The error of the Experiment with this constants configuration.
	'''

	# Suppress printing
	text_trap = io.StringIO()
	sys.stdout = text_trap

	constants = position_to_constants_dictionary(new_constants)
	# Function info is default from function_info.txt
	experiment = Experiment(constants=constants)
	experiment.show_animation = False
	experiment.disable_progress_bar = True

	experiment.run()

	return experiment.error

def Ntr_constrain_position(particle, extra_arguments):
	'''
	Constrains N, time_steps (t), repetitions (r) so the number of evaluations
	does not exceed the maximum number of evaluations.

	Returns a new position and velocity for a Particle that is within these
	special constraints.

	This function is given as a constraints_function argument in fn_info.
	It is a dual-purpose function for both creating initial positions and
	choosing a next position given a velocity.

	Parameters
	----------
	particle : Particle object

	extra_arguments : list
	A list of extra arguments containing 3 arguments in the following order.
		is_initial_positions : bool
		If set to True, initial positions will be returned instead of calculating
		a next position. If set to False, then a next position will be calculated
		according to the previous position and velocity.

		allowed_evaluations : int
		The number of evaluations allowed for the Particle.

		allowed_deviation : int
		The allowed deviation from the number of evaluations.

	Returns
	-------
	Depending on the is_initial_positions argument.
		If True, initial_positions : np.ndarray
		An array of initial positions for the Particle, which in this case is an array of
		randomly generated constants

		If False, (new_pos, vel)
			new_pos: np.ndarray
			An array indicating the new position of a Particle that is within the special
			constraints.

			vel : np.ndarray
			An array indicating the velocity of the particle.
	'''
	is_initial_positions = extra_arguments[0]
	allowed_evaluations = extra_arguments[1]
	allowed_deviation = extra_arguments[2]

	# Check if initial positions are required
	if is_initial_positions == True:
		print("Generating initial positions")
		initial_positions = generate_random_constants(allowed_evaluations, allowed_deviation)
		return initial_positions

	else:
		print("Generating next positions")
		allowed_n_evaluations = allowed_evaluations + allowed_deviation

		previous_pos = particle.pos
		previous_pos_n_evaluations = n_evaluations(previous_pos[2], previous_pos[3], previous_pos[4])
		if previous_pos_n_evaluations > allowed_n_evaluations:
			raise ValueError(f"The previous position is invalid given the special constraints.")
		
		possible_pos = previous_pos + particle.vel
		vel = particle.vel

		# Constrain all values to rectangular search area
		in_search_area = particle.is_in_search_area(possible_pos)
		possible_pos = np.where(in_search_area, possible_pos, particle.pos)
		vel = np.where(in_search_area, particle.vel, 0)

		# Check if special constraints already satisfied
		possible_n_evaluations = n_evaluations(math.ceil(possible_pos[2]), math.ceil(possible_pos[3]),\
			math.ceil(possible_pos[4]))
		if possible_n_evaluations > allowed_n_evaluations:
			# Further constrain N, time_steps and repetitions according to allowed number of evaluations
			# This is done by randomly selecting one of N, t, r, and reverting this back to
			# the previous value, repeated until the constraint is satisfied
			while True:
				one_of_Ntr = np.random.randint(2,5)
				possible_pos[one_of_Ntr] = previous_pos[one_of_Ntr]
				vel[one_of_Ntr] = 0
				possible_n_evaluations = n_evaluations(math.ceil(possible_pos[2]), math.ceil(possible_pos[3]),\
					math.ceil(possible_pos[4]))
				if possible_n_evaluations <= allowed_n_evaluations and possible_n_evaluations >= 2:
					new_pos = possible_pos
					break
				else:
					continue
		else:
			# Constraint already satisfied
			new_pos = possible_pos

		# Further constrain k to be less than or equal to N
		if new_pos[1] > new_pos[2]:
			new_pos[1] = new_pos[2]

		# Convert required constants to integers
		for constant in range(1,5):
			new_pos[constant] = math.ceil(new_pos[constant])

		return new_pos, vel

def position_to_constants_dictionary(position):
	'''
	Turns a 5D position array into the constants dictionary for later use.

	Parameters
	----------
	position : array
	The array is of the form [phi, k, N, time_steps, repetitions].

	Returns
	-------
	constants : dict
	A constants dictionary.
	'''
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

# If this file is run as a Python script, an experiment object is created
# with the default constants and function info from optimal_constants.txt
# and fn_info.txt. The experiment is run once.
if __name__ == "__main__":
	experiment = Experiment()
	experiment.run()