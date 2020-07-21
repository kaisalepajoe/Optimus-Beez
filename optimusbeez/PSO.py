# Particle Swarm Optimization
# This script finds the global MINIMUM of the
# selected function.

# This is the simplest version, PSO(0) from
# the book "Particle Swarm Optimization" by
# Maurice Clerc.

###################################################################

# Import required modules
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pkgutil
from .evaluate import evaluate

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

###################################################################

# The training session class is inherited by particles and swarms
# This class sets the required parameters from constants and fn_info
class Experiment:
	def __init__(self, constants=None, fn_info=None):
		if constants == None:
			constants = read_dictionary_from_file('optimal_constants.txt')
		if fn_info == None:
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
			self.xmin = fn_info["xmin"]
			self.xmax = fn_info["xmax"]
			self.show_animation = fn_info["show_animation"]

			# Calculate maximum velocity
			self.vmax = abs(self.xmax - self.xmin)/2

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
				"xmin":self.xmin, "xmax":self.xmax, "show_animation":self.show_animation}
			return fn_info
		elif type(dictionary) == dict:
			fn_info = dictionary
			return fn_info
		else:
			raise TypeError(f"Invalid type {type(dictionary)} for dictionary")

	def n_evaluations(self, N=None, time_steps=None, repetitions=None):
		if N==None or time_steps==None or repetitions==None:
			n = self.N*self.time_steps*self.repetitions + self.repetitions*self.N
			return n
		elif type(N) == int and type(time_steps) == int and type(repetitions) == int:
			n = N*time_steps*repetitions + repetitions*N
			return n
		else:
			raise TypeError(f"Invalid types {type(N)}, {type(time_steps)}, {type(repetitions)}\
				 for N, time_steps, repetitions. Must be integers.")

	def generate_random_constants(self, allowed_evaluations, allowed_deviation):
		if type(allowed_evaluations) != int and type(allowed_deviation) != int:
			raise TypeError(f"Invalid types {type(allowed_evaluations)} and {type(allowed_deviation)} for allowed_evaluations and allowed_deviation.")
		if allowed_evaluations <= allowed_deviation:
			raise ValueError(f"Allowed deviation cannot be larger than or equal to allowed evaluations")
		if allowed_evaluations < 2:
			raise ValueError(f"Allowed evaluations cannot be less than 2")
		if allowed_deviation < 0:
			raise ValueError(f"Allowed deviation cannot be less than 0")

		# Set minimum and maximum values for search
		N_min = 1
		N_max = 30
		repetitions_min = 1
		repetitions_max = 30

		time_steps_min = 1
		time_steps_max = allowed_evaluations + allowed_deviation

		k_min = 1
		phi_min = 2.00001
		phi_max = 2.4

		# Initiate empty dictionary
		constants = {}

		# Set N-t-r grid size
		NTR = np.ones((N_max-N_min+1, time_steps_max-time_steps_min+1, \
			repetitions_max-repetitions_min+1))
		# Populate grid with total time steps
		for n in range(len(NTR)):
			for t in range(len(NTR[n])):
				for r in range(len(NTR[n, t])):
					NTR[n,t,r] = self.n_evaluations(N=n+N_min, time_steps=t+time_steps_min, repetitions=r+repetitions_min)
		valid_NTR_choices = np.where((NTR >= (allowed_evaluations - allowed_deviation)) & (NTR <= (allowed_evaluations + allowed_deviation)))
		valid_NTR_choices = np.array([valid_NTR_choices[0], valid_NTR_choices[1], valid_NTR_choices[2]])
		# valid_NTR_choices contains the indices that correspond to parameters
		# with the allowed total number of time steps

		# Set N, time_steps, repetitions
		n, t, r = valid_NTR_choices[:,np.random.randint(0,valid_NTR_choices.shape[1])]
		N = n + N_min
		time_steps = t + time_steps_min
		repetitions = r + repetitions_min

		# Set parameters
		constants["phi"] = np.random.uniform(phi_min, phi_max)
		constants["N"] = N
		constants["k"] = np.random.randint(k_min, N+1)
		constants["time_steps"] = time_steps
		constants["repetitions"] = repetitions

		return constants

	def optimize_constants(self, tests=None, tests_with_each_constants=None, \
		allowed_evaluations=None, allowed_deviation=None, \
		save_configuration=None, create_file=None):
		if tests == None:
			while True:
				print("Set number of tests:")
				try:
					tests = int(input())
					break
				except:
					print("Invalid input for tests. Cannot convert to integer.")
					continue
		if tests_with_each_constants == None:
			while True:
				print("Set number of repetitions for each constants configuration:")
				try:
					tests_with_each_constants = int(input())
					break
				except:
					print("Invalid input for number of repetitions with each constants configuration.\
						Cannot convert to integer.")
					continue
		if allowed_evaluations == None:
			while True:
				print("Set allowed evaluations:")
				try:
					allowed_evaluations = int(input())
					break
				except:
					print("Invalid input for allowed evaluations. Cannot convert to integer.")
					continue
		if allowed_deviation == None:
			while True:
				print("Set allowed deviation from number of evaluations:")
				try:
					allowed_deviation = int(input())
					break
				except:
					print("Invalid input for allowed deviation from number of evaluations.\
						Cannot convert to integer.")
					continue

		tests = int(tests)
		tests_with_each_constants = int(tests_with_each_constants)
		allowed_evaluations = int(allowed_evaluations)
		allowed_deviation = int(allowed_deviation)

		print("Finding optimal constants...")

		best_error = np.inf

		fn_info = self.fn_info()

		for t in range(tests):
			print(f"Test {t+1}/{tests}")
			random_constants = self.generate_random_constants(allowed_evaluations, allowed_deviation)

			errors = np.inf*np.ones(tests_with_each_constants)

			# Repeat several times for this constants configuration		
			for constants_repetition in range(tests_with_each_constants):
				swarm = Swarm(random_constants, fn_info)
				swarm.distribute_swarm()
				swarm.run_algorithm()
				errors[constants_repetition] = swarm.error

			avg_error = np.average(errors)

			if avg_error < best_error:
				best_constants = random_constants
				best_error = avg_error


		print("The best found constants configuration is:")
		print(best_constants)
		print(f"This configuration has the error: {best_error}")

		if save_configuration == None:
			while True:
				print("Save found configuration for this Experiment object?")
				print("Write yes or no:")
				save_configuration = input()
				if save_configuration == "yes" or save_configuration == "no":
					break
				else:
					print("Invalid input. You must write yes or no.")
					continue

		if save_configuration == "yes":
			self.N = best_constants["N"]
			self.time_steps = best_constants["time_steps"]
			self.repetitions = best_constants["repetitions"]
			self.k = best_constants["k"]
			self.phi = best_constants["phi"]
		elif save_configuration == "no":
			pass
		elif type(save_configuration) != str:
			raise TypeError(f"save_configuration should be of type string, not type \
				{type(save_configuration)}")
		else:
			raise ValueError(f"Invalid argument for save_configuration: {save_configuration}.\
				Should be yes or no.")

		if create_file == None:
			while True:
				print("Create new txt file with this configuration?")
				print("Write yes or no:")
				create_file = input()
				if create_file =="yes" or create_file == "no":
					break
				else:
					print("Invalid input. You must write yes or no.")
							
		if create_file == "yes":
			while True:
				print("Write a path for the file:")
				filepath = input()
				try:
					write_dictionary_to_file(best_constants, filepath)
					return
				except:
					print("Are you sure you wrote the path correclty?")
					print(f"Your input: {filepath}")
					continue
		elif create_file == "no":
			return
		elif type(create_file) != str:
			raise TypeError(f"create_file should be of type string, not type \
				{type(create_file)}")
		else:
			raise ValueError(f"Invalid argument for create_file: {create_file}.\
				Should be yes or no.")

	def run(self, max_n_evaluations=None):
		if max_n_evaluations == None:
			max_n_evaluations = self.n_evaluations()
		else:
			if max_n_evaluations <= 2*self.N*self.repetitions:
				raise ValueError(f"Number of evaluations must be greater than 2Nr. In this case >= {2*self.N*self.repetitions}")
		
		self.time_steps = int((max_n_evaluations - self.repetitions*self.N)/(self.repetitions*self.N))
		print("Running algorithm...")

		constants = self.constants()
		fn_info = self.fn_info()

		self.swarm = Swarm(constants, fn_info)
		self.swarm.distribute_swarm()
		self.swarm.run_algorithm()
		true_n_evaluations = self.swarm.n_evaluations()

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

	# Set random positions, velocities, informants, and p-values for all particles
	def distribute_swarm(self):
		# Create array of initial positions
		initial_positions = np.random.uniform(self.xmin, self.xmax, (self.N, self.dim))

		# Create array of initial p-values by evaluating initial positions
		p_values = np.inf*np.ones((self.N, self.dim+1))
		for i, pos in enumerate(initial_positions):
			p_values[i,self.dim] = evaluate(pos, self.fn_name)
			p_values[i,0:self.dim] = pos

		# Create array of random velocities (up to limit)
		velocities = np.random.uniform(-self.vmax, self.vmax, (self.N, self.dim))

		constants = self.constants()
		fn_info = self.fn_info()

		# Create list of particle objects
		self.particles = []
		for i in range(self.N):
			pos = initial_positions[i]
			vel = velocities[i]
			p = p_values[i]
			particle = Particle(constants, fn_info)
			particle.set_initial_state(pos, vel, p)
			self.particles.append(particle)

		# Initiate k informants randomly
		self.random_informants()

		# Initialise array of positions for animation
		self.positions = np.inf*np.ones((self.time_steps, self.N, self.dim))
		self.positions[0,:,:] = initial_positions

	# Choose k informants randomly
	def random_informants(self):
		for particle in self.particles:
			particle.informants = np.random.choice(self.particles, particle.k)

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
		# Cannot simulate if dimension > 2
		if self.dim > 2:
			return
		# Plot initial positions of particles
		fig, ax = plt.subplots()
		ax.set_xlim(self.xmin, self.xmax)
		ax.set_ylim(self.xmin, self.xmax)
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
			scat.set_offsets(all_positions[best_value_index,j])
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

	def step(self):
		# Evaluate current position
		# Update p if current position is LOWER
		value = evaluate(self.pos, self.fn_name)
		if value < self.p[self.dim]:
			self.p[self.dim] = value
			self.p[0:self.dim] = self.pos
		if value < self.g[self.dim]:
			self.g[self.dim] = value
			self.g[0:self.dim] = self.pos

		# Communicate with informants, update g
		self.communicate()

		# Set confidence parameters
		c2, c3 = self.random_confidence()

		# Update velocity
		possible_vel = self.c1*self.vel + \
			c2*(self.p[0:self.dim] - self.pos) + \
			c3*(self.g[0:self.dim] - self.pos)
		# Constrain velocity
		for d in range(self.dim):
			if abs(possible_vel[d]) <= self.vmax:
				self.vel[d] = possible_vel[d]
			elif possible_vel[d] > self.vmax:
				self.vel[d] = self.vmax
			elif possible_vel[d] < -self.vmax:
				self.vel[d] = -self.vmax

		# Update position
		possible_pos = self.pos + self.vel
		# Constrain particle to search area
		# Set velocity to 0 if possible_pos
		# outside search area to avoid touching
		# the boundary again in the next time step.
		for d in range(self.dim):
			if self.xmin <= possible_pos[d] <= self.xmax:
				self.pos[d] = possible_pos[d]
			elif possible_pos[d] < self.xmin:
				self.pos[d] = self.xmin
				self.vel[d] = 0
			elif possible_pos[d] > self.xmax:
				self.pos[d] = self.xmax
				self.vel[d] = 0

###################################################################

if __name__ == "__main__":
	experiment = Experiment()
	experiment.run()