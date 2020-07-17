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
	data = pkgutil.get_data('optimusbeez', filename)
	dictionary = eval(data)
	return dictionary

def write_dictionary_to_file(dictionary, filename):
	file = open(filename, "w")
	file.write(str(dictionary))
	file.close()

# Determine number of repetitions given constants
def determine_n_evaluations(N, time_steps, repetitions):
	return N*time_steps*repetitions + repetitions*N

# Choose k informants randomly
def random_informants(particles):
	for particle in particles:
		particle.informants = np.random.choice(particles, particle.k)

# Determine error
def determine_error(true_position, position):
	xy_error = abs(true_position - position)
	error = np.sqrt(xy_error[0]**2 + xy_error[1]**2)
	return error

def set_random_constants(required_time_steps, allowed_deviation):
	# Set minimum and maximum values for search
	N_min = 3
	N_max = 30
	repetitions_min = 1
	repetitions_max = 30

	time_steps_min = 10
	time_steps_max = required_time_steps + allowed_deviation

	k_min = 1
	phi_min = 2.00001
	phi_max = 2.4

	# Initiate empty dictionary
	constants = {}

	# Set N-t-r grid size
	NTR = np.ones((N_max - N_min, 2*allowed_deviation, repetitions_max - repetitions_min))
	# Populate grid with total time steps
	for n in range(len(NTR)):
		for t in range(len(NTR[n])):
			for r in range(len(NTR[n, t])):
				NTR[n,t,r] = (n+N_min)*(t+time_steps_min)*(r+repetitions_min)
	valid_NTR_choices = np.where((NTR >= required_time_steps - allowed_deviation) & (NTR <= required_time_steps + allowed_deviation))
	valid_NTR_choices = np.array([valid_NTR_choices[0], valid_NTR_choices[1], valid_NTR_choices[2]])
	# valid_NTR_choices = np.array((zip(valid_NTR_choices[0], valid_NTR_choices[1], valid_NTR_choices[2])))
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

###################################################################

# The training session class is inherited by particles and swarms
# This class sets the required parameters from constants and function_info
class Experiment:
	def __init__(self, constants=None, function_info=None):
		if constants == None:
			constants = read_dictionary_from_file('optimal_constants.txt')
		if function_info == None:
			function_info = read_dictionary_from_file('function_info.txt')

		self.N = constants["N"]
		self.time_steps = constants["time_steps"]
		self.repetitions = constants["repetitions"]
		self.fn_name = function_info["fn_name"]
		self.true_position = function_info["true_position"]
		self.k = constants["k"]
		self.phi = constants["phi"]
		self.xmin = function_info["xmin"]
		self.xmax = function_info["xmax"]
		self.show_animation = function_info["show_animation"]

		# Calculate maximum velocity
		self.vmax = abs(self.xmax - self.xmin)/2

		# Calculate confidence parameters using phi
		self.c1 = 1/(self.phi-1+np.sqrt(self.phi**2-2*self.phi))
		self.cmax = self.c1*self.phi

	def constants(self):
		constants = {'phi': self.phi, 'N': self.N, 'k': self.k, 
			'time_steps': self.time_steps, 'repetitions': self.repetitions}
		return constants

	def function_info(self):
		function_info = {"fn_name":self.fn_name, "true_position":self.true_position,
			"xmin":self.xmin, "xmax":self.xmax, "show_animation":self.show_animation}
		return function_info
		
	def optimize_constants(self):
		print("Set number of tests:")
		tests = int(input())
		print("Set number of repetitions for each constants configuration:")
		tests_with_each_constants = int(input())
		print("Set allowed time_steps:")
		allowed_time_steps = int(input())
		print("Set allowed deviation from time steps:")
		allowed_deviation = int(input())

		print("Finding optimal constants...")

		best_error = np.inf

		function_info = self.function_info()

		for t in range(tests):
			print(f"Test {t+1}/{tests}")
			random_constants = set_random_constants(allowed_time_steps, allowed_deviation)

			errors = np.inf*np.ones(tests_with_each_constants)

			# Repeat several times for this constants configuration		
			for constants_repetition in range(tests_with_each_constants):
				swarm = Swarm(random_constants, function_info)
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

		while True:
			print("Save this configuration for this Experiment object?")
			print("Write yes or no:")
			answer = input()
			if answer == "yes":
				self.N = best_constants["N"]
				self.time_steps = best_constants["time_steps"]
				self.repetitions = best_constants["repetitions"]
				self.k = best_constants["k"]
				self.phi = best_constants["phi"],
				while True:
					print("Also choose")
					print("1 to create new file with this configuration")
					print("2 to overwrite optimal_constants.txt")
					print("3 to skip")
					print("Write 1, 2 or 3:")
					choice = input()
					if choice =="1":
						print("Write a name for the file:")
						filename = input()
						write_dictionary_to_file(best_constants, filename)
						return
					elif choice == "2":
						write_dictionary_to_file(best_constants, "optimal_constants.txt")
						return
					elif choice =="3":
						return
					else:
						print("Invalid input.")
						continue
				break
			elif answer == "no":
				print("You have chosen to discard this configuration")
				break
			else:
				print("Invalid input.")
				continue

	def run(self):
		# Prompt user for input
		default_evaluations = determine_n_evaluations(
			self.N, self.time_steps, self.repetitions) 
		print(f"Number of evaluations is set to {default_evaluations}")
		print("Change number of evaluations? Write yes or no:")
		set_evaluations = input()
		if set_evaluations == "yes":
			print("Set maximum number of evaluations:")
			n_evaluations = int(input())
			self.time_steps = int((n_evaluations - self.repetitions*self.N)/(self.repetitions*self.N))
		print("Running algorithm...")

		constants = self.constants()
		function_info = self.function_info()

		self.swarm = Swarm(constants, function_info)
		self.swarm.distribute_swarm()
		self.swarm.run_algorithm()
		self.n_evaluations = determine_n_evaluations(self.swarm.N, self.swarm.time_steps, self.swarm.repetitions)

		self.best_position = self.swarm.best_position
		self.best_f = self.swarm.best_f
		self.error = self.swarm.error

		print(f"{self.n_evaluations} evaluations made.")
		print(f"The best position is {self.best_position}.")
		print(f"The value at this position is {self.best_f}")
		print(f"Distance from true global minimum: {self.error}")

		if self.show_animation == False:
			pass
		else:
			self.swarm.simulate_swarm()	

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
		received = np.zeros((self.k, 3))
		for i, informant in enumerate(self.informants):
			received[i, :] = informant.g
		# Set g to LOWEST value
		i = np.argmin(received[:,2])
		self.g = received[i]

	# Randomly assign confidence parameters
	# c2 and c3 in the interval [0, cmax)
	def random_confidence(self):
		c2 = np.array([np.random.uniform(0, self.cmax), \
			np.random.uniform(0, self.cmax)])
		c3 = np.array([np.random.uniform(0, self.cmax), \
			np.random.uniform(0, self.cmax)])
		return (c2, c3)

	def step(self):
		# Evaluate current position
		# Update p if current position is LOWER
		value = evaluate(self.pos, self.fn_name)
		if value < self.p[2]:
			self.p[2] = value
			self.p[0:2] = self.pos
		if value < self.g[2]:
			self.g[2] = value
			self.g[0:2] = self.pos

		# Communicate with informants, update g
		self.communicate()

		# Set confidence parameters
		c2, c3 = self.random_confidence()

		# Update velocity
		possible_vel = self.c1*self.vel + \
			c2*(self.p[0:2] - self.pos) + \
			c3*(self.g[0:2] - self.pos)
		# Constrain velocity
		for d in range(2):
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
		for d in range(2):
			if self.xmin <= possible_pos[d] <= self.xmax:
				self.pos[d] = possible_pos[d]
			elif possible_pos[d] < self.xmin:
				self.pos[d] = self.xmin
				self.vel[d] = 0
			elif possible_pos[d] > self.xmax:
				self.pos[d] = self.xmax
				self.vel[d] = 0

###################################################################

class Swarm(Experiment):

	# Set random positions, velocities, informants, and p-values for all particles
	def distribute_swarm(self):
		# Create array of initial positions
		initial_positions = np.random.uniform(self.xmin, self.xmax, (self.N, 2))

		# Create array of initial p-values by evaluating initial positions
		p_values = np.inf*np.ones((self.N, 3))
		for i, pos in enumerate(initial_positions):
			p_values[i,2] = evaluate(pos, self.fn_name)
			p_values[i,0:2] = pos

		# Create array of random velocities (up to limit)
		velocities = np.random.uniform(-self.vmax, self.vmax, (self.N, 2))

		constants = self.constants()
		function_info = self.function_info()

		# Create list of particle objects
		self.particles = []
		for i in range(self.N):
			pos = initial_positions[i]
			vel = velocities[i]
			p = p_values[i]
			particle = Particle(constants, function_info)
			particle.set_initial_state(pos, vel, p)
			self.particles.append(particle)

		# Initiate k informants randomly
		random_informants(self.particles)

		# Initialise array of positions for animation
		self.positions = np.inf*np.ones((self.time_steps, self.N, 2))
		self.positions[0,:,:] = initial_positions

	# Update positions of particles for all time steps
	def evolve(self):
		for time_step in range(self.time_steps):
			for i, particle in enumerate(self.particles):
				particle.step()
				# Update positions for animation
				self.positions[time_step,i,:] = particle.pos
			# Select informants for next time step
			random_informants(self.particles)

	# Extract optimal parameters (from g-values)
	def get_parameters(self):
		final_g = np.inf*np.ones((self.N, 3))
		for i,particle in enumerate(self.particles):
			final_g[i,:] = particle.g
		optimal_i = np.argmin(final_g[2])
		x = final_g[optimal_i][0]
		y = final_g[optimal_i][1]
		f = final_g[optimal_i][2]
		return np.array([x, y, f])

	# Run the algorithm for required number of repetitions
	# Return best found position, value, and error
	def run_algorithm(self):

		# results contains the best found positions and values for each repetition
		results = np.inf*np.ones((self.repetitions, 3))
		# all_positions contains all the visited positions for each repetition
		# all_positions is used to create an animation of the swarm
		self.all_positions = np.inf*np.ones((self.repetitions, self.time_steps, self.N, 2))

		for r in range(self.repetitions):
			self.distribute_swarm()
			self.evolve()
			result = self.get_parameters()
			results[r] = result
			self.all_positions[r] = self.positions

		self.best_value_index = np.argmin(results[:,2])

		self.best_position = results[self.best_value_index][0:2]
		self.best_f = results[self.best_value_index][2]
		self.error = determine_error(self.true_position, self.best_position)

	def simulate_swarm(self):
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

if __name__ == "__main__":
	experiment = Experiment()