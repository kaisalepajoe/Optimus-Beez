# Particle Swarm Optimization
# This script finds the global MINIMUM of the
# selected function.

# This is the simplest version of PSO from
# the book "Particle Swarm Optimization" by
# Maurice Clerc.

# Import required modules
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set random seed
# np.random.seed(123)

###################################################################

# Initialize constants

initial_constants = {
	"N":9,					# Size of swarm
	"time_steps":50,			# Time steps for each repetition
	"repetitions":3,			# Number of repetitions
	"k":3,					# Number of informants for each particle
	"phi":2.2,				# Confidence constant, must be > 2
	"fn_name":"Rosenbrock",	# Name of function to be evaluated (Rosenbrock, Alpine or Griewank)
	"xmin":-100,				# Size of search field, minimum
	"xmax":100,				# Size of search field, maximum
	"show_animation":True		# Show animation of best rep
	}

# Determine number of repetitions given constants
def determine_n_evaluations(N, time_steps, repetitions):
	return N*time_steps*repetitions + repetitions*N

def set_constants(dictionary):
	# Get constants from dictionary
	global N, time_steps, repetitions, fn_name, k, phi
	global xmin, xmax, show_animation, vmax, c1, cmax
	N = dictionary["N"]
	time_steps = dictionary["time_steps"]
	repetitions = dictionary["repetitions"]
	fn_name = dictionary["fn_name"]
	k = dictionary["k"]
	phi = dictionary["phi"]
	xmin = dictionary["xmin"]
	xmax = dictionary["xmax"]
	show_animation = dictionary["show_animation"]
	
	# Calculate maximum velocity
	vmax = abs(xmax - xmin)/2

	# Calculate confidence parameters using phi
	c1 = 1/(phi-1+np.sqrt(phi**2-2*phi))
	cmax = c1*phi

###################################################################

# Helper functions

# Evaluate the required function
def evaluate(pos, fn_name):
	x = pos[0]
	y = pos[1]
	if fn_name == "Rosenbrock":
		f = (1-x)**2 + 100*(y-x**2)**2

	if fn_name == "Alpine":
		f = abs(x*np.sin(x) + 0.1*x) + \
			abs(y*np.sin(y) + 0.2*y)

	if fn_name == "Griewank":
		f = 1 + 1/4000*x**2 + 1/4000*y**2 \
			-np.cos(x)*np.cos(0.5*y*np.sqrt(2))

	return f

# Choose k informants randomly
def random_informants(particles):
	for particle in particles:
		particle.informants = np.random.choice(particles, k)

# Randomly assign confidence parameters
# c2 and c3 in the interval [0, cmax)
def random_confidence():
	c2 = np.array([np.random.uniform(0, cmax), \
		np.random.uniform(0, cmax)])
	c3 = np.array([np.random.uniform(0, cmax), \
		np.random.uniform(0, cmax)])
	return (c2, c3)

###################################################################

# Particle class
class Particle:

	def __init__(self, pos, vel, p):
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
		received = np.zeros((k, 3))
		for i, informant in enumerate(self.informants):
			received[i, :] = informant.g
		# Set g to LOWEST value
		i = np.argmin(received[:,2])
		self.g = received[i]

	def step(self):
		# Evaluate current position
		# Update p if current position is LOWER
		value = evaluate(self.pos, fn_name)
		if value < self.p[2]:
			self.p[2] = value
			self.p[0:2] = self.pos
		if value < self.g[2]:
			self.g[2] = value
			self.g[0:2] = self.pos

		# Communicate with informants, update g
		self.communicate()

		# Set confidence parameters
		c2, c3 = random_confidence()

		# Update velocity
		possible_vel = c1*self.vel + \
			c2*(self.p[0:2] - self.pos) + \
			c3*(self.g[0:2] - self.pos)
		# Constrain velocity
		for d in range(2):
			if abs(possible_vel[d]) <= vmax:
				self.vel[d] = possible_vel[d]
			elif possible_vel[d] > vmax:
				self.vel[d] = vmax
			elif possible_vel[d] < -vmax:
				self.vel[d] = -vmax

		# Update position
		possible_pos = self.pos + self.vel
		# Constrain particle to search area
		# Set velocity to 0 if possible_pos
		# outside search area to avoid touching
		# the boundary again in the next time step.
		for d in range(2):
			if xmin <= possible_pos[d] <= xmax:
				self.pos[d] = possible_pos[d]
			elif possible_pos[d] < xmin:
				self.pos[d] = xmin
				self.vel[d] = 0
			elif possible_pos[d] > xmax:
				self.pos[d] = xmax
				self.vel[d] = 0

###################################################################

# Initialise N particles with positions, velocities, p-values
def create_swarm():
	# Create array of random positions
	initial_positions = np.random.uniform(xmin, xmax, (N, 2))
	
	# Evaluate positions for initial p values
	p_values = np.inf*np.ones((N, 3))
	for i, pos in enumerate(initial_positions):
		p_values[i,2] = evaluate(pos, fn_name)
		p_values[i,0:2] = pos

	# Create list of random velocities (up to limit)
	velocities = np.random.uniform(-vmax, vmax, (N, 2))

	# Create list of particle objects
	particles = []
	for i in range(N):
		particles.append(Particle(\
			initial_positions[i], velocities[i], p_values[i]))

	# Initiate k informants randomly
	random_informants(particles)

	# Initialise array of positions for animation
	positions = np.inf*np.ones((time_steps, N, 2))
	positions[0,:,:] = initial_positions

	return particles, positions

# Update positions of particles for all time steps
def evolve(particles, positions):
	for step in range(time_steps):
		for i, particle in enumerate(particles):
			particle.step()
			# Update positions for animation
			positions[step,i,:] = particle.pos
		# Select informants for next time step
		random_informants(particles)

# Extract optimal parameters (from g)
def get_parameters(particles):
	final_g = np.inf*np.ones((N, 3))
	for i,particle in enumerate(particles):
		final_g[i,:] = particle.g
	optimal_i = np.argmin(final_g[2])
	x = final_g[optimal_i][0]
	y = final_g[optimal_i][1]
	f = final_g[optimal_i][2]
	return np.array([x, y, f])

###################################################################

# Run the algorithm, return best position and value
def run_algorithm(constants):
	# Call set_constants function
	set_constants(constants)

	results = np.inf*np.ones((repetitions, 3))
	all_positions = np.inf*np.ones((repetitions, time_steps, N, 2))
	for r in range(repetitions):
		particles, positions = create_swarm()
		evolve(particles, positions)
		result = get_parameters(particles)
		results[r] = result
		all_positions[r] = positions
		print(f"Repetition {r+1} finished.")
		print(result)
		print()

	best_value_index = np.argmin(results[:,2])

	best_x = results[best_value_index][0]
	best_y = results[best_value_index][1]
	best_f = results[best_value_index][2]

	return best_x, best_y, best_f, all_positions, best_value_index

###################################################################

# Simulate particle swarm
def simulate_swarm(all_positions):
	if show_animation == False:
		exit()

	# Plot initial positions of particles
	fig, ax = plt.subplots()
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(xmin, xmax)
	scat = ax.scatter(all_positions[best_value_index,0,:,0], all_positions[best_value_index,0,:,1], color="Black", s=1.5)

	# Create animation
	interval = 200_000 / (N * time_steps * repetitions)
	global animation
	animation = FuncAnimation(fig, func=update_frames, interval=interval, fargs=[scat])
	plt.show()

# Required update function for simulation
def update_frames(j, *fargs):
	scat = fargs[0]
	try:
		scat.set_offsets(all_positions[best_value_index,j])
	except:
		print("Simulation finished")
		animation.event_source.stop()

###################################################################

# Determine average error (abs(true position - position))

x, y, f, all_positions, best_value_index = run_algorithm(initial_constants)

print(f"Minimum is {f} at: [{x}, {y}]")
print(f"{determine_n_evaluations(N, time_steps, repetitions)} evaluations made.")

if show_animation == False:
	exit()
else:
	simulate_swarm(all_positions)