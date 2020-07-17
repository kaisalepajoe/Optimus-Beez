# This module optimizes the constants for the PSO algorithm

# Import required modules
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#########################################################################

# Return dictionary of random parameters according to 
# required time steps and allowed deviation of this number
# Required parameters are N, time_steps, repetitions, k, phi
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

def find_optimal_constants(tests, tests_with_each_constants, allowed_time_steps, allowed_deviation):

	# Read function info from txt file
	function_info = PSO.read_dictionary_from_file("function_info.txt")

	best_error = np.inf

	for t in range(tests):
		print(f"Test {t+1}/{tests}")
		constants = set_random_constants(allowed_time_steps, allowed_deviation)

		errors = np.inf*np.ones(tests_with_each_constants)

		# Repeat several times for this constants configuration		
		for constants_repetition in range(tests_with_each_constants):
			swarm = PSO.Swarm(constants, function_info)
			swarm.distribute_swarm()
			swarm.run_algorithm()
			errors[constants_repetition] = swarm.error

		avg_error = np.average(errors)

		if avg_error < best_error:
			best_constants = constants
			best_error = avg_error


	print("The best found constants configuration is:")
	print(best_constants)
	print(f"This configuration has the error: {best_error}")

	print("Save this configuration? This will overwrite optimal_constants.txt")
	print("Write yes or no:")
	answer = input()
	if answer == "yes":
		PSO.write_dictionary_to_file(best_constants, "optimal_constants.txt")

	return best_constants, best_error

###################################################################

if __name__ == "__main__":

	print("Set number of tests:")
	tests = int(input())
	print("Set number of repetitions for each constants configuration:")
	tests_with_each_constants = int(input())
	print("Set allowed time_steps:")
	allowed_time_steps = int(input())
	print("Set allowed deviation from time steps:")
	allowed_deviation = int(input())

	print("Finding optimal constants...")
	best_constants, best_error = find_optimal_constants(tests, tests_with_each_constants,
		allowed_time_steps, allowed_deviation)