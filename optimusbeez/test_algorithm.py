import optimusbeez as ob 
import matplotlib.pyplot as plt
import numpy as np 

# Test experiment configuration and plot histogram of results
def test_function(experiment, n_iterations):
	results = np.inf*np.ones(n_iterations)
	for n in range(n_iterations):
		experiment.run(1000)
		results[n] = experiment.best_f

	print("Function testing finished.")
	print(f"")

	plt.hist(results, 100)
	plt.show()

if __name__=='__main__':
	experiment = ob.Experiment({'phi': 2.518956142622854, 'k': 7, 'N': 9, 'time_steps': 10, 'repetitions': 1})
	test_function(experiment,500)