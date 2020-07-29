import optimusbeez as ob 
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm
import io
import sys

# Test experiment configuration and plot histogram of results
def evaluate_experiment(experiment, n_iterations):
	results = np.inf*np.ones(n_iterations)
	for n in tqdm(range(n_iterations), desc='Testing function:'):

		# Disable printing
		experiment.disable_progress_bar = True
		text_trap = io.StringIO()
		sys.stdout = text_trap

		experiment.run(200)
		results[n] = experiment.best_f

	# Restore printing
	sys.stdout = sys.__stdout__

	print("Experiment evaluation finished.")
	print(f"The mean result is {np.average(results)}")

	plt.hist(results, 100, (0, 10))
	plt.show()

if __name__=='__main__':
	experiment = ob.Experiment()
	test_function(experiment,1000)