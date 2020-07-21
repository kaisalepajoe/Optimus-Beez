import numpy as np

# Evaluate the required function

def evaluate(pos, fn_name):
	if type(pos) != np.ndarray and type(pos) != tuple and type(pos) != list:
		raise TypeError(f"Invalid type {type(pos)} for position")

	dim = len(pos)

	if type(fn_name) == str:
		if fn_name == "Rosenbrock":
			f = 0
			for d in range(dim-1):
				f += 100*(pos[d+1]-pos[d]**2)**2 + (1-pos[d])**2

		elif fn_name == "Alpine":
			f = 0
			for d in range(dim):
				f += abs(pos[d]*np.sin(pos[d]) + 0.1*pos[d])

		elif fn_name == "Griewank":
			f = 1
			for d in range(dim):
				f += 1/4000*pos[d]**2
			to_multiply = -np.inf*np.ones(dim)
			for d in range(1, dim+1):
				to_multiply[d-1] = np.cos(pos[d-1]/np.sqrt(d))
			product = np.prod(to_multiply)
			f = f - product
		else:
			raise ValueError(f"Function {fn_name} not defined")

		return f
	else:
		raise TypeError(f"Invalid type {type(fn_name)} for function name")