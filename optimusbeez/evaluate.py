import numpy as np

# Evaluate the required function

def Rosenbrock(pos):
	dim = len(pos)
	# Change the Rosenbrock parameters a and b
	a = 1
	b = 100

	f = 0
	for d in range(dim-1):
		f += b*(pos[d+1]-pos[d]**2)**2 + (a-pos[d])**2

	return f

def Alpine(pos):
	dim = len(pos)
	f = 0
	for d in range(dim):
		f += abs(pos[d]*np.sin(pos[d]) + 0.1*pos[d])
	return f

def Griewank(pos):
	dim = len(pos)
	f = 1
	for d in range(dim):
		f += 1/4000*pos[d]**2
	to_multiply = -np.inf*np.ones(dim)
	for d in range(1, dim+1):
		to_multiply[d-1] = np.cos(pos[d-1]/np.sqrt(d))
	product = np.prod(to_multiply)
	f = f - product
	return f
