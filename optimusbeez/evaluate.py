import numpy as np

# Evaluate the required function

def evaluate(pos, fn_name):
	x = pos[0]
	y = pos[1]
	if fn_name == "Rosenbrock":
		f = (1-x)**2 + 100*(y-x**2)**2

	elif fn_name == "Alpine":
		f = abs(x*np.sin(x) + 0.1*x) + \
			abs(y*np.sin(y) + 0.2*y)

	elif fn_name == "Griewank":
		f = 1 + 1/4000*x**2 + 1/4000*y**2 \
			-np.cos(x)*np.cos(0.5*y*np.sqrt(2))
	else:
		print("Invalid function name!")

	return f