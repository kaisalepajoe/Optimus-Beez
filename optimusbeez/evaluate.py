import numpy as np

# Evaluate the required function

def evaluate(pos, fn_name):
	if type(pos) == np.ndarray or type(pos) == tuple or type(pos) == list:
		if len(pos) == 2:
			if type(fn_name) == str:
				x = pos[0]
				y = pos[1]
				if fn_name == "Rosenbrock":
					f = (1-x)**2 + 100*(y-x**2)**2

				elif fn_name == "Alpine":
					f = abs(x*np.sin(x) + 0.1*x) + \
						abs(y*np.sin(y) + 0.1*y)

				elif fn_name == "Griewank":
					f = 1 + 1/4000*x**2 + 1/4000*y**2 \
						-np.cos(x)*np.cos(0.5*y*np.sqrt(2))

				elif fn_name == "Ackley":
					f = -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))\
						-np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))\
						+ np.exp(1)+20

				else:
					raise ValueError(f"Function {fn_name} not defined")

				return f
			else:
				raise TypeError(f"Invalid type {type(fn_name)} for function name")
		else:
			raise ValueError(f"Invalid length {len(pos)} for position")
	else:
		raise TypeError(f"Invalid type {type(pos)} for position")