# Optimus-Beez

This is a Particle Swarm Optimization (PSO) package. The PSO used is the simplest version presented by Maurice Clarc in "Particle Swarm Optimization".

# Installation

Run the following command:
pip install optimus-beez

# How to use Optimus-Beez

## Choosing the function to evaluate

The default function to evaluate is Rosenbrock. To change this, first check out evaluate.py. This file contains evaluate() that evaluates points for different functions. If the function you wish to use is not defined, then go ahead and add it to evaluate(). Then go to function_info.txt and change
- fn_name to the name of one of the functions in evaluate()
- true_position to the x,y-coordinates of the global minimum of your function

## Optimizing the parameters of the PSO

The optimization algorithm itself contains 5 parameters that need to be set by the user. These are set in the file optimal_constants.txt. In the code, these parameters are referred to as 'constants' so as not to confuse them with the x,y-coordinates.
It is a good idea to optimize these constants using optimize_constants.py. Run optimize_constants.py on your command line. This is just good old random search optimization. You will be prompted several times for input. Use a value of 'time steps' similar to what you want to use with PSO.
When the random search is completed, you will be asked if you want to overwrite the file optimal_constants.txt. Do this if you would like to use these constants in PSO.py.

## Using PSO

The main script is PSO.py. Run this in the command line. You will be asked if you want to change the number of evaluations. If the value you wish to set is much larger or smaller than the default value, it is advised you run optimize_constants.py again. Set the value, or use the default value and wait for the PSO to finish. You will see an animation of your swarm at the end.

## Testing

Nose is used to test the code. All tests are located in the 'tests' folder.To run the tests, execute:

>>> nosetests