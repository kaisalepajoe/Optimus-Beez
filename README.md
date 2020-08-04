# Optimus-Beez

This is a Particle Swarm Optimization (PSO) package. The PSO used is the simplest version presented by Maurice Clarc in "Particle Swarm Optimization" with some minor modifications.

# Installation

Run the following command:
<code>pip install optimusbeez</code>
Make sure you have installed the latest version. Old versions may be faulty.

# How to use Optimus-Beez

## Choosing the function to evaluate

The default function to evaluate is Rosenbrock. To change this, first check out evaluate.py. This file contains predefined functions including Rosenbrock, Alpine, Griewank, and a flat surface. If the function you wish to use is not defined, then go ahead and add it to evaluate.py. Then go to function_info.txt and
- change fn_name to the name of your function
- set the search space with xmin and xmax
Note that fn_info and constants are both dictionaries. To get information about the other keys in the fn_info dictionary, use help() on the Experiment object of optimusbeez.
If you define your own function, you must also import it in PSO.py.

## Creating an Experiment

Optimusbeez has an Experiment class. The first steps to using the optimusbeez package are
<code>import optimusbeez as ob
experiment = ob.Experiment()</code>
If no arguments are passed to the Experiment object, then it is created with default parameters (hereafter referred to as constants) from the file 'optimal_constants.txt' and function info from 'fn_info.txt'. You can easily change these after creating the experiment object. For example,
<code>experiment.N = 20</code>
changes the number of particles in the swarm to 20. You can also change the evaluation function.
<code>experiment.fn_name = 'Alpine'</code>
To see the current configuration of constants and function info, you can use
<code>experiment.constants()
experiment.fn_info()</code>

## Running the Experiment

To evolve the swarm through time, you must run the experiment.
<code>experiment.run(1000)</code>
The argument passed to the run function is the number of evaluations. The experiment will run and show a progress bar. If show_animation is set to True in fn_info, then an animation of the swarm will be shown at the end of the run. Results will be printed on the screen as well as returned in the format (best found position, value at that position, difference from optimal_f). optimal_f is the expected minimum value of the function, usually 0. It is defined in the fn_info dictionary.

## Running the Experiment several times

The function evaluate_experiment() is very useful to gauge the constants configuration and how the optimization algorithm fares in general. This function takes 3 arguments - an experiment object, the number of evaluations, and the number of times the experiment should be run. The function returns the average best value and its standard deviation. It also plots a histogram of the results from 0 to 10.

## Optimizing the Experiment constants

The PSO algorithm itself has 5 parameters: phi, N, k, time steps, and repetitions. These can be changed manually but Optimusbeez can also optimize itself with the function optimize_constants(). This function takes 4 arguments - allowed evaluations, allowed deviation from the number of evaluations, optimization time steps, and optimization repetitions. This function creates a swarm of particles with a 5D array of positions corresponding to the 5 constants. It runs an experiment with these constants and moves the swarm towards the best constants configuration. Usually it does a pretty good job, but it is better to test the constants configuration afterwards and adjust the constants manually.

## Testing

Nose is used to test the code. All tests are located in the 'tests' folder.To run the tests, execute:

<code> nosetests</code>