from setuptools import setup 

setup(
	name = 'optimus-beez',
	version = '0.0.1',
	description = 'Simple Particle Swarm Optimization',
	py_modules=['evaluate', 'PSO', 'optimize_constants'],
	package_dir={'': 'src'},
	classifiers=[
		'Programming Language :: Python :: 3',
		]
	)