from setuptools import setup 

setup(
	name = 'optimus-beez',
	version = '0.0.1',
	description = 'Simple Particle Swarm Optimization',
	py_modules=['evaluate', 'PSO', 'optimize_constants'],
	package_dir={'': 'src'},
	include_package_data=True,
	classifiers=['Programming Language :: Python :: 3'],
	install_requires=[
		'numpy>=1.18.1',
		'matplotlib>=3.1.3'],
	url='https://github.com/kaisalepajoe/Optimus-Beez',
	author='Kaisa Lepajõe',
	author_email='kaisa.lepajoe@gmail.com'
	)