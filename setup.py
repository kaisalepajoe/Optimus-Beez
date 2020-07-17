import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
	long_description = fh.read()

setup(
	name = 'optimusbeez',
	version = '0.0.7',
	description = 'Simple Particle Swarm Optimization',
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	url='https://github.com/kaisalepajoe/Optimus-Beez',
	author='Kaisa Lepajõe',
	author_email='kaisa.lepajoe@gmail.com',
	license='MIT',
	packages=setuptools.find_packages(),
	include_package_data=True,
	package_data={'optimusbeez': ['optimusbeez/optimal_constants.txt',
	'optimusbeez/function_info.txt']},
	classifiers=['Programming Language :: Python :: 3'],
	install_requires=[
		'numpy>=1.18.1',
		'matplotlib>=3.1.3',
		],
	test_suite='nose.collector',
	tests_require=['nose']
	
	)