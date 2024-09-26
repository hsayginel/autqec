from setuptools import setup, find_packages

setup(
    name='autqec',
    version='1.0.0',
    packages=find_packages(),
    author='Hasan Sayginel',
    author_email='hasan.sayginel.17@ucl.ac.uk',
    description='Clifford gates from code automorphisms.',
    install_requires=[
        'numpy==1.26.2',
        'pytest==8.3.1',
        'qiskit==1.1.1',
        'matplotlib==3.9.1',
        'pylatexenc==2.10',
        'numba==0.60.0'
    ],)

# MAGMA V2.28-8: http://magma.maths.usyd.edu.au/magma/. 