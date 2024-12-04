## add parent directory to python path
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = f'{parentdir}/..'
sys.path.insert(0, parentdir) 