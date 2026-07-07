import runpy
from setuptools import setup, find_packages

__version__ = runpy.run_path('matchmaker/__version__.py')['__version__']

requires = [
    'click',
    'cvxpy',
    'itk-elastix',
    'matplotlib',
    'numpy',
    'open3d',
    'pandas',
    'probreg',
    'python-elf',
    'pyyaml',
    'scikit-image',
    'scipy',
    'seaborn',
    'tifffile',
    'transforms3d',
    'z5py',
]

# NOTE: `mobie_utils` (imported as `mobie` in matchmaker/mobie_export.py) has no
# PyPI release; install it via conda when using the mobie export functionality:
#   conda install -c conda-forge mobie_utils

setup(
    name='matchmaker',
    version=__version__,
    description='Registration of instance segmentations .',
    url='https://github.com/kreshuklab/matchmaker/',
    packages=find_packages(include=['matchmaker']),
    python_requires='>=3.6',
    install_requires=requires,
    author='Elena Buglakova',
    author_email='elena.buglakova@embl.de',
    license='MIT'
)
