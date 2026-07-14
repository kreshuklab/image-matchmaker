from setuptools import setup, find_packages

setup(
    name='image_matchmaker',
    version='0.1',
    description='Registration of instance segmentations .',
    url='https://github.com/kreshuklab/matchmaker/',
    packages=find_packages(include=['image_matchmaker']),
    python_requires='>=3.6',
    install_requires=[],
    author='Elena Buglakova',
    author_email='elena.buglakova@embl.de',
    license='MIT'
)