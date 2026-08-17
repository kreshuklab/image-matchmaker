from setuptools import setup, find_packages

setup(
    name='image_matchmaker',
    version='0.1',
    description='Registration of instance segmentations .',
    url='https://github.com/kreshuklab/image-matchmaker',
    packages=find_packages(include=['image_matchmaker', 'image_matchmaker.*']),
    # Loaded at runtime relative to __file__, so they have to ship with the package
    package_data={
        'image_matchmaker': ['*.txt'],
        'image_matchmaker.cpd_parameter_tuning': ['*.yaml'],
    },
    python_requires='>=3.6',
    install_requires=[],
    author='Elena Buglakova',
    author_email='elena.buglakova@embl.de',
    license='MIT'
)
