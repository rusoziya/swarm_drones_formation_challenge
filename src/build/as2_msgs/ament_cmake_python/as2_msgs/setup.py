from setuptools import find_packages
from setuptools import setup

setup(
    name='as2_msgs',
    version='1.1.2',
    packages=find_packages(
        include=('as2_msgs', 'as2_msgs.*')),
)
