from setuptools import find_packages
from distutils.core import setup

setup(name='unitree_rl_gym',
      version='1.0.0',
      author='Unitree Robotics',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='support@unitree.com',
      description='Template RL environments for Unitree Robots',
      extras_require={'sim2sim': ['pynput==1.7.7']},
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy>=1.23.5,<1.24', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
