from setuptools import setup, find_packages

setup(
    name="Neuromatch-AJILE12",
    version="1.0.0",
    description="A package for exploration of AJILE12 data",
    author="Nima Dehghani",
    author_email="nima.dehghani@mit.edu",
    url="https://github.com/neurovium/Neuromatch-AJILE12",
    packages=["plot_utils", "spec_utils"],
    install_requires=requirements,
)
