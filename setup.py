from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

with open("LICENSE.txt") as f:
    license = f.read()
    
with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    name="Neuromatch-AJILE12",
    version="1.0.0",
    description="A package for exploration of AJILE12 data",
    author="Nima Dehghani",
    author_email="nima.dehghani@mit.edu",
    url="https://github.com/neurovium/Neuromatch-AJILE12",
    packages=["plot_utils"],
    install_requires=required,
)