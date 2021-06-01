#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import versioneer
from setuptools import setup, find_packages

with open("docs/readme.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]

setup(
    author="Data Revenue GmbH",
    author_email="carlos@datarevenue.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Nested cross validation for feature selection in Python",
    install_requires=[
        "matplotlib>=3.3.2",
        "numpy>=1.19.2",
        "pandas>=1.1.3",
        "scipy>=1.5.2",
        "scikit-learn>=0.23.2",
        "xgboost>=1.3.1",
        "progressbar2>=3.53.1",
    ],
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="muvr, muv",
    name="py-muvr",
    packages=find_packages(exclude=["tests"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/datarevenue-berlin/py-MUVR",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
