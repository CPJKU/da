#!/usr/bin/env python

from io import open
from setuptools import setup


setup(
    name="da",
    version="0.0.2",
    description="Pytorch domain adaptation package",
    author="Schmid, F. and Masoudian, S.",
    author_email="florian.schmid@jku.at, shahed.masoudian@jku.at",
    url="https://github.com/CPJKU/da",
    license="GPLv2",
    packages=["da"],
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords=["deep learning", "pytorch", "AI", "domain adaptation"],
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=["numpy>=1.21.5", "POT>=0.8.0", "scipy>=1.7.3", "torch>=1.10.1"],
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
