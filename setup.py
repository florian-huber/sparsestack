#!/usr/bin/env python
import os
# read the contents of your README file
from pathlib import Path
from setuptools import find_packages, setup


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "sparsestack", "__version__.py")) as f:
    exec(f.read(), version)

setup(
    name="sparsestack",
    version=version["__version__"],
    description="Python library to handle stacks of sparse COO arrays efficiently.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Florian Huber",
    author_email="florian.huber@hs-duesseldorf.de",
    url="https://github.com/florian-huber/stacked-sparse-array",
    packages=find_packages(exclude=['*tests*']),
    package_data={"stacked-sparse-array": ["data/*.csv"]},
    license="MIT",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
    python_requires='>=3.7',
    install_requires=[
        "numba",
        "numpy",
        "scipy",
    ],
    extras_require={"dev": ["bump2version",
                            "decorator",
                            "isort>=5.1.0",
                            "pylint<2.12",
                            "prospector[with_pyroma]",
                            "pytest",
                            "pytest-cov",
                            "testfixtures",
                            "yapf",]},
)
