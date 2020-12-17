#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("tianshou", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


setup(
    name="tianshou",
    version=get_version(),
    description="A Library for Deep Reinforcement Learning",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thu-ml/tianshou",
    author="TSAIL",
    author_email="trinkle23897@gmail.com",
    license="MIT",
    python_requires=">=3.6",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="reinforcement learning platform pytorch",
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    install_requires=[
        "gym>=0.15.4",
        "tqdm",
        "numpy!=1.16.0",  # https://github.com/numpy/numpy/issues/12793
        "tensorboard",
        "torch>=1.4.0",
        "numba>=0.51.0",
        "h5py>=3.1.0"
    ],
    extras_require={
        "dev": [
            "Sphinx",
            "sphinx_rtd_theme",
            "sphinxcontrib-bibtex",
            "flake8",
            "pytest",
            "pytest-cov",
            "ray>=1.0.0",
            "mypy",
            "pydocstyle",
            "doc8",
        ],
        "atari": ["atari_py", "cv2"],
        "mujoco": ["mujoco_py"],
        "pybullet": ["pybullet"],
    },
)
