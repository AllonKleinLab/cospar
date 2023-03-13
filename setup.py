import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from cospar import __version__

setup(
    name="cospar",
    version=__version__,
    python_requires=">=3.6",
    install_requires=[
        l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    extras_require=dict(
        dev=["black==19.10b0", "pre-commit==2.5.1"],
        docs=[r for r in Path("docs/requirements.txt").read_text("utf-8").splitlines()],
    ),
    packages=find_packages(),  # this is better than packages=["cospar"], which only include the top level files
    long_description_content_type="text/x-rst",
    author="Shou-Wen Wang",
    author_email="shouwen_wang@hms.harvard.edu",
    description="CoSpar: integrating state and lineage information for dynamic inference",
    long_description=Path("pypi.rst").read_text("utf-8"),
    license="BSD",
    url="https://github.com/ShouWenWang-Lab/cospar",
    download_url="https://github.com/ShouWenWang-Lab/cospar",
    keywords=[
        "dynamic inference",
        "lineage tracing",
        "single cell",
        "transcriptomics",
        "differentiation",
    ],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
