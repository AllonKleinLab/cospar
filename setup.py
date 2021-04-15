from setuptools import setup, find_packages
from pathlib import Path

setup(
    name = "cospar",
    packages = ['cospar'],
    version = '0.1.3',
    python_requires=">=3.6",
    install_requires=[
        l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    extras_require=dict(
        dev=["black==19.10b0", "pre-commit==2.5.1"],
        docs=[r for r in Path("docs/requirements.txt").read_text("utf-8").splitlines()],
    ),
    author = 'Shou-Wen Wang',
    author_email="shouwen_wang@hms.harvard.edu",
    description="CoSpar: integrating transcriptome and clonal information for dynamic inference",
    long_description=Path("pypi.rst").read_text("utf-8"),
    long_description_content_type='text/x-rst',
    license="BSD",
    url='https://github.com/AllonKleinLab/cospar',
    download_url='https://github.com/AllonKleinLab/cospar',
    keywords=[
        "dynamic inference",
        "lineage tracing",
        "single cell",
        "transcriptomics",
        "differentiation"
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
