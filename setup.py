"""Setup script for geoscience-transformers package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="geoscience-transformers",
    version="0.1.0",
    author="Geoscience Transformers Team",
    description="Large Language Models and Geoscience Transformers for Predictive Mapping of Critical Minerals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RichardScottOZ/geoscience-transformers-for-predictive-mapping-of-critical-minerals",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.0.0",
        "geopandas>=0.10.0",
        "rasterio>=1.3.0",
        "shapely>=2.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "geoscience-train=geoscience_transformers.cli:train",
            "geoscience-predict=geoscience_transformers.cli:predict",
        ],
    },
)
