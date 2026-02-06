from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="genomic-variant-classifier",
    version="0.1.0",
    author="Monzia Moodie",
    author_email="monzia.moodie@gmail.com",
    description="Ensemble ML system for pathogenic variant classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/monzia-moodie/genomic-variant-classifier",
    packages=find_packages(),
    python_requires=">=3.10",
)
