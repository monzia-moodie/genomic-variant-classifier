# Genomic Variant Classifier
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
An ensemble machine learning system for classifying pathogenic genetic variants
using data from ClinVar, gnomAD, and UniProt databases.
## Project Overview
- **Integrating** variant data from multiple biomedical databases
- **Engineering** features aligned with ACMG classification criteria
- **Training** an ensemble of gradient boosting, random forest, and neural network models
- **Evaluating** with clinical utility metrics and calibration analysis
## Quick Start
```bash
git clone https://github.com/monzia-moodie/genomic-variant-classifier.git
cd genomic-variant-classifier
pip install -r requirements.txt
pip install -e .
```
## Author
**Monzia Moodie** - [@monzia-moodie](https://github.com/monzia-moodie)
## License
MIT License
