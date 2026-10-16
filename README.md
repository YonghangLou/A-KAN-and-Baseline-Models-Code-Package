# A-KAN and Baseline Models Code Package

## Contents

- `A-KAN.py`: implementation of the proposed A-KAN model
- `KAN.py`: baseline Kolmogorov-Arnold Network implementation
- `SVM.py`: support vector machine baseline
- `CNN.py`: convolutional neural network baseline
- `efficiency_utils.py`: shared efficiency utilities used by all four model scripts
- `requirements.txt`: Python dependencies used by the code
- `LICENSE`: terms for reuse and redistribution

## Data

According to the confidentiality agreement, the data cannot be published publicly. Thus, the current package does not include the dataset file.

To run the code, place `dataset.csv` in this directory. Each model script checks
for this file and reports a clear error if it is absent.

## Usage

Install the dependencies and run one model at a time:

```bash
python -m pip install -r requirements.txt
python A-KAN.py
python KAN.py
python SVM.py
python CNN.py
```

All scripts use a fixed random seed (`42`) and the same data-preparation logic
to support reproducible comparisons. Model hyperparameters are defined in each
script's main block.

## Outputs

Running a model script creates an `outputs` directory inside this folder and saves:

- evaluation metrics
- efficiency metrics
- prediction results
- figures generated during training and testing

## License

The source code in this package is released under the MIT License. See
`LICENSE` for the complete terms.

