# German Credit KNN Mini Project

This mini project uses the German Credit dataset from OpenML to build a classification model.

The goal is to predict whether a credit applicant is classified as good or bad credit risk.

## Dataset

Dataset: German Credit / credit-g  
OpenML link: https://www.openml.org/d/31

The dataset file used in this project is:

```text
dataset_31_credit-g.arff
```

## Project Instructions Covered

This project follows the required mini-project instructions:

1. Download and read the dataset.
2. Load the dataset into a Pandas DataFrame.
3. Check for missing values and drop them if needed.
4. Select relevant numeric and nominal features.
5. Use at least 4 numeric features.
6. Use at least 3 nominal features.
7. Preprocess the selected features.
8. Scale numeric features.
9. Encode nominal features.
10. Split the data into:
    - 80% training set
    - 10% validation set
    - 10% test set
11. Train a KNN classifier.
12. Try different values of k.
13. Choose the best k using validation accuracy.
14. Print the final test accuracy.
15. Print the confusion matrix.
16. Optional: Train another classifier, Random Forest.

## Selected Numeric Features

```text
duration
credit_amount
installment_commitment
residence_since
age
existing_credits
num_dependents
```

## Selected Nominal Features

```text
checking_status
credit_history
purpose
savings_status
employment
personal_status
property_magnitude
housing
job
```

## Target Variable

```text
class
```

The target variable has two classes:

```text
good
bad
```

## Preprocessing

The preprocessing step uses `StandardScaler` for numeric features and `OneHotEncoder` for nominal features.

## Model

The main required model is:

```text
K-Nearest Neighbors Classifier
```

The value of `k` is selected by testing multiple values and checking validation accuracy.

The optional second model is:

```text
Random Forest Classifier
```

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Python file:

```bash
python train_credit_g.py
```

## Output

The code prints:

1. Dataset shape
2. First five rows
3. Missing values
4. Selected numeric and nominal features
5. Training, validation, and test set sizes
6. Validation accuracy for different k values
7. Best k
8. Final KNN test accuracy
9. Confusion matrix
10. Classification report
11. Random Forest results
