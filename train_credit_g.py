import pandas as pd
import numpy as np

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 
# STEP 1 + STEP 2: Read the dataset into a Pandas DataFrame
# 

DATA_PATH = "dataset_31_credit-g.arff"

data, meta = arff.loadarff(DATA_PATH)
df = pd.DataFrame(data)

# ARFF files load text columns as bytes.
# This converts byte columns into normal strings.
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.decode("utf-8")

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isna().sum())

# Drop missing values if any exist
df = df.dropna()

print("\nShape after dropping missing values:")
print(df.shape)

print("\nTarget class distribution:")
print(df["class"].value_counts())


# 
# STEP 3: Feature Selection
# Requirement:
# At least 4 numeric features and at least 3 nominal features
# 

numeric_features = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents"
]

nominal_features = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "property_magnitude",
    "housing",
    "job"
]

target = "class"

X = df[numeric_features + nominal_features]
y = df[target]

print("\nSelected numeric features:")
print(numeric_features)

print("\nSelected nominal features:")
print(nominal_features)

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# 
# STEP 4: Preprocessing
# Scaling numeric features
# Encoding nominal features
# 

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), nominal_features)
    ]
)


# 
# STEP 5: Split the data
# 80% training set
# 10% validation set
# 10% test set
# 

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining set:")
print(X_train.shape, y_train.shape)

print("\nValidation set:")
print(X_val.shape, y_val.shape)

print("\nTest set:")
print(X_test.shape, y_test.shape)


# 
# STEP 6: Train KNN classifier
# Choose the best k using the validation set
# 

k_values = list(range(1, 32, 2))
validation_scores = []

for k in k_values:
    knn_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=k))
        ]
    )

    knn_model.fit(X_train, y_train)

    y_val_pred = knn_model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)

    validation_scores.append(val_accuracy)

    print(f"k = {k:2d} | validation accuracy = {val_accuracy:.4f}")

best_index = int(np.argmax(validation_scores))
best_k = k_values[best_index]
best_validation_accuracy = validation_scores[best_index]

print("\nBest k:")
print(best_k)

print("\nBest validation accuracy:")
print(round(best_validation_accuracy, 4))


# 
# Final KNN model using best k
# 

final_knn_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k))
    ]
)

final_knn_model.fit(X_train, y_train)

y_test_pred = final_knn_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nFinal KNN Test Accuracy:")
print(round(test_accuracy, 4))

print("\nConfusion Matrix with labels ['bad', 'good']:")
print(confusion_matrix(y_test, y_test_pred, labels=["bad", "good"]))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


# 
# STEP 7 Optional: Train another model
# Random Forest classifier
# 

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

rf_model.fit(X_train, y_train)

rf_val_pred = rf_model.predict(X_val)
rf_test_pred = rf_model.predict(X_test)

print("\nRandom Forest Validation Accuracy:")
print(round(accuracy_score(y_val, rf_val_pred), 4))

print("\nRandom Forest Test Accuracy:")
print(round(accuracy_score(y_test, rf_test_pred), 4))

print("\nRandom Forest Confusion Matrix with labels ['bad', 'good']:")
print(confusion_matrix(y_test, rf_test_pred, labels=["bad", "good"]))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_test_pred))
