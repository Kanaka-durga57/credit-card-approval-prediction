import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


print("==========================================")
print("CREDIT CARD RISK PREDICTION - MODEL TRAINING")
print("==========================================")


# ==========================================
# 1. LOAD PREPARED DATASET
# ==========================================

print("\nLoading prepared dataset...")

data = pd.read_csv(
    "database/credit_card_dataset.csv"
)

print("Dataset shape:", data.shape)


# ==========================================
# 2. REMOVE ID
# ==========================================

if "ID" in data.columns:
    data = data.drop("ID", axis=1)


# ==========================================
# 3. CHECK TARGET
# ==========================================

print("\nTarget distribution:")

print(
    data["CREDIT_RISK"].value_counts()
)


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

print("\nHandling missing values...")

categorical_columns = data.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_columns = data.select_dtypes(
    exclude=["object", "str"]
).columns.tolist()

for column in categorical_columns:

    data[column] = data[column].fillna(
        "Unknown"
    )

for column in numerical_columns:

    if column != "CREDIT_RISK":

        data[column] = data[column].fillna(
            data[column].median()
        )


# ==========================================
# 5. ENCODE CATEGORICAL FEATURES
# ==========================================

print("Encoding categorical features...")

label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    data[column] = encoder.fit_transform(
        data[column].astype(str)
    )

    label_encoders[column] = encoder


# ==========================================
# 6. SEPARATE FEATURES AND TARGET
# ==========================================

X = data.drop(
    "CREDIT_RISK",
    axis=1
)

y = data["CREDIT_RISK"]


print("\nNumber of features:", X.shape[1])
print("Number of records:", X.shape[0])


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# ==========================================
# 8. FEATURE SCALING
# ==========================================

print("\nScaling numerical features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# 9. LOGISTIC REGRESSION
# ==========================================

print("\n==========================================")
print("LOGISTIC REGRESSION")
print("==========================================")

logistic_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test_scaled
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    zero_division=0
)


print(
    "Accuracy :",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(logistic_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(logistic_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(logistic_f1 * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions,
        zero_division=0
    )
)


# ==========================================
# 10. DECISION TREE
# ==========================================

print("\n==========================================")
print("DECISION TREE")
print("==========================================")

decision_tree = DecisionTreeClassifier(
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)

tree_predictions = decision_tree.predict(
    X_test
)


tree_accuracy = accuracy_score(
    y_test,
    tree_predictions
)

tree_precision = precision_score(
    y_test,
    tree_predictions,
    zero_division=0
)

tree_recall = recall_score(
    y_test,
    tree_predictions,
    zero_division=0
)

tree_f1 = f1_score(
    y_test,
    tree_predictions,
    zero_division=0
)


print(
    "Accuracy :",
    round(tree_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(tree_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(tree_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(tree_f1 * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        tree_predictions,
        zero_division=0
    )
)


# ==========================================
# 11. RANDOM FOREST
# ==========================================

print("\n==========================================")
print("RANDOM FOREST")
print("==========================================")

random_forest = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest.fit(
    X_train,
    y_train
)

forest_predictions = random_forest.predict(
    X_test
)


forest_accuracy = accuracy_score(
    y_test,
    forest_predictions
)

forest_precision = precision_score(
    y_test,
    forest_predictions,
    zero_division=0
)

forest_recall = recall_score(
    y_test,
    forest_predictions,
    zero_division=0
)

forest_f1 = f1_score(
    y_test,
    forest_predictions,
    zero_division=0
)


print(
    "Accuracy :",
    round(forest_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(forest_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(forest_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(forest_f1 * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        forest_predictions,
        zero_division=0
    )
)


# ==========================================
# 12. MODEL COMPARISON
# ==========================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print("\nLogistic Regression")

print(
    "Accuracy :",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(logistic_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(logistic_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(logistic_f1 * 100, 2),
    "%"
)


print("\nDecision Tree")

print(
    "Accuracy :",
    round(tree_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(tree_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(tree_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(tree_f1 * 100, 2),
    "%"
)


print("\nRandom Forest")

print(
    "Accuracy :",
    round(forest_accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(forest_precision * 100, 2),
    "%"
)

print(
    "Recall   :",
    round(forest_recall * 100, 2),
    "%"
)

print(
    "F1 Score :",
    round(forest_f1 * 100, 2),
    "%"
)


# ==========================================
# 13. SELECT BEST MODEL
# ==========================================

# For this project, F1-score is more useful than
# accuracy because the dataset is imbalanced.

scores = {
    "Logistic Regression": logistic_f1,
    "Decision Tree": tree_f1,
    "Random Forest": forest_f1
}

best_model_name = max(
    scores,
    key=scores.get
)


if best_model_name == "Logistic Regression":

    best_model = logistic_model
    best_scaler = scaler

elif best_model_name == "Decision Tree":

    best_model = decision_tree
    best_scaler = None

else:

    best_model = random_forest
    best_scaler = None


print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print(
    "Selected model:",
    best_model_name
)

print(
    "F1 Score:",
    round(
        scores[best_model_name] * 100,
        2
    ),
    "%"
)


# ==========================================
# 14. SAVE MODEL
# ==========================================

print("\nSaving model...")

joblib.dump(
    best_model,
    "model/credit_card_model.pkl"
)


# ==========================================
# 15. SAVE SCALER
# ==========================================

joblib.dump(
    best_scaler,
    "model/scaler.pkl"
)


# ==========================================
# 16. SAVE ENCODERS
# ==========================================

joblib.dump(
    label_encoders,
    "model/label_encoders.pkl"
)


# ==========================================
# 17. SAVE FEATURE COLUMNS
# ==========================================

joblib.dump(
    list(X.columns),
    "model/feature_columns.pkl"
)


print("\n==========================================")
print("FILES SAVED SUCCESSFULLY")
print("==========================================")

print("model/credit_card_model.pkl")
print("model/scaler.pkl")
print("model/label_encoders.pkl")
print("model/feature_columns.pkl")

print("\n==========================================")
print("MODEL TRAINING COMPLETED")
print("==========================================")