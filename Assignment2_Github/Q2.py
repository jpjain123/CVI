#Jayneel Pratap Jain    
#101983237
#jpjain@myseneca.ca
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

print("Loading MNIST data...")

train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

y_train_full = train_data.iloc[:, 0].values
X_train_full = train_data.iloc[:, 1:].values

y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values

print("Train samples:", len(X_train_full))
print("Test samples:", len(X_test))

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ]),
    "svm_rbf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf"))
    ]),
    "knn": KNeighborsClassifier(n_neighbors=3),
    "random_forest": RandomForestClassifier(n_estimators=200),
    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(max_iter=300))
    ])
}

best_acc = 0
best_model = None
best_name = ""

for name, model in models.items():
    print(f"\n=== Training: {name} ===")
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)
    print(f"Validation accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print("\nBest model:", best_name, "with accuracy:", best_acc)

print("\n=== Testing best model on test set ===")
y_pred_test = best_model.predict(X_test)


print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))
