#Jayneel Pratap Jain    
#101983237
#jpjain@myseneca.ca

import os
import numpy as np
import joblib
from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


IMG_SIZE = (128, 128)  
TRAIN_PATH = "train"
TEST_PATH = "test"
MODEL_PATH = "q1_best_cat_dog_model.joblib"

def load_images(folder):
    data = []
    labels = []

    for label in ["Cat", "Dog"]:
        class_dir = os.path.join(folder, label)
        print(f"Loading {label} from {class_dir} ...")

        for file in os.listdir(class_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img = imread(os.path.join(class_dir, file))

                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]

            
                img_resized = resize(img, IMG_SIZE, anti_aliasing=True)

                if img_resized.ndim == 3:
                    img_gray = np.dot(img_resized[..., :3], [0.299, 0.587, 0.114])
                else:
                    img_gray = img_resized

                data.append(img_gray.ravel())
                labels.append(label)

    return np.array(data), np.array(labels)

print("Loading training data...")
X_train, y_train = load_images(TRAIN_PATH)
print("Loading test data...")
X_test, y_test = load_images(TEST_PATH)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

pipelines = {
    "logreg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "svm_rbf": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf"))
    ]),
    "random_forest": Pipeline([
        ("clf", RandomForestClassifier())
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ]),
    "mlp": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(max_iter=500))
    ])
}


param_grids = {
    "logreg": {"clf__C": [0.1, 1, 10]},
    "svm_rbf": {"clf__C": [1, 10, 100], "clf__gamma": [1e-3, 1e-4]},
    "random_forest": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 20]},
    "knn": {"clf__n_neighbors": [3, 5, 7]},
    "mlp": {"clf__hidden_layer_sizes": [(128,), (256,)], "clf__alpha": [0.0001, 0.001]},
}



best_model = None
best_name = ""
best_acc = 0

for name in pipelines:
    print(f"\nTraining: {name}")

    grid = GridSearchCV(
        pipelines[name],
        param_grids[name],
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

  
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{name} best params: {grid.best_params_}")
    print(f"{name} test accuracy: {acc:.4f}")

    if acc > best_acc:
        best_model = grid.best_estimator_
        best_name = name
        best_acc = acc

print("\n=============================")
print(f"BEST MODEL: {best_name} (accuracy = {best_acc:.4f})")
print("Classification Report:")
print(classification_report(y_test, best_model.predict(X_test)))

# save model
joblib.dump(best_model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
