import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def create_demo_data(samples: int = 400, features: int = 15):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(samples, features))
    y = ((X[:, 0] * 0.7) + (X[:, 3] * 0.5) - X[:, 5] > 0).astype(int)
    return X, y

def main():
    X, y = create_demo_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Demo: Pneumonia Classification with Random Forest")
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    main()
