import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def make_dataset(samples: int = 500, features: int = 12):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(samples, features))
    y = (X[:, 1] - 0.4 * X[:, 4] + 0.6 * X[:, 7] > 0).astype(int)
    return X, y

def main():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    rf = RandomForestClassifier(n_estimators=120, random_state=42)

    ada.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    ada_acc = accuracy_score(y_test, ada.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    print("Demo: Accident Prediction")
    print(f"AdaBoost Accuracy: {ada_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

if __name__ == "__main__":
    main()
