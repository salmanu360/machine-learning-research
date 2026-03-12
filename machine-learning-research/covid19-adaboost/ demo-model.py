import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def generate_demo_dataset(samples: int = 300, features: int = 20, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    X = rng.normal(0, 1, size=(samples, features))
    y = (X[:, 0] + 0.8 * X[:, 1] - 0.5 * X[:, 2] > 0).astype(int)
    return X, y

def main():
    X, y = generate_demo_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = AdaBoostClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Demo: COVID-19 Pneumonia Classification with AdaBoost")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()