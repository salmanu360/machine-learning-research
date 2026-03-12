import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

def build_data(samples: int = 350, features: int = 18):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(samples, features))
    y = ((X[:, 2] + X[:, 5] - X[:, 9]) > 0.2).astype(int)
    return X, y

def main():
    X, y = build_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = AdaBoostClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Demo: Malicious Website Prediction")
    print("Precision:", round(precision_score(y_test, preds), 4))
    print("Recall:", round(recall_score(y_test, preds), 4))

if __name__ == "__main__":
    main()
