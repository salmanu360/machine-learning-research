import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_dataset(samples: int = 300, features: int = 10):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(samples, features))
    y = (X[:, 0] + X[:, 1] - X[:, 4] > 0).astype(int)

    missing_mask = rng.random(X.shape) < 0.1
    X[missing_mask] = np.nan
    return X, y

def main():
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_imputed, y_train)
    preds = model.predict(X_test_imputed)

    print("Demo: Missing Values and Mean Imputation")
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))

if __name__ == "__main__":
    main()
