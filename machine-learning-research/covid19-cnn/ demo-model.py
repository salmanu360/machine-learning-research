import numpy as np

def build_demo_dataset(samples: int = 120, height: int = 32, width: int = 32):
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(samples, height, width, 1))
    y = rng.integers(0, 2, size=(samples,))
    return X, y

def main():
    X, y = build_demo_dataset()
    print("Demo: CNN-based COVID-19 Classification")
    print("Synthetic image tensor shape:", X.shape)
    print("Synthetic labels shape:", y.shape)
    print("This demo intentionally avoids heavy framework dependencies.")
    print("You can later replace this with TensorFlow or PyTorch training code.")

if __name__ == "__main__":
    main()
