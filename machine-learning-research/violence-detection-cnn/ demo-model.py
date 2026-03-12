import numpy as np

def generate_video_frame_features(samples: int = 200, frames: int = 10, features: int = 16):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(samples, frames, features))
    y = rng.integers(0, 2, size=(samples,))
    return X, y

def main():
    X, y = generate_video_frame_features()
    print("Demo: Automatic Violence Detection using Deep CNN")
    print("Synthetic sequence feature shape:", X.shape)
    print("Synthetic label shape:", y.shape)
    print("Replace this with TensorFlow/PyTorch sequence or frame classification later.")

if __name__ == "__main__":
    main()
