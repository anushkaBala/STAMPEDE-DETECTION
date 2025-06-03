import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load CSV data
data = pd.read_csv("crowd_data.csv")

# Simulate PoseVariance if not present in the dataset
if "PoseVariance" not in data.columns:
    # Simulate pose variance (e.g., random values between 0 and 1 for demonstration)
    np.random.seed(42)
    data["PoseVariance"] = np.random.uniform(0, 1, size=len(data))

# Features and target
X = data[["Density", "Speed", "PoseVariance"]].values  # Updated input features
y = data["Behavior"].values  # Target (behavior labels)

# Encode behavior labels (e.g., Calm=0, Dispersing=1, Aggressive=2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Create sequences for LSTM (e.g., use past 10 frames to predict next behavior)
sequence_length = 10
X_sequences = []
y_sequences = []

for i in range(len(X) - sequence_length):
    X_sequences.append(X[i:i + sequence_length])
    y_sequences.append(y_encoded[i + sequence_length])

X_sequences = np.array(X_sequences)  # Shape: (samples, sequence_length, 3)
y_sequences = np.array(y_sequences)  # Shape: (samples,)

# Split into train and test sets (80% train, 20% test)
split = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:split], X_sequences[split:]
y_train, y_test = y_sequences[:split], y_sequences[split:]

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("label_encoder_classes.npy", label_encoder.classes_)