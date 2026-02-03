import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
DATA_PATH = "data/vdss_dataset.csv"   # must contain features + target
TARGET_COL = "VDss"                   # change if needed
MODEL_OUT = "vdss_rdkit_macss_cnn_fixed.keras"
EPOCHS = 40
BATCH_SIZE = 32
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL]).values

# -------------------------
# Scale features
# -------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler (IMPORTANT for inference)
os.makedirs("artifacts", exist_ok=True)
np.save("artifacts/feature_mean.npy", scaler.mean_)
np.save("artifacts/feature_scale.npy", scaler.scale_)

# CNN expects (features, 1)
X = X[..., np.newaxis]

# -------------------------
# Train / test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# =========================
# MODEL (Keras-3 SAFE)
# =========================
model = models.Sequential([
    layers.Input(shape=(X.shape[1], 1)),  # 🔑 NO batch_shape

    layers.Conv1D(32, kernel_size=5, activation="relu"),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(64, kernel_size=3, activation="relu"),
    layers.GlobalMaxPooling1D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(1)  # regression
])

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="mse"
)

model.summary()

# =========================
# TRAIN
# =========================
cb = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    verbose=1
)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test).ravel()

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

# =========================
# SAVE MODEL (FIXED)
# =========================
model.save(MODEL_OUT)
print(f"\n✅ Model saved as: {MODEL_OUT}")
