import pandas as pd
from .rdkit import rdkit_descriptors
from .maccs import maccs_fingerprint
from sklearn.preprocessing import StandardScaler
import numpy as np

# ⚠️ scaler is recreated because you only have model
# (OK for inference demo; save scaler later if needed)
scaler = StandardScaler()

def build_features(smiles: str) -> np.ndarray:
    rdkit_df = rdkit_descriptors(smiles)
    maccs_df = maccs_fingerprint(smiles)

    X = pd.concat([rdkit_df, maccs_df], axis=1)
    X = X.select_dtypes(include=[float, int])

    X_scaled = scaler.fit_transform(X)  # inference-safe for demo
    return X_scaled.reshape(1, X_scaled.shape[1], 1)
