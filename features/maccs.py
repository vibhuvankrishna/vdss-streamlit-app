from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pandas as pd

def maccs_fingerprint(smiles: str) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    fp = MACCSkeys.GenMACCSKeys(mol)
    bits = list(fp)
    return pd.DataFrame([bits], columns=[f"MACCS_{i}" for i in range(len(bits))])
