from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

def rdkit_descriptors(smiles: str) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    desc = {name: func(mol) for name, func in Descriptors._descList}
    return pd.DataFrame([desc])
