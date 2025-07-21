# utils/data_loader.py

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_fingerprints(filepath, fingerprint_type='MACCS'):
    df = pd.read_csv(filepath)
    smiles_list = list(df.iloc[:, 0])
    
    # Create molecules and filter out invalid ones
    mols = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_indices.append(i)
        else:
            print(f"Warning: Could not parse SMILES string: {smi}")
    
    # Generate fingerprints only for valid molecules
    if fingerprint_type == 'MACCS':
        X = np.array([AllChem.GetMACCSKeysFingerprint(mol) for mol in mols])
    elif fingerprint_type == 'Morgan':
        X = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024) for mol in mols])
    elif fingerprint_type == 'RDKit':
        X = np.array([Chem.RDKFingerprint(mol) for mol in mols])
    elif fingerprint_type == 'TopologicalTorsion':
        X = np.array([AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol) for mol in mols])
    elif fingerprint_type == 'AtomPairsFP':
        X = np.array([AllChem.GetHashedAtomPairFingerprintAsBitVect(mol) for mol in mols])
    else:
        raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")

    # Filter labels to match valid molecules
    y = df['Label'].values[valid_indices]
    
    print(f"Processed {len(mols)} valid molecules out of {len(smiles_list)} total.")
    
    return X, y


def load_descriptors(descriptors_filepath, max_value=1e10, scale=False):
    df = pd.read_csv(descriptors_filepath)


    if 'SMILES' not in df.columns or 'Label' not in df.columns:
        raise ValueError("descriptors.csv 必须包含 'SMILES' 和 'Label' 列。")

 
    df.replace([np.inf, -np.inf], np.nan, inplace=True)


    initial_len = len(df)
    df.dropna(inplace=True)
    final_len = len(df)
    if initial_len != final_len:
        print(f"信息：已删除 {initial_len - final_len} 个包含 NaN 或无限值的样本。")


    X = df.iloc[:, 2:].values.astype(np.float64)
    y = df['Label'].values

    max_abs_val = np.max(np.abs(X))
    if max_abs_val > max_value:
        print(f"警告：描述符数据的最大绝对值 {max_abs_val} 超过了 {max_value}，正在进行裁剪。")
        X = np.clip(X, -max_value, max_value)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("信息：描述符数据已标准化。")

    return X, y
