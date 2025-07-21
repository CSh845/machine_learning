import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Load the CSV file
input_csv = 'data.csv'  # Replace with your file path
df = pd.read_csv(input_csv)

# Standardize SMILES function
def standardize_smiles(smiles_list):
    standardized_smiles = []
    for smi in smiles_list:
        try:
            # Convert SMILES to Mol object
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"Invalid SMILES skipped: {smi}")
                standardized_smiles.append(None)
                continue

            # Use RDKit's standardization tool
            normalizer = rdMolStandardize.CleanupParameters()
            mol = rdMolStandardize.Cleanup(mol, normalizer)

            # Neutralize charges
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)

            # Get standardized SMILES
            smi_standardized = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            standardized_smiles.append(smi_standardized)

        except Exception as e:
            print(f"Failed to standardize SMILES '{smi}': {e}")
            standardized_smiles.append(None)

    return standardized_smiles

# Assuming the SMILES column is named 'smiles'
if 'SMILES' in df.columns:
    # Standardize SMILES column
    df['standardized_smiles'] = standardize_smiles(df['SMILES'])

    # Save the updated dataframe to a new CSV file
    output_csv = 'standardized_smiles_data.csv'
    df.to_csv(output_csv, index=False)
    print(f"Standardized SMILES saved to {output_csv}")
else:
    print("No 'smiles' column found in the CSV file.")
