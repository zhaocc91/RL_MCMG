chembl_path = 'chembl_wash_smiles.txt'
import random
import pandas as pd
from rdkit import Chem
max_length = 140
chose_num = 450000
smiles_s = []
with open(chembl_path,'r') as f:
    lines = f.readlines()
for line in lines:
    smiles = line.split('\n')[0]
    if len(smiles) > 15 and len(smiles) < max_length:
        smiles_s.append(smiles)

smiles_s = list(set(smiles_s))
chosed_smiles_1 = random.sample(smiles_s,1000000)

can_smis = []
for smi in chosed_smiles_1:
    try:
        can_s = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if len(can_s)> max_length:
            print('the smiles is too long:',can_s)
            continue
        can_smis.append(can_s)
    except:
        print('error smiles:',smi)

chosed_smiles_2 = list(set(can_smis))
chosed_smiles = random.sample(chosed_smiles_2,chose_num)

data = pd.DataFrame(chosed_smiles,columns=['SMILES'])
data.to_csv('random_filter_ChEMBL.csv', index=False)
