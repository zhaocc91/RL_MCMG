
from rdkit import Chem
import pandas as pd

prior_file_path = 'save_prior_generated_mols/prior_generated_mols.csv'
df = pd.read_csv(prior_file_path, sep=' ')
df_smiles = df.loc[ : ,"SMILES"]

def cal_valid_unique_smi(sml_s):
    can_smis = []
    num_sm = len(sml_s)
    invalid = 0
    for smi in sml_s:
        try:
            mol = Chem.MolFromSmiles(smi)
            can_smi = Chem.MolToSmiles(mol)
            can_smis.append(can_smi)
        except:
            invalid+=1
    precent_valid = (num_sm - invalid)/num_sm
    print('the percentage of valid molecules(%):',precent_valid*100)
    nuique_smi = list(set(can_smis))
    percent_nuique_mol = len(nuique_smi)/num_sm
    print('the percentage of nuique molecules(%):',percent_nuique_mol*100)
    dic = {'SMILES':can_smis}
    data=pd.DataFrame(dic)
    data.to_csv('nuique_valid_smi.csv',index=False)

cal_valid_unique_smi(df_smiles)