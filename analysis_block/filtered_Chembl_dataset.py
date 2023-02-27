
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
#from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import random

LigandDescriptors = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
                      'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
                      'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'BalabanJ',
                      'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
                      'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1',
                      'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA14', 'SMR_VSA1', 'SMR_VSA10',
                      'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
                      'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
                      'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6',
                      'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10',
                      'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
                      'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1',
                      'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
                      'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3',
                      'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
                      'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
                      'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors',
                      'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                      'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP',
                      'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_N',
                      'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO',
                      'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O',
                      'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
                      'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide',
                      'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azo', 'fr_barbitur',
                      'fr_benzene', 'fr_bicyclic', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
                      'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
                      'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
                      'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
                      'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitroso', 'fr_oxazole',
                      'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
                      'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_pyridine', 'fr_quatN',
                      'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
                      'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_urea']

DescCalc = MolecularDescriptorCalculator(LigandDescriptors)

def GetRDKitDescriptors(mol):
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    return np.array((DescCalc.CalcDescriptors(mol)))

def fingerprints_from_mol(mol):  # use ECFP4
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features

def cal_feature(mol):
    try:
        fp = fingerprints_from_mol(mol)
        des =GetRDKitDescriptors(mol)
        feature = np.concatenate((fp,des))
    except:
        print('the mol is error')
        feature = np.array([0.0 for i in range(1194)])
    return feature

chembl_path = 'chembl_wash_smiles.txt'
n_clusters = 1024
n_clusters_v = 50
choice_length = 10   ##### the no. of choosed mols is n_clusters*n_clusters_v*choice_length

smiles_s = []
with open(chembl_path,'r') as f:
    lines = f.readlines()
lines = list(set(lines))
print(len(lines))
print(lines[0:5])
if len(lines) > 5000000: 
    lines_1 = random.sample(lines,5000000)
else:
    lines_1 = lines

for line in lines_1:
    smiles = line.split('\n')[0]
    if len(smiles) > 15 and len(smiles) < 200:
        pass
    else:
        continue

    try:
        can_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        smiles_s.append(can_smi)
    except:
        print('error smiles:', smiles)

smiles_s = list(set(smiles_s))
if len(smiles_s)> 2000000:
    smiles_s = random.sample(smiles_s,2000000)
features = []
for smiles in smiles_s:
    mol = Chem.MolFromSmiles(smiles)
    fe = cal_feature(mol)
    features.append(fe)

cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
y_pred = cluster.labels_
mol_set = {}
for i in range(n_clusters):
    mol_set[i]=[]
for index,i in enumerate(y_pred):
    mol_set[i].append(smiles_s[index])

#print(mol_set)
choosed_smiles = []
for k,v in mol_set.items():
    features_v = []
    for smiles in v:
        mol = Chem.MolFromSmiles(smiles)
        fe = cal_feature(mol)
        features_v.append(fe)
    try:
        cluster_v = KMeans(n_clusters=n_clusters_v, random_state=0).fit(features_v)
        y_pred_v = cluster_v.labels_
        mol_set_v = {}
        for i in range(n_clusters_v):
            mol_set_v[i] = []
        for index, i in enumerate(y_pred_v):
            mol_set_v[i].append(v[index])

        for key, val in mol_set_v.items():

            vv = list(val)
            if len(vv) > choice_length:
                sample = random.sample(vv, choice_length)
            else:
                sample = vv
            choosed_smiles.extend(list(sample))
    except:
        # n samples is less than n cluster
        choosed_smiles.extend(list(v))
        print('the k mols is too less',k)
        #print('the key is',k)
        #print('v is',v)
        #exit()


data = pd.DataFrame(choosed_smiles,columns=['SMILES'])
data.to_csv('filter_ChEMBL.csv', index=False)



