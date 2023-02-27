
from rdkit import Chem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prior_file_path = '/Users/zcc/MCMG_project/prior_generated_mols.csv'
df = pd.read_csv(prior_file_path, sep=' ')
df_smiles = df.loc[ : ,"SMILES"]
df_steps = df.loc[:,'step']
num_step = 8000

def plot_scatter(x,y,path):
    # 绘图
    # 1. 确定画布
    plt.figure(figsize=(8, 4))  # figsize:确定画布大小

    # 2. 绘图
    plt.scatter(x,  # 横坐标
                y,  # 纵坐标
                c='red',  # 点的颜色
                label='function')  # 标签 即为点代表的意思
    # 3.展示图形
    plt.legend()  # 显示图例
    plt.savefig(path)
    #plt.show()  # 显示所绘图形


def plot_success_distri(num_step,df_steps):
    success_num_dict = {i:0 for i in range(num_step)}
    for step in df_steps:
        success_num_dict[int(step)] +=1

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
    data.to_csv('unique_valid_smi.csv',index=False)

cal_valid_unique_smi(df_smiles)