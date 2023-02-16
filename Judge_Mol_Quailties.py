
import pandas as pd
import numpy as np
# this should be replaced by other method
from upload_external_model import Get_External_Models
from hyper import train_agent_hyper as hy
from rdkit import Chem
import re
class Judge_Mol_Scores():

    def __init__(self,smiles_list):
        # step1: choice yourself constrain conditional:
        self.you_chosen_con = hy.you_chosen_con
        self.you_chosen_good_threshold = hy.you_chosen_good_threshold
        self.you_chosen_bad_threshold = hy.you_refused_bad_threshold
        # step2: give the path of datafile, then, split it with train and validate dataset
        self.smiles_list = smiles_list
        '''
        model_dict = {'active_model':'path',
                      'logp_model':'path',
                      'qed_model':'path',
                      'sa_model':'path'}
        '''
        self.generate_special_token()
    def generate_special_token(self):
        self.special_token = ['EOS','GO']
        #special_dict = {}
        for con in self.you_chosen_con:
            for des in ['good_','bad_']:
                self.special_token.append(des+con)
        return self.special_token
    def extract_num_from_str(self,str):

        num = float(re.findall(r"\d+\.?\d*", str)[0])
        return num
    def build_cal_score_func(self,good_thre_val,
                bad_thre_val,data):

        if len(good_thre_val) > 1:
            good_judge_str = f'{data}{good_thre_val[0]} {good_thre_val[1]} {data}{good_thre_val[2]}'
        else:
            good_judge_str = f'{data}{good_thre_val[0]}'

        if len(bad_thre_val) > 1:
            bad_judge_str = f'{data}{bad_thre_val[0]} {bad_thre_val[1]} {data}{bad_thre_val[2]}'
        else:
            bad_judge_str = f'{data}{bad_thre_val[0]}'
        if eval(good_judge_str):
            # new_data.append(1)
            y = 1.0
            return y
        if eval(bad_judge_str):
            # new_data.append(0)
            y = 0.0
            return y

        # line part
        if ('and' in good_thre_val) and ('or' in bad_thre_val): # shape:__/-\__
            print('good_thre_val',good_thre_val)
            print('bad_thre_val',bad_thre_val)
            p00 = self.extract_num_from_str(good_thre_val[0])
            p01 = self.extract_num_from_str(good_thre_val[2])
            p10 = self.extract_num_from_str(bad_thre_val[0])
            p11 = self.extract_num_from_str(bad_thre_val[2])
            points = ([[p00,p01].sort(),[p10,p11].sort()])
            #points = [[float(good_thre_val[0].split()[1]),float(good_thre_val[2].split()[1])].sort(),
            #          [float(bad_thre_val[0].split()[1]),float(bad_thre_val[2].split()[1])].sort()]

            if data < p00:
                # line_1 left
                y = ((1.0-0)/(p00-p10))*(data-p10) + 0
            if data > p01:
                # line-2 right
                y = ((-1.0 + 0) / (p11 - p01)) * (data - p01) + 1
            return y

        elif 'and' not in good_thre_val:  # shape: __/-- or --\__
            p0 = self.extract_num_from_str(good_thre_val[0])
            p1 = self.extract_num_from_str(bad_thre_val[2])
            #points = [[p0],
            #          [p1]]
            if p0 > p1: # shape __/--
                y = ((1.0-0)/(p0 -p1)) * (data-p1) + 0
            else:  # shape __/--
                y = ((1.0 - 0) / (p0 - p1)) * (data - p1) + 0
            return y
        else: # other type
            print('you supplied an invalid condition')
            return 0.0

    def get_scores(self):
        flite_all_smiles = []
        all_smiles = self.smiles_list
        '''
        for smiles in all_smiles:
            try:
                m = Chem.MolFromSmiles(smiles)
                if m != None:
                    flite_all_smiles.append(smiles)
            except:
                pass
        all_smiles = flite_all_smiles
        '''

        #con_data_ok = []
        con_data_need_predict =  self.you_chosen_con
        data_dict = {'SMILES': all_smiles}
        data_dict_values = {'SMILES': all_smiles}

        if len(con_data_need_predict) > 0:
            #  need external environment supply some models

            for con_p in con_data_need_predict:
                values = []
                get_models = Get_External_Models()
                get_models.smiles_s = all_smiles
                print('cin_p',con_p)
                my_models = get_models.get_need_model([con_p],hy.need_external_qasa_model,hy.external_qasa_model_dict)
                con_values = list(my_models.values())[0]
                index = self.you_chosen_con.index(con_p)
                good_thre_val = self.you_chosen_good_threshold[index]
                bad_thre_val = self.you_chosen_bad_threshold[index]
                # data_dict[con_p] = con_values
                #print('con_values',con_values)

                for data in con_values:
                    y = self.build_cal_score_func(good_thre_val, bad_thre_val, data)
                    values.append(y)
                    #print('conp:y:',con_p,y)
                    '''
                    if eval(judge_str):
                        #new_data.append(1)
                        new_data.append('good_' + con_p)
                    else:
                        #new_data.append(0)
                        new_data.append('bad_' + con_p)
                    '''

                data_dict[con_p] = values

        new_all_data = pd.DataFrame(data_dict)

        '''
        len_data = len(new_all_data)
        len_train_data = int(len_data *
                             (self.split_ratio[0] / (self.split_ratio[0] + self.split_ratio[1])))
        #len_validate_date = len_data - len_train_data
        train_data = new_all_data[0:len_train_data]
        validate_data = new_all_data[len_train_data:-1]
        '''

        return new_all_data

if __name__ == "__main__":
    gd = Judge_Mol_Scores(smiles_list = ['CCCCCC','c1ccccc1','123'])
    t_d =  gd.get_scores()
    print(t_d)


