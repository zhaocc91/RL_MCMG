import pandas as pd
import numpy as np
# this should be replaced by other method
from upload_external_model import Get_External_Models
from hyper import prior_hyper as hy

class Generate_Prior_Data():

    def __init__(self):
        ############################################# the section is that you need input or modify #################
        # step1: choice yourself constrain conditional:
        self.you_chosen_con = hy.you_chosen_con
        self.you_chosen_good_threshold = hy.you_chosen_good_threshold
        # step2: give the path of datafile, then, split it with train and validate dataset
        self.data_path = hy.data_path
        self.split_ratio = hy.split_ratio
        #############################################################################################################
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
    def rewrite_input_data(self):
        all_data = pd.read_csv(self.data_path)

        con_in_file = all_data.columns.values
        all_smiles = list(all_data.loc[:, "SMILES"])
        con_data_ok = [con for con in self.you_chosen_con if con in con_in_file]
        con_data_need_predict = [con for con in self.you_chosen_con if con not in con_in_file]
        data_dict = {'SMILES': all_smiles}

        if len(con_data_ok) > 0:

            for con_o in con_data_ok:
                new_data = []
                con_o_data = list(all_data.loc[:, con_o])
                index = self.you_chosen_con.index(con_o)
                thre_val = self.you_chosen_good_threshold[index]

                for data in con_o_data:
                    if len(thre_val) > 1:
                        judge_str = f'{data}{thre_val[0]} {thre_val[1]} {data}{thre_val[2]}'
                    else:
                        judge_str = f'{data}{thre_val[0]}'
                    if eval(judge_str):
                        #new_data.append(1)
                        new_data.append('good_'+con_o)
                    else:
                        #new_data.append(0)
                        new_data.append('bad_' + con_o)
                data_dict[con_o] = new_data
                # data_dict[con_o] = list(all_data.loc[ : ,con_o])

        if len(con_data_need_predict) > 0:
            #  need external environment supply some models

            for con_p in con_data_need_predict:

                new_data = []
                get_models = Get_External_Models()
                get_models.smiles_s = all_smiles
                my_models = get_models.get_need_model([con_p],hy.need_external_qasa_model,hy.external_qasa_model_dict)
                con_values = list(my_models.values())[0]
                index = self.you_chosen_con.index(con_p)
                thre_val = self.you_chosen_good_threshold[index]
                # data_dict[con_p] = con_values

                for data in con_values:
                    if len(thre_val) > 1:
                        judge_str = f'{data}{thre_val[0]} {thre_val[1]} {data}{thre_val[2]}'
                    else:
                        judge_str = f'{data}{thre_val[0]}'

                    if eval(judge_str):
                        #new_data.append(1)
                        new_data.append('good_' + con_p)
                    else:
                        #new_data.append(0)
                        new_data.append('bad_' + con_p)
                data_dict[con_p] = new_data

        new_all_data = pd.DataFrame(data_dict)

        len_data = len(new_all_data)
        len_train_data = int(len_data *
                             (self.split_ratio[0] / (self.split_ratio[0] + self.split_ratio[1])))
        #len_validate_date = len_data - len_train_data
        train_data = new_all_data[0:len_train_data]
        validate_data = new_all_data[len_train_data:-1]

        return train_data, validate_data

if __name__ == "__main__":
    gd = Generate_Prior_Data()
    t_d, v_d =  gd.rewrite_input_data()
    print(t_d)
    print(v_d)

