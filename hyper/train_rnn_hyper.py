

import os
import torch

you_chosen_con = ['active', 'logp', 'sa']
you_chosen_good_threshold = [['>= 0.5'], ['>=4', 'and', '<6'], ['<=4']]
init_from_file="data/Voc_RE1"  # voc file

len_con = len(you_chosen_con)
need_external_qasa_model=False
external_qasa_model_dict=None

def generate_special_token():
    special_token = ['EOS', 'GO']
    # special_dict = {}
    for con in you_chosen_con:
        for des in ['good_', 'bad_']:
            special_token.append(des + con)
    return special_token
token_list = ['good_'+tok for tok in you_chosen_con]
special_token = generate_special_token()
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
#train_data = 'data/drd2/drd_train_bk.csv'
train_data = 'save_prior_generated_mols/prior_generated_mols.csv'
save_model_f = 'save_rnn_model'
if not os.path.exists(save_model_f):
    os.mkdir(save_model_f)
save_model_path = os.path.join(save_model_f,'rnn_model.ckpt')
save_loss_path = os.path.join(save_model_f,'rnn_loss.csv')

max_length = 140
batch_size = 128
learning_rate = 0.001
epochs = 10
