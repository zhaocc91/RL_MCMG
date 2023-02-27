

import torch
import os

############################## this variable you neen attention #########
you_chosen_con = ['active', 'logp', 'sa']
init_from_file="data/Voc_RE1"  # voc file
max_seq_length = 140
batch_size = 128
n_steps = 4000   # generate about 100w mols
# the number of molecules is generated is n_steps*batch_size
##########################################################################
def generate_special_token():
    special_token = ['EOS', 'GO']
    # special_dict = {}
    for con in you_chosen_con:
        for des in ['good_', 'bad_']:
            special_token.append(des + con)
    return special_token
token_list = ['good_'+tok for tok in you_chosen_con]
special_token = generate_special_token()
d_model = 128
num_decoder_layers = 12
dim_feedforward = 512
nhead = 8
pos_dropout = 0.1
trans_dropout = 0.1
n_warmup_steps = 500
#num_epochs = 600

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
map_location={'cuda:3': 'cuda:3'}
load_prior_model_from = "save_prior_model/prior_model.ckpt"

save_mols_f = 'save_prior_generated_mols'
if not os.path.exists(save_mols_f):
    os.mkdir(save_mols_f)
save_prior_path = os.path.join(save_mols_f,'prior_generated_mols.csv')
