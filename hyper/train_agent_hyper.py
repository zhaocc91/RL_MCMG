
import os
import torch

you_chosen_con = ['active', 'logp', 'sa']
you_chosen_good_threshold = [['>= 0.5'], ['>=4', 'and', '<6'], ['<=4']]
you_refused_bad_threshold = [['<0.2'],['<2', 'or','>=8'],['>5']]
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
special_token = generate_special_token()
good_token_list = ['good_'+tok for tok in you_chosen_con]
n_steps = 8000
batch_size = 128
sigma = 60
learning_rate = 0.0001
max_length = 140
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
map_location={'cuda:3': 'cuda:3'}
restore_prior_from = 'save_rnn_model/rnn_model.ckpt'
restore_agent_from = 'save_rnn_model/rnn_model.ckpt'

save_model_f = 'save_agent_model'
if not os.path.exists(save_model_f):
    os.mkdir(save_model_f)
save_model_path = os.path.join(save_model_f,'agent_model.ckpt')
save_loss_path = os.path.join(save_model_f,'agent_loss.csv')
save_mols_dir = 'agent_generated_mols'
if not os.path.exists(save_mols_dir):
    os.mkdir(save_mols_dir)

init_from_file="data/Voc_RE1"  # voc file



