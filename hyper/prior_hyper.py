
import torch
import os

#########################input data & constaint conditions ######################
you_chosen_con = ['active', 'logp', 'sa']
you_chosen_good_threshold = [['>= 0.5'], ['>=4', 'and', '<6'], ['<=4']]
#data_path = 'data/drd2/drd_train_bk.csv'
data_path = 'data/ChEMBL_45w/random_filter_ChEMBL_1.csv'
split_ratio = [4, 1]
init_from_file="data/Voc_RE1"  # voc file
len_con = len(you_chosen_con)
#########################qasa model #############################################
need_external_qasa_model=False
external_qasa_model_dict=None

#########################the prior model parameters#############################

max_seq_length = 140
d_model = 128
num_decoder_layers = 12
dim_feedforward = 512
nhead = 8
pos_dropout = 0.1
trans_dropout = 0.1
n_warmup_steps = 500
num_epochs = 600   # the max_epoch to trian prior model
batch_size = 128
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
save_prior_f = 'save_prior_model'
if not os.path.exists(save_prior_f):
    os.mkdir(save_prior_f)
save_prior_path = os.path.join(save_prior_f,'prior_model.ckpt')
save_loss_path = os.path.join(save_prior_f,'prior_model_loss.csv')
need_load_pre_train = False
pretrained_prior_model = 'pretrained_prior_model'
if not os.path.exists(pretrained_prior_model):
    os.mkdir(pretrained_prior_model)
load_pretrain_path = os.path.join(pretrained_prior_model,'pretrain_prior_model.ckpt')
