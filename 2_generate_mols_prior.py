#!/usr/bin/env python
import argparse

import torch
import time

from models.model_MCMG import transformer_RL
from MCMG_utils.data_structs import Vocabulary
from MCMG_utils.utils import seq_to_smiles
import pandas as pd
from hyper import prior_gen_hyper as hy

def Transformer_generator(load_prior_model_from=hy.load_prior_model_from,
                          save_prior_path=hy.save_prior_path,
                          batch_size=hy.batch_size,
                          n_steps=hy.n_steps,

                          ):
    voc = Vocabulary(init_from_file=hy.init_from_file)
    voc.special_tokens =hy.special_token
    voc.update_attri_values()  # update the words bank

    start_time = time.time()

    Prior = transformer_RL(voc, hy.d_model, hy.nhead, hy.num_decoder_layers,
                           hy.dim_feedforward, hy.max_seq_length,
                           hy.pos_dropout, hy.trans_dropout,hy.device)

    Prior.decodertf.eval()

    # By default restore middle_RNN to same model as Prior, but can restore from already trained middle_RNN too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        #Prior.decodertf.load_state_dict(torch.load(load_prior_model_from, map_location=hy.map_location))
        Prior.decodertf.load_state_dict(torch.load(load_prior_model_from))
    else:
        Prior.decodertf.load_state_dict(
            torch.load(load_prior_model_from, map_location=lambda storage, loc: storage))

    Prior.decodertf.to(hy.device)

    smile_list = []

    for i in range(n_steps):
        seqs = Prior.generate(batch_size, max_length=hy.max_seq_length, con_token_list=hy.token_list)

        smiles = seq_to_smiles(seqs, voc)

        smile_list.extend(smiles)

        print('step: ', i)

    smile_list = pd.DataFrame(smile_list,columns=['SMILES'])
    smile_list.to_csv(save_prior_path,  index=False)


if __name__ == "__main__":
    Transformer_generator()
    '''
    max_seq_length = 140
    # num_tokens=71
    # vocab_size=71
    d_model = 128
    # num_encoder_layers = 6
    num_decoder_layers = 12
    dim_feedforward = 512
    nhead = 8
    pos_dropout = 0.1
    trans_dropout = 0.1
    n_warmup_steps = 500

    num_epochs = 600
    batch_size = 128

    n_steps = 5000

    token_list = ['is_DRD2', 'high_QED', 'good_SA']

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=500)
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=128)
    parser.add_argument('--prior', action='store', dest='restore_prior_from',
                        default='./data/Prior.ckpt',
                        help='Path to an c-Transformer checkpoint file to use as a Prior')

    parser.add_argument('--save_molecules_path', action='store', dest='save_file',
                        default='test.csv')
    arg_dict = vars(parser.parse_args())  
    Transformer_generator(**arg_dict)                 
    '''






