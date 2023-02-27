#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from MCMG_utils.data_structs import MolData, Vocabulary
from models.model_MCMG import transformer_RL
from torch.optim import Adam
from MCMG_utils.Optim import ScheduledOptim
from MCMG_utils.early_stop.pytorchtools import EarlyStopping
from generate_prior_model_dataset import Generate_Prior_Data
from hyper import prior_hyper as hy
load_pretrain_model = hy.need_load_pre_train
def train_prior():

    """Trains the Prior decodertf"""
    gd = Generate_Prior_Data()
    train_data, valid_data = gd.rewrite_input_data()
    special_token = gd.special_token
    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=hy.init_from_file)
    voc.special_tokens = special_token
    voc.update_attri_values()  # update the words bank
    #print('vocab is:',voc.vocab)
    # Create a Dataset from a SMILES file
    moldata = MolData(train_data, voc)
    valid = MolData(valid_data, voc)

    train_data = DataLoader(moldata, batch_size=hy.batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    valid_data = DataLoader(valid, batch_size=hy.batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = transformer_RL(voc, hy.d_model, hy.nhead, hy.num_decoder_layers,  #Prior is a object
                           hy.dim_feedforward, hy.max_seq_length,
                           hy.pos_dropout, hy.trans_dropout,hy.device)

    if load_pretrain_model:
        try:
            if torch.cuda.is_available():
                # Prior.decodertf.load_state_dict(torch.load(load_prior_model_from, map_location=hy.map_location))
                Prior.decodertf.load_state_dict(torch.load(hy.load_pretrain_path))
            else:
                Prior.decodertf.load_state_dict(
                    torch.load(hy.load_pretrain_path, map_location=lambda storage, loc: storage))
        except:
            print('You have not pro_trained model, but you want to load it.')

    Prior.decodertf.to(hy.device)

    optim = ScheduledOptim(
        Adam(Prior.decodertf.parameters(), betas=(0.9, 0.98), eps=1e-09),
        hy.d_model * 8,hy.n_warmup_steps)

    train_losses, val_losses = train(train_data, valid_data, Prior, optim, hy.num_epochs,hy.save_prior_path)

    torch.cuda.empty_cache()

def train(train_data, valid_data, model, optim, num_epochs,save_prior_path):
    model.decodertf.to(hy.device)
    #print('device is',device)
    #print('the structure of model is',model.decodertf)
    #exit()
    model.decodertf.train()
    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0

    early_stopping = EarlyStopping(patience=5, verbose=False)
    loss_file = hy.save_loss_path
    for epoch in range(num_epochs):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
        total_loss = 0
        for step, batch in tqdm(enumerate(train_data), total=len(train_data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss, each_molecule_loss is the loss of  each molecule

            loss, each_molecule_loss = model.likelihood(seqs,hy.len_con)
            # loss = - log_p.mean()

            # Calculate gradients and take a step
            optim.zero_grad()
            loss.backward()
            optim.step_and_update_lr()
            # print(loss)

            total_loss += loss.item()
            # train_losses.append((step, loss.item()))

            # if step % print_every == print_every - 1:


            if step % 200 == 0 and step != 0:
                # decrease_learning_rate(optim, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.data))

        print('average epoch loss:', total_loss / len(train_data))
        val_loss = validate(valid_data, model)
        val_losses.append((total_step, val_loss))
        #### write loss to csv
        s = f'{epoch},{total_loss/len(train_data)},{val_loss}\n'
        with open(loss_file,'a+') as f:
            f.write(s)
        early_stopping(val_loss, model.decodertf, 'RE1_Prior')

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Save the Prior
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model.decodertf.state_dict(), save_prior_path)
        print(f'Val Loss: {val_loss}')
    return train_losses, val_losses

def validate(valid_data, model):
    # pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.decodertf.to(hy.device)
    model.decodertf.eval()
    total_loss = 0

    for step, batch in tqdm(enumerate(valid_data), total=len(valid_data)):
        with torch.no_grad():
            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss, each_molecule_loss is the loss of  each molecule
            loss, each_molecule_loss = model.likelihood(seqs,hy.len_con)
            # loss = - log_p.mean()

            total_loss += loss.item()
            # train_losses.append((step, loss.item()))
    return total_loss / len(valid_data)



if __name__ == "__main__":

    train_prior()

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
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    print(device)

    parser = argparse.ArgumentParser(description="Main script for running the model")
    #parser.add_argument('--train-data', action='store', dest='train_data')
    #parser.add_argument('--valid-data', action='store', dest='valid_data')
    parser.add_argument('--save-prior-path', action='store', dest='save_prior_path',
                        default='./data/Prior.ckpt',
                        help='Path to save an c-Transformer checkpoint.')

    arg_dict = vars(parser.parse_args())
    '''
