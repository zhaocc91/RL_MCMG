#!/usr/bin/env python
import argparse

import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from MCMG_utils.data_structs import MolData, Vocabulary
from models.model_rnn import RNN
from MCMG_utils.utils import  decrease_learning_rate
rdBase.DisableLog('rdApp.error')
from hyper import train_rnn_hyper as hy
from generate_rnn_model_dataset import Generate_RNN_Data

def train_middle(train_data=hy.train_data, save_model=hy.save_model_path):
    """Trains the Prior RNN"""
    gd = Generate_RNN_Data()
    train_data = gd.rewrite_input_data()

    # Read vocabulary from a file
    #voc = Vocabulary(init_from_file="data/Voc_RE1")
    voc = Vocabulary(init_from_file=hy.init_from_file)
    voc.special_tokens = hy.special_token
    voc.update_attri_values()  # update the words bank
    # Create a Dataset from a SMILES file
    moldata = MolData(train_data, voc)
    data = DataLoader(moldata, batch_size=hy.batch_size, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    Prior = RNN(voc,hy.device)
    print('the device of rnn model is',Prior.device)
    loss_file = hy.save_loss_path
    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = hy.learning_rate)
    for epoch in range(0, hy.epochs):

        for step, batch in tqdm(enumerate(data), total=len(data)):
            #print('step is:',step)
            #print('batch is:',batch.size())
            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p = Prior.likelihood(seqs,hy.len_con)
            loss = - log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
              
            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                print(loss.cpu().data)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.cpu().data))
                seqs, likelihood, _ = Prior.sample(128,hy.max_length,hy.token_list)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    #smile = smile.split('_SA')[-1]
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), save_model)
        s = f'{epoch},{loss.item()}\n'
        with open(loss_file, 'a+') as f:
            f.write(s)
        # Save the Prior
        torch.save(Prior.rnn.state_dict(), save_model)

if __name__ == "__main__":

    train_middle()

    '''
    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--train-data', action='store', dest='train_data')
    parser.add_argument('--save-middle-path', action='store', dest='save_model',
                        help='Path and name of middle model.')

    arg_dict = vars(parser.parse_args())

    train_middle(**arg_dict)
    '''


