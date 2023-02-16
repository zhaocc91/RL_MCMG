#!/usr/bin/env python
import argparse
import warnings

import torch

import numpy as np
import pandas as pd
import time

from models.model_rnn import RNN
from MCMG_utils.data_structs import Vocabulary, Experience
from MCMG_utils.properties import get_scoring_function, qed_func, sa_func,multi_scoring_functions_one_hot
from MCMG_utils.utils import Variable, seq_to_smiles, fraction_valid_smiles, unique

warnings.filterwarnings("ignore")
import os
from hyper import train_agent_hyper as hy
from Judge_Mol_Quailties import Judge_Mol_Scores
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def train_agent(restore_prior_from=hy.restore_prior_from,
                restore_agent_from=hy.restore_agent_from,
                agent_save=hy.save_model_path,
                batch_size=hy.batch_size, n_steps=hy.n_steps, sigma=hy.sigma, save_dir=hy.save_mols_dir,
                experience_replay=0):

    voc = Vocabulary(init_from_file=hy.init_from_file)
    voc.special_tokens = hy.special_token
    voc.update_attri_values()  # update the words bank

    start_time = time.time()

    Prior = RNN(voc,hy.device)
    Agent = RNN(voc,hy.device)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=hy.map_location))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load(restore_prior_from, map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=hy.learning_rate)

    experience = Experience(voc)  # record experience

    print("Model initialized, starting training...")

    # Scoring_function
    #scoring_function1 = get_scoring_function('drd2')  # pre_trained model to give a score
    smiles_save = []
    expericence_step_index = []
    loss_file = hy.save_loss_path
    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size=batch_size, max_length=hy.max_length)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood = Prior.likelihood(Variable(seqs),hy.len_con)
        smiles = seq_to_smiles(seqs, voc)
        '''
        smiles_1 = []
        for smi in smiles:
            smile = smi.split('_SA')[-1]
            smiles_1.append(smile)
        smiles = smiles_1
        '''

        #score1 = scoring_function1(smiles)
        # score2 = scoring_function2(smiles)
        judge_mol =Judge_Mol_Scores(smiles)
        all_data = judge_mol.get_scores()
        all_data['score'] = all_data.iloc[:,1:].sum(axis=1)
        success_score =hy.len_con - 0.0001
        success_data = all_data[all_data.score >= success_score]
        score = torch.tensor(all_data.iloc[:,1:].sum(axis=1).tolist())
        print('score',score)
        success_smiles = success_data['SMILES'].tolist()
        if len(success_smiles) > 0:
            record_step = [step for i in range(len(success_smiles))]
            success_data['step'] = record_step

        '''
        qed = qed_func()(smiles)
        sa = np.array([float(x < 4.0) for x in sa_func()(smiles)],
                      dtype=np.float32)  # to keep all reward components between [0,1]
        score = score1 + qed + sa
        '''

        '''
        # 判断是否为success分子，并储存
        success_score = multi_scoring_functions_one_hot(smiles, ['drd2', 'qed', 'sa'])
        itemindex = list(np.where(success_score ==3))
        success_smiles = np.array(smiles)[itemindex]
        #print('itemindex',itemindex)
        #print('success_smiles',success_smiles)
        smiles_save.extend(success_smiles)
        #print('smiles_save',smiles_save)
        #print('score1',score1[itemindex])
        #print('qed',qed[itemindex])
        #print('sa',sa[itemindex])
        #exit()
        record_step = [step for i in range(len(itemindex[0]))]
        if len(itemindex)>0:
            #try:
            mol_dict['smiles'].extend(success_smiles[0])
            mol_dict['score'].extend(score1[itemindex][0])
            mol_dict['qed'].extend(qed[itemindex][0])
            mol_dict['sa'].extend(sa[itemindex][0])
            mol_dict['epoch'].extend(record_step)
            #except:
            #    mol_dict = {'smiles':success_smiles[0],'score':score1[itemindex][0],
            #               'qed':qed[itemindex][0],'sa':sa[itemindex][0],
            #                'epoch':record_step}
        '''


        expericence_step_index = expericence_step_index + len(success_smiles) * [step]

        if (step+1) % 100 == 0 and step != 0:
            #print('num: ', len(smiles_save))
            # save_smiles_df = pd.concat([pd.DataFrame(smiles_save), pd.DataFrame(expericence_step_index)], axis=1)
            #print('mol_dict:',mol_dict)
            if len(success_smiles) > 0:
                save_smiles_df = success_data
                file = f'{step}_MCMG_drd.csv'
                save_path = os.path.join(save_dir, file)
                save_smiles_df.to_csv(save_path, index=False, header=True)

        if (step+1) >= n_steps:
            '''
            # save_smiles_df = pd.concat([pd.DataFrame(smiles_save), pd.DataFrame(expericence_step_index)], axis=1)
            save_smiles_df = pd.DataFrame(mol_dict)
            file = f'all_MCMG_drd.csv'
            save_path = os.path.join(save_dir,file)
            save_smiles_df.to_csv(save_path, index=False, header=False)
            '''
            break
        if step % 100 == 0 and step != 0:
            torch.save(Agent.rnn.state_dict(), agent_save)

        # Calculate augmented likelihood  #
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        s = f'{step},{loss.item()}\n'
        with open(loss_file, 'a+') as f:
            f.write(s)
        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
            step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(2):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))
if __name__ == "__main__":
    train_agent()
    '''
    parser = argparse.ArgumentParser(description="Main script for running the model")
    parser.add_argument('--num-steps', action='store', dest='n_steps', type=int,
                        default=5000)
    parser.add_argument('--batch-size', action='store', dest='batch_size', type=int,
                        default=128)
    parser.add_argument('--sigma', action='store', dest='sigma', type=int,
                        default=60)
    parser.add_argument('--middle_from', action='store', dest='restore_prior_from',
                        default='./data/DM_middle_drd.ckpt',
                        help='Path to an RNN checkpoint file to use as a Prior')
    parser.add_argument('--agent_from', action='store', dest='restore_agent_from',
                        default='./data/DM_middle_drd.ckpt',
                        help='Path to an RNN checkpoint file to use as a Prior')
    parser.add_argument('--agent_save', action='store', dest='agent_save',
                        default='./data/DM_middle_drd.ckpt',
                        help='Path to an RNN checkpoint file to use as a Agent.')
    parser.add_argument('--save-file-path', action='store', dest='save_dir',
                        help='Path where results and model are saved. Default is data/results/run_<datetime>.')

    arg_dict = vars(parser.parse_args())

    train_agent(**arg_dict)
    '''


