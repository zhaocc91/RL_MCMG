#!/bin/bash

echo "step 1: training prior model"
python 1_train_prior_model.py > train_prior.out
echo "step 2: prior model generates mols"
python 2_generate_mols_prior.py > prior_generated_mols.out
echo "step 3: training rnn model"
python 3_train_rnn_model.py > train_rnn.out
echo 'step 4: training agent model and generatting mols'
4_train_agent_model.py > train_agent.out
echo 'finished mcmg workflow !!!'
echo '##################################################'
echo 'May kind people be blessed with life-long peace.'
echo '##################################################'
