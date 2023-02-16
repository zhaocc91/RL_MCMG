#import prior_hyper as ph
import json

# train_prior_json
train_mprior_block = {
    'entrypoint':'python 1_train_prior_model.py',
    'params' :[{'name':'you_chosen_con',
                'type':'list',
                'value':['active', 'logp', 'sa']},
               {'name':'you_chosen_good_threshold',
                'type':'list',
                'value':[['>= 0.5'], ['>=4', 'and', '<6'], ['<=3', 'or', '>=8']]},
               {'name':'data_path',
                'type':'str',
                'value':'data/drd2/drd_train.csv'},
               {'name':'split_ratio',
                'type':'list',
                'value':[4,1]},
               {'name':'init_from_file',
                'type':'str',
                'value':"data/Voc_RE1"},
               {'name':'need_external_qasa_model',
                'type':'boolean',
                'value':False},
               {'name':'external_qasa_model_dict',
                'type':'dict',
                'value':None},
               {'name': 'max_seq_length',
                'type': 'int',
                'value': 140},
               {'name':'d_model',
                'type':'int',
                'value':128},
               {'name':'num_decoder_layers',
                'type':'int',
                'value':12},
               {'name':'dim_feedforward',
                'type':'int',
                'value':512},
               {'name':'nhead',
                'type':'int',
                'value':8},
               {'name':'pos_dropout',
                'type':'float',
                'value':0.1},
               {'name':'trans_dropout',
                'type':'float',
                'value':0.1},
               {'name':'n_warmup_steps',
                'type':'int',
                'value':600},
               {'name':'batch_size',
                'type':'int',
                'value':128},
               {'name':'device',
                'type':'str',
                'value':'cuda:2'},
               {'name':'save_prior_f',
                'type':'str',
                'value':'save_prior_model'
               }
    ]

}
#sort_keys=True, indent=4, separators=(',', ': ')
pri_json = json.dumps(train_mprior_block,sort_keys=True, indent=4, separators=(',', ': '))
print(pri_json)