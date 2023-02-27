
import pickle
from sklearn import ensemble
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem
import numpy as np
from rdkit.Chem import Descriptors
from MCMG_utils.scripts import  sascorer
'''
model_dict = {'active_model':'path',
              'logp_model':'path',
              'qed_model':'path',
              'sa_model':'path'}
'''

class Get_External_Models():
    '''
    you need supply all the path of QSAR models;
    the dictionary of 'my_model_dict' will rewrite according your QSAR model
    other methods of the class may be deleted
    '''
    def __init__(self):
        self.active_path = 'external_model/rf_classify_NA.pkl'
        self.qed_path = ''
        self.smiles_s = []
    def get_need_model(self,name_list,need_external_qasa_model=False,
                       external_qasa_model_dict=None):    #get_need_model(self,*args,)
        need_model_names = name_list
        my_model_dict = {'active':self._active_model(),
                         'qed':self._qed_model(),
                         'logp':self._logp_model(),
                         'sa':self._sa(),
                         'other':self._beat_zcc_model()}
        if need_external_qasa_model:
            my_model_dict = external_qasa_model_dict
        chosed_model = {}
        for arg in need_model_names:
            chosed_model[arg] = my_model_dict[arg]
        return chosed_model

    def fingerprints_from_mol(self, mol):  # use ECFP4
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features.reshape(1, -1)

    def _calcul_fingerprints(self,):
        # smiles_list = df['smiles']
        fps = []
        #print(self.smiles_s)
        for i, smiles in enumerate(self.smiles_s):
            try:
                mol = Chem.MolFromSmiles(smiles)
                fp = self.fingerprints_from_mol(mol)
                fp.tolist()
            except:
                print('cal fp error')
                print('the bad mol is:',smiles)
                fp = np.array([[0*i for i in range(2048)]])
            fps.append(fp)
        #print(fps)
        fps = np.concatenate(fps, axis=0)
        #print(fps)
        return fps
    def _active_model(self,):
        # molodel_path = '/Users/zcc/MCMG_project/mcmg_QSAR/rf_classify_NA.pkl'
        with open(self.active_path, 'rb') as f:
            my_rf = pickle.loads(f.read())
        # my_rf = pickle.load(model_path)
        fps = self._calcul_fingerprints()
        active_values = my_rf.predict(fps).tolist()
        return active_values
    def _sa (self,):
        scores = []
        for smiles in self.smiles_s:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    scores.append(11)
                else:
                    scores.append(sascorer.calculateScore(mol))
            except:
                scores.append(11)


        return scores
    def _logp_model(self):
        logp_values = []
        for smi in self.smiles_s:
            try:
                des = Descriptors.MolLogP(Chem.MolFromSmiles(smi))
            except:
                des = -5
            logp_values.append(des)
        return logp_values

    def _qed_model(self, ):
        qed_values = []
        for smi in self.smiles_s:
            try:
                qe = QED.qed(Chem.MolFromSmiles(smi))
            except:
                qe = 0
            qed_values.append(qe)
        #qed_values = [QED.qed(Chem.MolFromSmiles(smi)) for smi in self.smiles_s]

        return qed_values

    def _beat_zcc_model(self, ):
        #qed_values = [QED.qed(Chem.MolFromSMiles(smi)) for smi in smiles_s]
        return 100

if __name__ == "__main__":
    get_models = Get_External_Models()
    get_models.smiles_s = ['CCCCC','CCCCC=O']
    my_models = get_models.get_need_model(['active'],False,None)
    con_values = list(my_models.values())[0]
    print(con_values)
