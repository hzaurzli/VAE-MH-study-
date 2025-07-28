import sys
import os 

from save_encodings import save_train_encodings
from train_sampling_models import train_qcz
from train_sampling_models import train_qz
from sample_seqs_revise import sampling_seqs





# from utils import extension_dataset_label
# from vae_extension.results.con_15_large import configure_file as extension_params 
# from vae_extension.results.con_15_large.model_file import RNN_VAE as extension_model 

# model_path='../vae_extension/results/con_15_large/models/model_100.ckpt'
# extension_filename='label_extension15_train'
# labelled_data_path='silent_with_IL.csv'
# save_folder='./sampling_models/con15'


# qcz_params={'N_boots':4,'model_name':'SVC','save_folder':'./sampling_models/con15/qcz/XGB'}
# qz_params={'model_name':'GMM','n_components':5,'save_folder':'./sampling_models/con15/qz'}


# MH_sampling_params={'exploration_rate':[0.5,0.1],'N_chains':1000,'burn_in':500,\
# 'extension_model':extension_model,'model_path':model_path,'extension_params':extension_params,\
# 'labeled_encoding_path':save_folder+'/'+extension_filename,\
# 'labeled_data_path':labelled_data_path,'base':'YPEDILDKHLQRVIL',\
# 'seqs_dict':{17:{},18:{},19:{},20:{},21:{},22:{}},'lens':[17,18,19,20,21,22],'desired_N':[15,15,15,15,15,15],\
# 'result_file_name':'with_IL_results'}




from utils import extension_dataset_label
from vae_extension.results.con_13 import configure_file as extension_params 
from vae_extension.results.con_13.model_file import RNN_VAE as extension_model 

model_path='../vae_extension/results/con_13/models/model_160.ckpt'  ### the peptide encoding model 
save_folder= '../../results/beta'#'./sampling_models/con13/beta'  # the folder to save the qcz model and the encodings 
extension_filename='label_extension13_train_beta' ### the sub folder to save the encodings 
labelled_data_path= 'labelled_data/silent_no_IL.csv'#  the labelled data folder. 'PDZ_results/2nd/pdz_labeled_2nd.csv' #'labeled_data_warm_up.csv'



### qcz_params: the qcz model settings, N_boots: bootstrap N times; 'model_name': KNN, SVM, XGBoost and LR, 'save_folder': the folder to save the pcz model
qcz_params={'N_boots':4,'model_name':'KNN','save_folder':save_folder+'/qcz_f1/KNN'} 
### to_train_qcz: train pcz model or load pcz model using the above save_folder path
to_train_qcz=True 
### qz_params: model_name: 'GMM', the model for qz; 'n_components': N cluster for GMM; 'save_folder': the save folder for the qz model   
qz_params={'model_name':'GMM','n_components':5,'save_folder':save_folder+'/qz'}

### MM parameters: 
### exploration_rate: the stds to sample the proposed encoding. [Initial std, exploration std]; 
### N_chain: MH iteration; burn_in: the warmup interation 
### base: peptide base 
### seqs_dict: the collection of sampled peptide for different legnth; length as the key 
### lens: the desired peptide length, should be the same as the keys of the seqs_dict
### desired_N: desired number of peptide for each peptide length. 
### result_file_name: the file_name for the sampled peptides   

MH_sampling_params={'exploration_rate':[0.5,0.1],'N_chains':1000,'burn_in':500,\
'extension_model':extension_model,'model_path':model_path,'extension_params':extension_params,\
'labeled_encoding_path':save_folder+'/'+extension_filename,\
'labeled_data_path':labelled_data_path,'base':'YPEDILDKHLQRV', \
'seqs_dict':{15:{},16:{},17:{},18:{},19:{},20:{},21:{},22:{}},'lens':[15,16,17,18,19,20,21,22],'desired_N':[15,15,15,15,15,15,15,15],\
# 'seqs_dict':{10:{}},'lens':[10],'desired_N':[250],\
'result_file_name':save_folder+'/beta_1st'}  ### seq_lens = base peptide length + desired extension length
#'YPEDILDKHLQRV' 
#'STIEEQAKTFLDKFNHEAEDLFYQS'
#'GGGW'

print ('######### 1.saving encodings ###########')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
sampling_model_training_data=save_train_encodings(labelled_data_path,model_path,extension_model,extension_params,extension_dataset_label,extension_filename,save_folder)
encodings = sampling_model_training_data[:,:-1]
labelled_encodings = sampling_model_training_data
print ('######### finished saving encodings ###########')

# sys.exit(0)

print ('######### 2.training qcz/qz models ###########')
if to_train_qcz:
    qcz_models=train_qcz(labelled_encodings,**qcz_params)
else: 
    from os import listdir
    import pickle 
    qcz_models=[]
    for qcz_model in listdir(qcz_params['save_folder']):
        model_name=qcz_params['save_folder']+'/'+qcz_model
        print (model_name)
        qcz_models+=[pickle.load(open(model_name, 'rb'))]


# sys.exit(0)
qz_model=train_qz(encodings,**qz_params)
print ('######### finished training qcz/qz ###########')

print ('######### 3.MH sampling ###########')
sampling_seqs(qcz_models,qz_model,**MH_sampling_params)
print ('######### finished MH sampling ###########')




