import torch
import numpy as np

device=torch.device('cpu')
sample_frac=0.2
log_results=True 
Result_root='results/con_15_large/'
debug=False

### resume training setup ###
resume_train=True
training_info_folder='results/con_15_300'
resume_model_path=training_info_folder+'/models/'+ 'model_100.ckpt'#'train_model_cont.ckpt'





### Dataset settings ###
token={'unk':0,'pad':1,'start':2,'eos':3,'A':4,'R':5,'N':6,'D':7,'C':8,'E':9,'Q':10,'G':11,'H':12,'I':13,'L':14,'K':15,
'M':16,'F':17,'P':18,'S':19,'T':20,'W':21,'Y':22,'V':23}

token_reverse={0:'unk',1:'pad',2:'start',3:'eos',4:'A',5:'R',6:'N',7:'D',8:'C',9:'E',10:'Q',11:'G',12:'H',13:'I',14:'L',15:'K',
16:'M',17:'F',18:'P',19:'S',20:'T',21:'W',22:'Y',23:'V'}



dataset_params={
'data_path':'../data_processing/my_data/unlabeled_revised_2.csv',
'max_seq_len':15,  ### conditional sequence length 
'train_eval_ratio':0.8,
'token':token,
}


### training setups ###
training_params={
'model_save_folder':Result_root+'models',
'lr':10**-3,
'epochs':100000,
'train_batch_size':64,
'test_batch_size':64,
'shuffle':True}

use_kl=False
loss_weights={
'pz':1,
'recon':1,
'prop':1, 
'beta':{'val':[1,2],'iter':[0,10000]},
'lambda_logvar_L1':0,
'lambda_logvar_KL':1e-3
}
wae_mmd={'sigma':7.0,'kernel':'gaussian','rf_dim':500,'rf_resample':False}

### Model setups ###
GRUEncoder_params={'h_dim':80,'biGRU':True,'layers':1,'p_dropout':0.0}
GRUDecoder_params={'p_word_dropout':0.3,'p_out_dropout':0.3,'skip_connetions':True}


RNN_VAE_params={'n_vocab':len(token),'emb_dim':100,'z_dim':100,'max_seq_len':dataset_params['max_seq_len'],'PAD_IDX':token['pad'],
'device':device,'GRUEncoder_params':GRUEncoder_params,'GRUDecoder_params':GRUDecoder_params}

