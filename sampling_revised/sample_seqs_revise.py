import numpy as np 
import os 
import pickle
import pandas as pd 
import torch
import sys 
import h5py

import VAE_configuration as Param 
from utils import idxTostr

from get_sequence_encodings import get_seq_encodings
from utils import extension_dataset_label
import json



def compute_log_prob(qcz_models,zs): 
    ### qczs: q(c|z) models 
    ### zs: N*dim_z
    ### return: q(c=1|z), N*1 where N is batch size
    probs=np.array([qcz.predict_proba(zs) for qcz in qcz_models]) ### N_qcz*N*2
    probs[probs==0]=10**-10
    avg_probs=np.log(np.mean(probs,axis=0)[:,1]) ### N*1
    return avg_probs


def compute_log_acceptance_prob(qcz_models,qz_model,prev_zs,cur_zs):
    ### qczs: q(c|z) models 
    ### qz: q(z) model 
    cur_probs=compute_log_prob(qcz_models,cur_zs)  ### N*1
    prev_probs=compute_log_prob(qcz_models,prev_zs) ### N*1
    B= cur_probs-prev_probs
    rate= ((cur_probs-prev_probs)<0)*B
    cur_avg_probs= np.mean(cur_probs)
    N_1= np.sum(cur_probs==0)
    return rate,cur_avg_probs,N_1 

def batch_MH_process(sigma,sampled_zs,qcz_models,qz_model,batch):
    #### MH process to get encoding z 
    #### return z: N*1
    proposed_zs=np.array([np.random.normal(z1,sigma) for z1 in sampled_zs])
    acceptance_rate,cur_avg_probs,N_1=compute_log_acceptance_prob(qcz_models,qz_model,sampled_zs,proposed_zs)  ### N*1
    u=np.log(np.random.uniform(size=batch))
    compare=u<=acceptance_rate ### N*1 bool 
    sampled_zs_temp=np.zeros_like(sampled_zs)
    sampled_zs_temp[compare]=proposed_zs[compare]
    sampled_zs_temp[~compare]=sampled_zs[~compare]
    sampled_zs=sampled_zs_temp
    output_zs=proposed_zs[compare]
    return sampled_zs,acceptance_rate,compare,cur_avg_probs,output_zs,N_1

def load_qcz_models(model_folder):
    model_files=[file for file in os.listdir(model_folder) if file.split('.')[1]=='sav']
    qcz_models = [pickle.load(open(model_folder+'/'+model_file, 'rb')) for model_file in model_files]
    return qcz_models
    
def load_qz_model(model_folder):
    # print (os.listdir(model_folder))
    model_file=[file for file in os.listdir(model_folder) if file.split('.')[1]=='sav'] 
    qz_model = pickle.load(open(model_folder+'/'+model_file[0], 'rb'))
    return qz_model


def get_ecoding_from_df(df):
    encodings=df['encodings']
    encodings= [np.fromstring(encoding.strip('[').strip(']'), dtype=float, sep='  ') for encoding in encodings]
    return encodings


def sampling_seqs(qcz_models,qz_model,exploration_rate,N_chains,burn_in,extension_model,model_path,extension_params,labeled_encoding_path,labeled_data_path,base,seqs_dict,lens,desired_N,result_file_name):
    MP=extension_params.RNN_VAE_params
    model = extension_model(**MP).to(extension_params.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    batch=N_chains
    # print ('labeled_data_path',labeled_data_path)
    with h5py.File(labeled_encoding_path+'.h5', 'r') as hf:
      data=np.array(hf['data'])
      print ('MH process: labelled data shape',data.shape)   
    z= data[:,:-1]
    labels=data[:,-1]
    pos_z1= z[labels==1]
    init_sigma = np.std(pos_z1,axis=0)*exploration_rate[0]
    sigma = np.std(pos_z1,axis=0)*exploration_rate[1]

    mean = np.mean(pos_z1,axis=0)
    sampled_z1 = np.array([np.random.normal(mean,init_sigma) for i in range(batch)])  #### batch list of encodings 
    sampled_zs = sampled_z1 
    total_compare=np.zeros(batch)

    ########## Start MH burn-in ############
    for i in range(burn_in):
        sampled_zs,acceptance_rate,compare,cur_avg_probs,_,N_1=batch_MH_process(sigma,sampled_zs,qcz_models,qz_model,batch)
        total_compare+=compare*1
        if i==0 or (i+1)%10==0:
            print('burn-in iter:', i,'AR:',np.mean(acceptance_rate),np.std(acceptance_rate),'compare:',compare.any(),'cur_avg_prob:',cur_avg_probs,'total_1:',N_1)

    ######## Start MH sampling ######
    label_df= pd.read_csv(labeled_data_path)
    label_1_seqs= label_df[label_df['label']==1]['seq'].to_list()
    good_seqs=[]
    good_seq_tokens=[]
    good_encs=[]
    Iter_debug=0
    Break=0

    # prev_z=0
    # ii=0
    Iter=0

    ### init seqs_dict ###
    for l in lens:
        seqs_dict[l]['seqs']=[]
        seqs_dict[l]['encodings']=[]

    while  Break<len(lens):#len(good_seqs)<desired_N: 
        sampled_zs,acceptance_rate,compare,cur_avg_probs,output_zs,N_1=batch_MH_process(sigma,sampled_zs,qcz_models,qz_model,batch)
        if len(output_zs)!=0:
            
            probs=qcz_models[0].predict_proba(output_zs)[:,1]
            print ('avg output probs',np.mean(probs))

            sampled_zs_torch= torch.tensor(output_zs).float().to(extension_params.device)
            seqs_tokens= model.sampling_sequences_shift(sampled_zs_torch)
            seqs_tokens=seqs_tokens.detach().cpu().tolist()
            seqs,good_tokens,good_token_idx=idxTostr(seqs_tokens,base)
            selected_idx= [idx for idx,seq in enumerate(seqs) if seq not in good_seqs and seq not in label_1_seqs and len(seq)!=0]
            if len(selected_idx)!=0:
                selected_encodings= np.array([output_zs[good_token_idx[idx]] for idx in selected_idx])
                probs=qcz_models[0].predict_proba(selected_encodings)[:,1]
                print ('avg selected probs',np.mean(probs),'N_select:',len(selected_idx))    
            # else: 
            #     print ('len(selected_idx)',len(selected_idx),'len(output_zs)',len(output_zs))       

                selected_seq=[seqs[idx] for idx in selected_idx]
                good_seqs+=selected_seq
                selected_encodings = [output_zs[good_token_idx[idx]] for idx in selected_idx]
                good_encs += selected_encodings
                Break=0
                for l_idx,l in enumerate(lens): 
                    idx_l=[idx for idx,seq in enumerate(selected_seq) if len(seq)==l]
                    seq_l=[seq for idx,seq in enumerate(selected_seq) if len(seq)==l]
                    # idx_l,seq_l= zip(*[ (idx,seq) for idx,seq in enumerate(selected_seq) if len(seq)==l])
                    encoding_l = [selected_encodings[idx] for idx in idx_l]
                    seqs_dict[l]['seqs']+=seq_l
                    seqs_dict[l]['encodings']+=encoding_l                            
                    if len(seqs_dict[l]['seqs'])>=desired_N[l_idx]:
                        Break+=1 
        print ([len(seqs_dict[l]['seqs']) for l in lens])
        print('iter:',Iter,'AR:',np.mean(acceptance_rate),np.std(acceptance_rate),'len good seqs:',len(good_seqs),'cur_avg_prob:',cur_avg_probs,'total_1:',N_1)
        Iter+=1

    ### Writing results #######
    for l_idx,l in enumerate(lens):
        file_ID=open(result_file_name+'_{}.txt'.format(l),'a')
        # desired_seqs=seqs_dict[l]['seqs']
        desired_seqs=seqs_dict[l]['seqs'][:desired_N[l_idx]]
        for s in desired_seqs:
            file_ID.write(s+'\n')
        file_ID.close()
        desired_encodings=np.array(seqs_dict[l]['encodings'][:desired_N[l_idx]])
        np.save(result_file_name+'_{}'.format(l),desired_encodings)

    # with open("sampled_results.json", "w") as outfile:
    #     json.dump(seqs_dict, outfile)




