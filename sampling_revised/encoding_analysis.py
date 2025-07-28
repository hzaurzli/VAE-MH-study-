import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

# from vae_condition.results.con_13.model_file import RNN_VAE as condtion_model 
# import vae_condition.results.con_13.configure_file as condition_param 


from vae_extension.results.con_13.model_file import RNN_VAE as extension_model 
import vae_extension.results.con_13.configure_file as extension_param


import VAE_configuration as Params

import seaborn as sns 
from matplotlib import pyplot as plt 


from get_sequence_encodings import get_encodings,get_seq_encodings,enc2seq
from utils import extension_dataset_label

import pickle
import h5py


if __name__=='__main__':



    lens=[15,16,17,18,19,20,21,22]

    file_names= ['no_IL_results_'+str(i)+'.txt' for i in lens]
    sampled_seqs = []
    for file in file_names:
        file_ID=open(file,'r')
        for seq in file_ID.read().splitlines():
            sampled_seqs.append(seq)        
    
    sampled_encodings = []
    enc_file_name = ['no_IL_results_'+str(i)+'.npy' for i in lens]
    for idx,file in enumerate(enc_file_name):
        sampled_enc=np.load(file,'r')
        if idx ==0:
            sampled_encodings= sampled_enc
        else: 
            sampled_encodings= np.concatenate([sampled_encodings,sampled_enc],axis=0)
    print ('sampled_encodings',sampled_encodings.shape)

     

    data_path='silent_no_IL.csv'
    base='YPEDILDKHLQRV'
    filename='no_IL'
    extension_model_path= 'vae_extension/results/con_13/models/model_160.ckpt'

    # dateset_class=Peptide_condition_dataset
    # condition_encodings_2,df_con15_2=get_encodings(data_path,condtion_model,condition_param,condition_model_path,dateset_class,base=base,filename=filename)
    dateset_class=extension_dataset_label
    _,raw_df=get_encodings(data_path,extension_model,extension_param,extension_model_path,dateset_class,base=base,filename=filename)

    with h5py.File('sampling_models/con13/label_extension13_train.h5', 'r') as hf:
        data=np.array(hf['data'])
        print (data.shape)
        extension_encodings=data[:,:100]


    # sampled_encodings,_=get_seq_encodings(sampled_seqs,extension_model,extension_param,extension_model_path,dateset_class)
    print ('extension_encodings',extension_encodings.shape)
    print ('sampled_encodings',sampled_encodings.shape)



    complete_encoding= np.concatenate([extension_encodings,sampled_encodings],axis=0)




    from sklearn.manifold import TSNE
    t_sne = TSNE(n_components=2,perplexity=30,random_state=2022)
    # from sklearn.decomposition import PCA 
    # t_sne = PCA(n_components=2)
    print ('complete_encoding_total',complete_encoding.shape)
    encoding_tsne = t_sne.fit_transform(complete_encoding)


    rank_df= pd.read_csv(data_path)
    labels = rank_df['label'].to_list()


    model_path= 'sampling_models/con15/qcz/XGB/SVC0.sav'
    model= pickle.load(open(model_path, 'rb'))
    knn_labels=model.predict(extension_encodings)
    print (knn_labels.shape)
    
    N_wrong=np.sum(np.array(labels)-np.array(knn_labels))
    print ('N_wrong',N_wrong)
    sampled_labels = model.predict(sampled_encodings)
    print ('predicted_sampled_labels',sampled_labels)


    # debug_seqs=[]
    # debug_file_ID=open('with_IL_results_debug.txt','r')
    # for seq in debug_file_ID.read().splitlines():
    #     debug_seqs.append(seq)   

    # print ('debug seqs',debug_seqs)
    # debug_encodings_1,_=get_seq_encodings(debug_seqs,extension_model,extension_param,extension_model_path,dateset_class)
    # encs_debug=np.load('with_IL_results_debug.npy')
    # print ('encs_debug_shape',encs_debug.shape)
    # encs_debug_token=np.load('with_IL_results_token_debug.npy')
    # print ('encs_debug_token',encs_debug_token)
    # # print (encs_debug_token)
    # print ('get enc',debug_encodings_1[0][:10])
    # print ('get enc saved',encs_debug[0][:10])

    # new_probs = model.predict_proba(debug_encodings_1)[:,1]
    # print ('new_probs 1',new_probs)


    # new_probs = model.predict_proba(encs_debug)[:,1]
    # print ('new_probs 2',new_probs)
    

    # tokens=enc2seq(encs_debug,extension_model,extension_param,extension_model_path)
    # print ('enc2seq tokens',tokens)
    # sys.exit(0)

    # print ('rank_df',rank_df)
    # print ('raw_df',raw_df)
    # sys.exit(0)


    get_prop=lambda x:raw_df[x].to_list()
    I_scs=get_prop('I_sc')

    sort=np.sort(I_scs)
    theshold = sort[int(0.1*len(I_scs))]
    print (theshold)
    cat=[]
    for ele in I_scs:
        if ele <= theshold:
            cat.append(0)
        else:
            cat.append(1)

    tsne_df= {'x':encoding_tsne[:,0],'y':encoding_tsne[:,1],'cat':labels+[2]*len(sampled_encodings),'evaluation':[0]*len(extension_encodings)+[1]*len(sampled_encodings)}
    tsne_df=pd.DataFrame(tsne_df)
    sns.scatterplot(data=tsne_df, x="x", y="y", hue="cat",style='evaluation', palette="deep")
    plt.show()

    from sklearn.decomposition import PCA 
    pca = PCA(n_components=2)
    encoding_pca = pca.fit_transform(complete_encoding)
    pca_df= {'x':encoding_pca[:,0],'y':encoding_pca[:,1],'cat':labels+[2]*len(sampled_encodings),'evaluation':[0]*len(extension_encodings)+[1]*len(sampled_encodings)}
    pca_df=pd.DataFrame(pca_df)
    sns.scatterplot(data=pca_df, x="x", y="y", hue="cat",style='evaluation', palette="deep")
    plt.show()