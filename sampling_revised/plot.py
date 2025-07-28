import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

# from vae_condition.results.con_13.model_file import RNN_VAE as condtion_model 
# import vae_condition.results.con_13.configure_file as condition_param 





import VAE_configuration as Params

import seaborn as sns 
from matplotlib import pyplot as plt 


from get_sequence_encodings import get_encodings,get_seq_encodings,enc2seq
from utils import extension_dataset_label

import pickle
import h5py


if __name__=='__main__':

    ### for no IL, use the following settings ####

    from vae_extension.results.con_13.model_file import RNN_VAE as extension_model 
    import vae_extension.results.con_13.configure_file as extension_param
    lens=[15,16,17,18,19,20,21,22]
    raw_data_encoding_path='sampling_models/con13/label_extension13_train.h5'
    file_names= ['no_IL_results_'+str(i)+'.txt' for i in lens]
    enc_file_name = ['no_IL_results_'+str(i)+'.npy' for i in lens]
    data_path='silent_no_IL.csv'
    base='YPEDILDKHLQRV'
    filename='no_IL'
    extension_model_path= 'vae_extension/results/con_13/models/model_160.ckpt'
    prop_to_plot='I_sc'



    ### for with IL, use the following settings ####
    # from vae_extension.results.con_15_large.model_file import RNN_VAE as extension_model 
    # import vae_extension.results.con_15_large.configure_file as extension_param
    # lens=[17,18,19,20,21,22]
    # raw_data_encoding_path='sampling_models/con15/label_extension15_train.h5'
    # file_names= ['with_IL_results_'+str(i)+'.txt' for i in lens]
    # enc_file_name = ['with_IL_results_'+str(i)+'.npy' for i in lens]
    # data_path='silent_with_IL.csv'
    # base='YPEDILDKHLQRVIL'
    # filename='with_IL'
    # extension_model_path= 'vae_extension/results/con_15_large/models/model_1-0.ckpt'
    # prop_to_plot='I_sc'









    sampled_seqs = []
    for file in file_names:
        file_ID=open(file,'r')
        for seq in file_ID.read().splitlines():
            sampled_seqs.append(seq)        
    
    sampled_encodings = []
    for idx,file in enumerate(enc_file_name):
        sampled_enc=np.load(file,'r')
        if idx ==0:
            sampled_encodings= sampled_enc
        else: 
            sampled_encodings= np.concatenate([sampled_encodings,sampled_enc],axis=0)
    print ('sampled_encodings',sampled_encodings.shape)


    # dateset_class=Peptide_condition_dataset
    # condition_encodings_2,df_con15_2=get_encodings(data_path,condtion_model,condition_param,condition_model_path,dateset_class,base=base,filename=filename)
    dateset_class=extension_dataset_label
    extension_encodings,raw_df=get_encodings(data_path,extension_model,extension_param,extension_model_path,dateset_class,base=base,filename=filename)

    # with h5py.File(raw_data_encoding_path, 'r') as hf:
    #     data=np.array(hf['data'])
    #     print (data.shape)
    #     extension_encodings=data[:,:100]


    # sampled_encodings,_=get_seq_encodings(sampled_seqs,extension_model,extension_param,extension_model_path,dateset_class)
    print ('extension_encodings',extension_encodings.shape)
    print ('sampled_encodings',sampled_encodings.shape)



    complete_encoding= np.concatenate([extension_encodings,sampled_encodings],axis=0)




    from sklearn.manifold import TSNE
    t_sne = TSNE(n_components=2,perplexity=30,random_state=2022)
    print ('complete_encoding_total',complete_encoding.shape)
    encoding_tsne = t_sne.fit_transform(complete_encoding)
    rank_df= pd.read_csv(data_path)
    labels = rank_df['label'].to_list()

    get_prop=lambda x:raw_df[x].to_list()
    I_scs=get_prop(prop_to_plot)




    sort=np.sort(I_scs)
    th1 = sort[int(0.1*len(I_scs))]
    th2 = sort[int(0.2*len(I_scs))]
    th3 = sort[int(0.3*len(I_scs))]
    th4 = sort[int(0.4*len(I_scs))]


    cat=[]
    for ele in I_scs:
        if ele <= th1:
            cat.append(4-0)
        elif ele>th1 and ele <= th2:
            cat.append(4-1)
        elif ele>th2 and ele <= th3:
            cat.append(4-2)
        elif ele>th3 and ele <= th4:
            cat.append(4-3)
        else:
            cat.append(4)

    N_raw=len(extension_encodings)

    tsne_df_raw= {'x':encoding_tsne[:N_raw,0],'y':encoding_tsne[:N_raw,1],'cat':cat}
    tsne_df_raw=pd.DataFrame(tsne_df_raw)
    ax=sns.kdeplot(data=tsne_df_raw, x="x", y="y", hue="cat",fill=True)


    tsne_df_sampled= {'x':encoding_tsne[N_raw:,0],'y':encoding_tsne[N_raw:,1]}
    tsne_df_sampled=pd.DataFrame(tsne_df_sampled)
    sns.scatterplot(data=tsne_df_sampled, x="x", y="y",ax=ax)
    plt.show()

    # from sklearn.decomposition import PCA 
    # pca = PCA(n_components=2)
    # encoding_pca = pca.fit_transform(complete_encoding)
    # pca_df= {'x':encoding_pca[:,0],'y':encoding_pca[:,1],'cat':labels+[2]*len(sampled_encodings),'evaluation':[0]*len(extension_encodings)+[1]*len(sampled_encodings)}
    # pca_df=pd.DataFrame(pca_df)
    # sns.scatterplot(data=pca_df, x="x", y="y", hue="cat",style='evaluation', palette="deep")
    # plt.show()