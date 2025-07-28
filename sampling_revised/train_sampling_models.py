import torch
import pandas as pd 
import numpy as np
import sys
from shutil import copyfile

from torch import nn
import pickle 


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

import h5py
import os 



def train_qcz(data,N_boots,model_name,save_folder):
    # all_train_encodings=np.load('train_encoding_x_14.npy')
    # all_labels=np.load('train_encoding_y_14.npy')
    data=data
    data_1 = data[data[:,-1]==1]
    data_0 = data[data[:,-1]==0]
    np.random.shuffle(data_1)
    np.random.shuffle(data_0)

    ratio = 0.8

    train_N_1=int(len(data_1)*ratio)
    train_N_0=int(len(data_0)*ratio)
    training= np.concatenate([data_1[:train_N_1],data_0[:train_N_0]],axis=0)
    np.random.shuffle(training)

    # train_N_1=2
    # train_N_0=2

    testing= np.concatenate([data_1[train_N_1:],data_0[train_N_0:]],axis=0)
    np.random.shuffle(testing)

    all_train_encodings=training[:,:-1]
    all_labels=training[:,-1]
    # print ('all_labels',np.sum(all_labels==0))
    # print ('all_labels',np.sum(all_labels==1))
    # sys.exit(0)

    if N_boots==1:
        sample_N = all_train_encodings.shape[0]
    else:
        sample_N = int(all_train_encodings.shape[0]*0.8)


    boot_models=[]
    for N_boot in range(N_boots):
        print ('N_boot:',N_boot)
        random_idx=np.random.permutation(all_train_encodings.shape[0])[:sample_N]
        train_encodings=all_train_encodings[random_idx]
        labels=all_labels[random_idx]
        if model_name=='LR':
            LRC = LogisticRegression(random_state=0).fit(train_encodings, labels)
        elif model_name=='SVC':
            LRC = SVC(probability=True,kernel='poly').fit(train_encodings, labels)
        elif model_name=='XGB':
            LRC = XGBClassifier(eval_metric='auc')
        elif model_name=='KNN':
            LRC = KNeighborsClassifier(n_neighbors=3,metric='cosine')

        LRC.fit(train_encodings,labels)
        # model_name='SVC'+str(N_boot)+'.sav'
        directory=save_folder+'/'+model_name
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_file=directory+str(N_boot)+'.sav'

        # model_name='sampling_models/qcz/LR/LR'+str(N_boot)+'.sav'
        pickle.dump(LRC, open(model_file, 'wb'))
        # LRC=pickle.load(open(model_name, 'rb'))
        boot_models.append(LRC)



    def boot_prediction(models,X): 
        probs=[]
        for model in models: 
            probs.append(model.predict_proba(X))
        # print (np.shape(probs))       
        probs=np.mean(probs,axis=0)[:,1]
        # print (np.shape(probs))   
        pred_results=probs>0.5
        return probs,pred_results

    # print ('im here')


    train_encodings=training[:,:-1]
    train_labels=training[:,-1]
    probs,pred=boot_prediction(boot_models,train_encodings)   
    # test_label=np.load('test_encoding_y_14.npy')
    train_acc=np.divide(np.sum(train_labels==pred),len(train_labels))
    idx_1=pred==1
    tp=np.sum((pred[idx_1]==train_labels[idx_1]))
    fp=np.sum((pred[idx_1]!=train_labels[idx_1]))
    if (tp+fp)==0:
        precision = 0 
    else:
        precision = tp/(tp+fp)
    print ('stats:','acc',train_acc,'len pred positive',np.sum(idx_1),'precision',precision)
    return boot_models



def train_qz(encodings,model_name,n_components,save_folder):
    # with h5py.File(condition_filename+'.h5', 'r') as hf:
    #   con_encodings=np.array(hf['data'])
    #   print (con_encodings.shape)

      # sys.exit(0)
    # encodings=np.concatenate([con_encodings,ext_encodings],axis=1)
    # print (encodings.shape)
    # encodings=np.load('train_encoding_x_14.npy')
    # print (encodings.shape)
    # seqs,good_encodings,good_embedings,good_idx=MU.decode(encodings,obtain_embeding=True,allow_pad=True)
    from sklearn.mixture import GaussianMixture #50
    gm = GaussianMixture(n_components=n_components, random_state=0).fit(encodings)
    # model_name='GMM2_14.sav'
    directory=save_folder
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(gm, open(directory+'/'+model_name, 'wb'))
    return gm



def load_model_test():
    def boot_prediction(models,X): 
        probs=[]
        for model in models: 
            probs.append(model.predict_proba(X))
        # print (np.shape(probs))       
        probs=np.mean(probs,axis=0)[:,1]
        # print (np.shape(probs))   
        pred_results=probs>0.5
        return probs,pred_results
    folder='sampling_models/qcz/SVC/'
    boot_model_files=[folder+'SVC0.sav',folder+'SVC1.sav',folder+'SVC2.sav']
    boot_models=[]
    for boot_model_file in boot_model_files:
        boot_models.append(pickle.load(open(boot_model_file,'rb')))

    test_encoding=np.load('test_encoding_x.npy')
    print ('im here 1')
    probs,pred=boot_prediction(boot_models,test_encoding)   
    print ('im here 2')
    test_label=np.load('test_encoding_y.npy')
    acc=np.divide(np.sum(test_label==pred),len(test_label))
    print ('acc',acc)   
    


if __name__=='__main__':
    # load_model_test()
    train_qcz()
    # condition_filename = 'unlabel_condition_train'
    
    # extension_filename = 'unlabel_extension13_train'
    # model_name = 'GMM_ext.sav'
    # train_qz(extension_filename,model_name,n_components=5)


    filename='label_extension13_train'
    with h5py.File(filename+'.h5', 'r') as hf:
        data=np.array(hf['data'])


    with h5py.File(extension_filename+'.h5', 'r') as hf:
      ext_encodings=np.array(hf['data'])
      print (ext_encodings.shape)
