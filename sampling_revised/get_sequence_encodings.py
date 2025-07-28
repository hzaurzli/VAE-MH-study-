import pandas as pd 
import numpy as np 
from torch.utils.data import DataLoader
import torch

import VAE_configuration as Params







def get_encodings(data_path,model,param,model_path,peptide_dataset,base,filename):
    df=pd.read_csv(data_path, keep_default_na=False)
    # get_file = lambda x: filename in x
    # df=df[df['filename'].apply(get_file)].reset_index()
    # df['seq']= [base+s for s in df['Seq'].to_list()]
    dataset= peptide_dataset(df,has_label=False)
    MP=param.RNN_VAE_params
    model = model(**MP).to(Params.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)


    for idx,seq_tokens in enumerate(dataloader):

        with torch.no_grad():
          (z_mu, z_logvar), z, dec_logits = model(seq_tokens)
          if idx==0:
            z_final=z_mu.numpy()
          else: 
            z_final=np.concatenate([z_final,z_mu.numpy()],axis=0)
    print (z_final.shape)
    return z_final,df           



def get_seq_encodings(seqs,model,param,model_path,peptide_dataset):
    df=pd.DataFrame({'seq':seqs})
    dataset= peptide_dataset(df,has_label=False)
    MP=param.RNN_VAE_params
    model = model(**MP).to(Params.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    for idx,seq_tokens in enumerate(dataloader):

        with torch.no_grad():
            (z_mu, z_logvar), z, dec_logits = model(seq_tokens)
            if idx==0:
                z_final=z_mu.numpy()
            else: 
                z_final=np.concatenate([z_final,z_mu.numpy()],axis=0)
    print (z_final.shape)
    return z_final,df           


def enc2seq(encodings,model,param,model_path):
    encodings=torch.tensor(encodings,device=Params.device).float()
    MP=param.RNN_VAE_params
    model = model(**MP).to(Params.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tokens_sampled=model.sampling_sequences_shift(encodings)
    return tokens_sampled       




