import pandas as pd 
import sys 
import torch  
import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import h5py
import os 

# sys.path.insert(0,'../vae_condition')
# from train_vae import Peptide_dataset as condition_dataset
# from results.con_15_300 import configure_file as condition_params 
# from results.con_15_300.model_file import RNN_VAE as condition_model 


# sys.path.insert(0,'../vae_extension')
# from train_vae import Peptide_dataset as extension_dataset
# from train_vae import Peptide_dataset_ext_labeled as extension_dataset_label
# from results.con_13 import configure_file as extension_params 
# from results.con_13.model_file import RNN_VAE as extension_model 






def save_train_encodings(data_path,model_PATH,model,param,dataset,filename,save_folder):
  # dataset = AttributeDataLoader(mbsize=92351, max_seq_len=cfg.max_seq_len,
  #                           device=device,
  #                           attributes=cfg.attributes,
  #                           **cfg.data_kwargs)

  df=pd.read_csv(data_path,keep_default_na=False)
  dataset= dataset(df,has_label=True)
  MP=param.RNN_VAE_params
  model = model(**MP).to(param.device)
  model.load_state_dict(torch.load(model_PATH))
  model.eval()

  dataloader = DataLoader(dataset, batch_size=512, shuffle=False)

  enc_file_folder = save_folder+'/'+filename+'.h5'
  if not os.path.exists(enc_file_folder):
    for idx,(seq_tokens,labels) in enumerate(dataloader):
      with torch.no_grad():
        (z_mu, z_logvar), z, dec_logits = model(seq_tokens)
        z_mu=z_mu.cpu().numpy()
        labels=labels.cpu().numpy()
        data = np.concatenate([z_mu,labels],axis=1)
        with h5py.File(save_folder+'/'+filename+'.h5', 'a') as hf:
          if idx ==0: 
            hf.create_dataset('data', data=data,compression="gzip",chunks=True, maxshape=(None,data.shape[1]))
          else: 
            N_z=z_mu.shape[0]
            hf['data'].resize(hf['data'].shape[0]+N_z,axis=0)
            hf['data'][-N_z:]=data

  
  with h5py.File(save_folder+'/'+filename+'.h5', 'r') as hf:
    data=np.array(hf['data'])
    print (data.shape)

  return data










