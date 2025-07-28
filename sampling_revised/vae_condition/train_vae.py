import torch
import pandas as pd 
import torch.nn.functional as F
import numpy as np

from model.model import RNN_VAE
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import VAE_configuration as Params
from shutil import copyfile
# from torchsummary import summary
# from torchviz import make_dot
from matplotlib import pyplot as plt 
import losses

if Params.log_results:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(Params.Result_root+'runs')






class Peptide_dataset(Dataset):
    def __init__(self,df):
        self.device=Params.device
        self.data=df
        self.token=Params.dataset_params['token']
        self.max_len=Params.dataset_params['max_seq_len'] 


    def __tokenize__(self,seq):
        token=self.token
        seq_list=[token['start']]
        for letter in seq[:self.max_len]:  
            if letter in token.keys():
                seq_list.append(token[letter])
            else: 
                seq_list.append(token['unk'])
        seq_list.append(token['eos'])
        return seq_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data['seq'][idx]
        seq_token=self.__tokenize__(seq)
        seq_token=torch.tensor(seq_token,device=self.device)
        return seq_token




def recon_dec(gt_seq, logits):
    PAD_IDX=Params.dataset_params['token']['pad']
    recon_loss = F.cross_entropy(  # this is log_softmax + nll
        logits.view(-1, logits.size(2)), gt_seq.view(-1), reduction='mean',
        ignore_index=PAD_IDX  # padding doesnt contribute to recon loss & gradient
    )
    return recon_loss


def kl_gaussianprior(mu, logvar):
    """ analytically compute kl divergence with unit gaussian. """
    return torch.mean(0.5 * torch.sum((logvar.exp() + mu ** 2 - 1 - logvar), 1))


def pz_losses(z,z_mu,z_logvar,cur_step):
    if Params.use_kl==False:
        z_regu_loss  = losses.wae_mmd_gaussianprior(z, method='rf')
    else:
        z_regu_loss = kl_gaussianprior(z_mu, z_logvar)
    
    z_logvar_L1 = z_logvar.abs().sum(1).mean(0)  # L1 in z-dim, mean over mb.
    z_logvar_KL_penalty = losses.kl_gaussian_sharedmu(z_mu, z_logvar)
    beta=losses.anneal(cur_step)

    pz_loss = beta * z_regu_loss \
           +  Params.loss_weights['lambda_logvar_L1']* z_logvar_L1 \
           + Params.loss_weights['lambda_logvar_KL']* z_logvar_KL_penalty

    return pz_loss


def running_avg(avg_val,cur_val,it):
    it=it+1  ## due to the model is 1 based 
    return avg_val*(it-1)/it+cur_val/it


def calculate_loss(model,dataset,cur_step,is_train):

    seq_token = next(dataset)
    # print (seq_token)
    # sys.exit(0)
    (z_mu, z_logvar), z, dec_logits = model(seq_token)

    ### Terry: add constrastive learning ###
    # z shape : n x 100
    # bio: n x 1  
    # pairwise(z): distance (n x n)
    # pairwise(bio): distance (n x n) -> threshod -> nxn binary matrix  
    # constrastive (paiewsie z and pairwise bio): n x n:real number, return avg     


    recon_loss = recon_dec(seq_token, dec_logits)
    pz_loss = pz_losses(z,z_mu,z_logvar,cur_step)
    # kl_loss = kl_gaussianprior(z_mu, z_logvar)
    # total_loss=recon_loss*Params.loss_weights['recon']+kl_loss*Params.loss_weights['kl']
    total_loss=recon_loss*Params.loss_weights['recon']+pz_loss*Params.loss_weights['pz']
    cur_losses=[total_loss,recon_loss,pz_loss]

    if is_train==True:
        return cur_losses

    else: 
        original_seqs=seq_token[:2]
        sampled_seqs=model.sampling_sequences(z_mu[:2])
        N_mismatch=torch.sum(sampled_seqs!=original_seqs)
        cur_losses=cur_losses+[N_mismatch]

        sampled_seqs_shift=model.sampling_sequences_shift(z_mu[:2])
        N_mismatch_shift=torch.sum(sampled_seqs_shift!=original_seqs)
        cur_losses=cur_losses+[N_mismatch_shift]
        print (original_seqs)
        print (sampled_seqs)
        print (sampled_seqs_shift)
        
        sys.exit(0)
        return cur_losses



def step_model(model,cur_epoch,dataset,optimizer,require_grad=True):
    
    dataset_iter=iter(dataset)
    if require_grad:
        avg_losses=[0,0,0]
        for it in range(len(dataset)): #tqdm(range(len(train_dataloader)),disable=None):
            cur_step=cur_epoch*(len(dataset_iter))+it
            cur_losses =calculate_loss(model,dataset_iter,cur_step,is_train=require_grad)
            model.zero_grad()
            optimizer.zero_grad()
            cur_losses[0].backward()
            optimizer.step()
            with torch.no_grad():
                avg_losses=[running_avg(avg_losses[idx],cur_losses[idx],it).item() for idx in range(len(avg_losses))]
    else:
        with torch.no_grad():
            avg_losses=[0,0,0,0,0]
            all_pred_props=[]
            all_gt_props=[]
            for it in range(len(dataset)): #tqdm(range(len(train_dataloader)),disable=None):
                cur_step=cur_epoch*(len(dataset_iter))+it
                cur_losses=calculate_loss(model,dataset_iter,cur_step,is_train=require_grad)
                avg_losses=[running_avg(avg_losses[idx],cur_losses[idx],it).item() for idx in range(len(avg_losses))]
    return avg_losses




def train(training_params,model_params,dataset_params,device):
    TP=training_params
    MP=model_params
    DP=dataset_params


    print('Training base vae ...')


    if Params.resume_train == False:
        from model.model import RNN_VAE
        model=RNN_VAE(**MP)
        optimizer = torch.optim.Adam(model.parameters(), lr=TP['lr'])
    else:
        import sys
        sys.path.insert(0,Params.training_info_folder)
        import configure_file as params
        #### load trained model files ####
        from model_file import RNN_VAE
        MP=params.RNN_VAE_params
        model=RNN_VAE(**MP)
        optimizer = torch.optim.Adam(model.parameters(), lr=TP['lr'])
        checkpoint = torch.load(Params.resume_model_path, map_location=Params.device)
        model.load_state_dict(checkpoint)
        
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # sys.exit(0)
        # model.load_state_dict(torch.load(Params.resume_train_path))
        # optimizer = torch.optim.Adam(model.parameters(), lr=TP['lr'])
        # optimizer = model.load_state_dict(torch.load(PATH))



    np.random.seed(2021)
    df=pd.read_csv(DP['data_path'])
    
    if Params.debug==True:
        df=df[:1000]
    
    df=df.sample(frac=1).reset_index(drop=True)
    max_seq_len=DP['max_seq_len']
    train_eval_ratio=DP['train_eval_ratio']
    # df=Peptide_dataset(df,preprocess=True,plot=False,normalization_params=None,previous_N_data=0).data

    N_seq=len(df)
    N_train=int(N_seq*train_eval_ratio)
    df_train=df[:N_train].sample(frac=1).reset_index(drop=True)
    train_dataset= Peptide_dataset(df_train)
    df_test=df[N_train:].sample(frac=1).reset_index(drop=True)
    test_dataset= Peptide_dataset(df_test)
    cur_test_loss=1000
    # sys.exit(0)
    print ('len train data',len(df_train))
    print ('len test data',len(df_test))
    


    for epoch in range(TP['epochs']):#tqdm(range(epochs),disable=None): 
        train_dataloader = DataLoader(train_dataset, batch_size=TP['train_batch_size'], shuffle=TP['shuffle'])
        train_losses= step_model(model,epoch,train_dataloader,optimizer,require_grad=True)
        print ('Train epoch:'+str(epoch),train_losses)
        
        test_dataloader = DataLoader(test_dataset, batch_size=TP['test_batch_size'], shuffle=True)
        test_losses=step_model(model,epoch,test_dataloader,optimizer,require_grad=False)
        print ('Test epoch:'+str(epoch),test_losses)


        if cur_test_loss>=test_losses[0]:
            cur_test_loss=test_losses[0]
            torch.save(model.state_dict(), TP['model_save_folder']+'/model_{}.ckpt'.format(epoch))
        if epoch % 20 ==0:
            torch.save(model.state_dict(), TP['model_save_folder']+'/model_{}.ckpt'.format(epoch))
        if epoch % 100 ==0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses
            }, TP['model_save_folder']+'/train_model_{}.ckpt'.format(epoch))

        if Params.log_results:
            train_setup={'total_loss':train_losses[0],'recon_loss':train_losses[1],
            'pz_loss':train_losses[2]}
            test_setup={'total_loss':test_losses[0],'recon_loss':test_losses[1],
            'pz_loss':test_losses[2],'N_mismatch':test_losses[3],'N_mismatch_shift':test_losses[4]}
            [writer.add_scalar('Train/'+key, value, epoch) for key,value in train_setup.items()]
            [writer.add_scalar('Test/'+key, value, epoch) for key,value in test_setup.items()]
    torch.cuda.empty_cache()



if __name__=='__main__':
    copyfile('VAE_configuration.py', Params.Result_root+'configure_file.py')
    copyfile('model/model.py', Params.Result_root+'model_file.py')
    device=Params.device
    training_params=Params.training_params
    RNN_VAE_params=Params.RNN_VAE_params
    dataset_params=Params.dataset_params
    train(training_params,RNN_VAE_params,dataset_params,device)