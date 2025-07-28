import torch
import VAE_configuration as Params
from torch.utils.data import Dataset




class extension_dataset_label(Dataset):
    def __init__(self,df,has_label):
        self.device=Params.device
        self.data=df.reset_index()
        self.token=Params.dataset_params['token']
        self.max_len=Params.extension_dataset_params['max_seq_len']
        self.condition_length=Params.extension_dataset_params['condition_length'] 
        self.has_label=has_label
        # print ('self.condition_length',self.condition_length)
        # print ('self.max_len',self.max_len)
        # sys.exit(0)


    def __tokenize__(self,seq):
        token=self.token
        seq_list=[token['start']]
        # print ('seq',seq)
        left_seq=seq[self.condition_length:]
        # print (len(left_seq))
        # print (seq)
        # print ('self.condition_length',self.condition_length)
        
        # print ('seq',seq)
        # print ('left_seq',left_seq)
        # sys.exit(0)
        seq_len=len(left_seq)
        for letter in left_seq:  
            if letter in token.keys():
                seq_list.append(token[letter])
            else: 
                seq_list.append(token['unk'])
        if seq_len<self.max_len: 
            seq_list.append(token['eos'])
            seq_list.extend([token['pad']]*(self.max_len-seq_len))
        else: 
            seq_list=seq_list[:self.max_len+1]  ### +1 due to start token 
            seq_list.append(token['eos'])
        # print (seq_list)
        # sys.exit(0)
        return seq_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data['seq'][idx]
        seq_token=self.__tokenize__(seq)
        seq_token=torch.tensor(seq_token,device=self.device)
        if self.has_label==True:
            label = [self.data['label'][idx]]
            label = torch.tensor(label,device=self.device)
            return seq_token,label
        else: 
            return seq_token



def idxTostr(indices,base):
    ## also check valid tokens ##
    all_seqs=[]
    good_tokens=[]
    good_idx=[]
    for token_idx,seq_token in enumerate(indices): 
        seq = []
        for i,idx in enumerate(seq_token): ## skip 1 due to start token 
            if i==0 and idx!=2:  ## invalid token 
                break 
            if 3 not in seq_token: ### end tpken must present 
                break 
            if i == 0: ## no need to append start token 
                continue 
            if idx ==3: ### ignore tokens after end token 
                break 
            else:
                aa =  Params.token_reverse[idx]
                if aa == 'unk' or aa == 'pad' or aa == 'start':
                    break  ### invalid token 
                else:    
                    seq.append(Params.token_reverse[idx])
        if len(seq)==0:
            continue 
        else:
            all_seqs.append(base+''.join(seq))
            good_tokens.append(seq_token)
            good_idx.append(token_idx)

    return all_seqs,good_tokens,good_idx