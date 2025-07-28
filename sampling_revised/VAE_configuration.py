import torch
import numpy as np

device=torch.device('cpu')

extension_dataset_params={
'max_seq_len': 9, #7,  ###  max extension length
'condition_length': 13, #13,#15, ### conditional peptide length 
'max_peptide_length': 22,#22, ###  max length of the peptide 
}


condition_dataset_params={
'max_seq_len':0,  ### conditional sequence length 
}





### Dataset settings (No need to change) ###




token={'unk':0,'pad':1,'start':2,'eos':3,'A':4,'R':5,'N':6,'D':7,'C':8,'E':9,'Q':10,'G':11,'H':12,'I':13,'L':14,'K':15,
'M':16,'F':17,'P':18,'S':19,'T':20,'W':21,'Y':22,'V':23}

token_reverse={0:'unk',1:'pad',2:'start',3:'eos',4:'A',5:'R',6:'N',7:'D',8:'C',9:'E',10:'Q',11:'G',12:'H',13:'I',14:'L',15:'K',
16:'M',17:'F',18:'P',19:'S',20:'T',21:'W',22:'Y',23:'V'}


dataset_params={
'token':token,  ### conditional sequence length 
}