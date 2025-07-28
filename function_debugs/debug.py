import numpy as np 






token={'unk':0,'pad':1,'start':2,'eos':3,'A':4,'R':5,'N':6,'D':7,'C':8,'E':9,'Q':10,'G':11,'H':12,'I':13,'L':14,'K':15,
'M':16,'F':17,'P':18,'S':19,'T':20,'W':21,'Y':22,'V':23}
token_reverse={0:'unk',1:'pad',2:'start',3:'eos',4:'A',5:'R',6:'N',7:'D',8:'C',9:'E',10:'Q',11:'G',12:'H',13:'I',14:'L',15:'K',
16:'M',17:'F',18:'P',19:'S',20:'T',21:'W',22:'Y',23:'V'}



def idxTostr(indices,base):
    ## also check valid tokens ##
    all_seqs=[]
    for seq_idx in indices: 
        seq = []
        for i,idx in enumerate(seq_idx): ## skip 1 due to start token 
            if i==0 and idx!=2:  ## invalid token 
                break 
            if 3 not in seq_idx: ### end tpken must present 
                break 
            if i == 0: ## no need to append start token 
                continue 
            if idx ==3: ### ignore tokens after end token 
                break 
            else:
                aa =  token_reverse[idx]
                if aa == 'unk' or aa == 'pad' or aa == 'start':
                    break  ### invalid token 
                else:    
                    seq.append(token_reverse[idx])
        if len(seq)==0:
            continue 
        else:
            all_seqs.append(base+''.join(seq))
    return all_seqs



indices=np.array([[2, 22, 22, 22, 22, 22, 22, 22, 18],[2, 23, 23, 5, 5, 5, 5, 5, 5]])
base=''
seqs=idxTostr(indices,base)
print (seqs)
