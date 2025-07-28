import pandas as pd 
import numpy as np 
import seaborn as sns 
from matplotlib import pyplot as plt 









 



df1=pd.read_csv('results_Cterminal-avg.csv')
df2=pd.read_csv('results_Cterminal-v2-avg.csv')
# print (df1)
# print (df2)

with_IL = lambda x: 'no_IL' in x
IL_df1 = df1[df1['filename'].apply(with_IL)].reset_index()
IL_df2 = df2[df2['filename'].apply(with_IL)].reset_index()
print (IL_df1)
print (IL_df2)

N1=len(IL_df1)
N2=len(IL_df2)
print ('N1',N1)
print ('N2',N2)

to_list= lambda x: IL_df1[x].to_list()+IL_df2[x].to_list()

df_sc= {'df':[0]*N1+[1]*N2,'I_sc':to_list('I_sc'),'I_bsa':to_list('I_bsa'),'rmsALL_if':to_list('rmsALL_if'),'pep_sc':to_list('pep_sc')}
df_sc=pd.DataFrame(df_sc)


# print (df_sc)
fig, axs = plt.subplots(2, 2)
sns.histplot(data=df_sc, x="I_sc", hue="df",ax=axs[0,0])
sns.histplot(data=df_sc, x="I_bsa", hue="df",ax=axs[0,1])
sns.histplot(data=df_sc, x="rmsALL_if", hue="df",ax=axs[1,0])
sns.histplot(data=df_sc, x="pep_sc", hue="df",ax=axs[1,1])
plt.show()



# fig, axs = plt.subplots(2, 2)
# sns.violinplot(data=df_sc, x="df", y="I_sc",ax=axs[0,0])
# sns.violinplot(data=df_sc, x="df", y="I_bsa",ax=axs[0,1])
# sns.violinplot(data=df_sc, x="df", y="rmsALL_if",ax=axs[1,0])
# sns.violinplot(data=df_sc, x="df", y="pep_sc",ax=axs[1,1])
plt.show()




