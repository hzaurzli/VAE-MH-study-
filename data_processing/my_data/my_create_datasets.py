### create dataset for our own model ###
import pandas as pd 
import numpy as np 



def add_label_to_csvfile(in_file,coln_name,label): 
	### add labels to all seqs in the file  
	### 1: valid IC_50  0: otherwise 
	df=pd.read_csv(in_file)
	
	n_rows=df.shape[0]
	n_colns=df.shape[1]
	# print (n_colns)
	df.insert(n_colns,coln_name,[label]*n_rows)
	return df

def filter_delta_G(df,th): 
	df_ed=df[df['dG_cross']<=th]
	return df_ed
	


def keep_colns_csv_file(df,coln_name_keep):
	df_ed=df[coln_name_keep]
	# print (df_ed)
	return df_ed



def uniprot_df(file_path,label):
	file_ID=open(file_path)
	seqs=[]
	for idx,line in enumerate(file_ID.readlines()):
		if ' ' not in line:
			seqs.append([line.strip('\n'),label])
	df=pd.DataFrame(seqs,columns=['seq','IC_50'])
	return df

def process_label_uniprot_data_to_csv(file_paths,label):
	### process_uniprot_data
	dfs=[uniprot_df(file,label=label) for file in file_paths]
	concat_df=pd.concat(dfs)
	return concat_df 


def process_thpdb(in_file):
	### process thpdb 
	# print ('in_file',in_file)
	# sys.exit(0)
	# print ('in_file',in_file)
	df=pd.read_excel(in_file)
	# print ('df',df['Therapeutic peptide Name'])
	# sys.exit(0)
	df=df.drop_duplicates(subset ="Therapeutic peptide Name", keep = 'first', inplace = False)['Peptide Sequence']
	df.to_csv('test.csv')
	# sys.exit(0)
	return df



def window_slice(seq,ws):
	dict={'seq':[]}
	for i in range(len(seq)-ws):
		# print (seq[i:i+ws])
		dict['seq'].append(seq[i:i+ws])
	
	new_df=pd.DataFrame(dict)
	return new_df  
	

def random_slice(seq,window_size_range): 
	# print ('seq',seq)
	if len(seq)<window_size_range[-1]:
		window_size_range=np.arange(2,len(seq)+1)
	# else: 
	# 	window_size_range=np.arange(2,len(seq)+1)

	# print ('window_size_range',window_size_range)

	
	ws=np.random.choice(window_size_range,1)[0]
	# print ('ws',ws)
	


	start_pos=np.random.choice(len(seq)-ws+1)
	sliced_seq=seq[start_pos:start_pos+ws]
	return sliced_seq


def slicing_thpdb(df,previous_existing_seqs,window_size_range,N_number):
	all_seqs=list(df['seq'])
	existing_seqs=[]
	while len(existing_seqs)<N_number:
		seq_to_slice=np.random.choice(all_seqs,1)[0]
		sliced_seq=random_slice(seq_to_slice,window_size_range)
		if sliced_seq not in existing_seqs+previous_existing_seqs:
			existing_seqs.append(sliced_seq)
		print (len(existing_seqs))
	new_df=pd.DataFrame({'seq' : existing_seqs})

	# new_df=pd.DataFrame({'seq' : []})

	# for STR in df['seq']: 
	# 	new_df=new_df.append(window_slice(STR,ws))
	# print ('new_df',new_df)
	return new_df







def combine_all_csv_files(dfs): 
	### assume all files have the same colnumns 
	### dfs: data frame from different files 
	concat_df=pd.concat(dfs)
	# print (concat_df)
	return concat_df



def get_length_less_than(df,th): 
	return df[df['seq'].apply(lambda x:len(x)<=th)]


def get_length_greater_than(df,th): 
	return df[df['seq'].apply(lambda x:len(x)>=th)]



def replace_with_x(Str):
	Str_result=''
	# print ('Im here')
	for i,c in enumerate(Str):
		# print ('Im here')
		# print (c)
		if c.islower()==1:
			c='x'
		Str_result+=c	
	return Str_result

def replace_synthetic_with_x(df):
	### replace synthetic amoid acid with lower case x 
	func=lambda x: replace_with_x(x)
	new_seq=df['seq'].apply(func)
	df=df.drop(['seq'], axis=1)
	df.insert(0,'seq',new_seq)
	return df


def add_letter_space(Str):
	Str_result=''
	for i,c in enumerate(Str):
		Str_result+=c+' '
	return Str_result


def add_space(df):
	### add space for each seq  
	func=lambda x: add_letter_space(x)
	new_seq=df['text'].apply(func)
	df=df.drop(['text'], axis=1)
	df.insert(0,'text',new_seq)
	return df




def search_letter_space(Str):
	# Str_result=''
	for i,c in enumerate(Str):
		if c=='_':
			print ('im here')
			print (Str)
		# Str_result+=c+' '
	return Str


def search_space(df):
	### add space for each seq  
	func=lambda x: search_letter_space(x)
	new_seq=df['text'].apply(func)
	return 


def add_label_to_df(df,coln_name,label): 
	### add labels to all seqs in the file  
	### 1: valid IC_50  0: otherwise 
	n_rows=df.shape[0]
	n_colns=df.shape[1]
	# print (n_colns)
	# print ('df',df)
	df.insert(n_colns,coln_name,[label]*n_rows)
	# print ('df',df)
	# sys.exit(0)
	return df



def reverse_str_in_csv(df):
	### add space for each seq  
	func=lambda x: x[::-1]
	new_seq=df['text'].apply(func)
	df=df.drop(['text'], axis=1)
	df.insert(0,'text',new_seq)
	return df




# a='as4as5d'
# search_letter_space(a)

if __name__=='__main__':
	# ###### set raw data folder path to parse the data ###  
	In_folder='raw_data/'  ### raw data folder 
	Out_folder='./' ### folder to save processed data 

	#### No need to change ####
	### db file ###
	file_1000valid_path=In_folder+'db_result.csv'
	db_df=add_label_to_csvfile(file_1000valid_path,'IC_50',1)
	db_df=filter_delta_G(db_df,-35)
	db_df=keep_colns_csv_file(db_df,['seq','IC_50'])
	# db_df=get_length_less_than(db_df,th=25)
	db_df=get_length_greater_than(db_df,th=2)
	# print ('db_df',len(db_df))

	# db_df.to_csv(Out_folder+'new_db_result.csv')

	### thpdb ### 
	thpdb_file_path=In_folder+'thpdb.csv'
	previous_existing_seqs=[]
	label_dfs=[]
	thpdb_df=pd.read_csv(thpdb_file_path)
	thpdb_df=get_length_greater_than(thpdb_df,th=2)
	window_size_range=np.arange(2,9)

	N_seqs_per_boot=10000

	for repeat in range(4):
		label_df= pd.concat([thpdb_df,db_df])
		label_df= slicing_thpdb(label_df,previous_existing_seqs,window_size_range,N_number=N_seqs_per_boot)
		previous_existing_seqs.extend(list(label_df['seq']))
		label_df=add_label_to_df(label_df,'IC_50',1)
		label_dfs.append(label_df)
	labeled_seqs=list(combine_all_csv_files(label_dfs)['seq'])

	labeled_seqs=[]

	### uniprot ###
	uniprot_file_path=[In_folder+'uniprot_review_no.fasta',In_folder+'uniprot_review_yes.fasta']
	uniprot_df=process_label_uniprot_data_to_csv(uniprot_file_path,label=0)
	uniprot_df=get_length_less_than(uniprot_df,th=50)
	uniprot_df=get_length_greater_than(uniprot_df,th=2)


	# previous_existing_seqs=[]
	

	# uniprot_df=get_length_less_than(uniprot_df,th=25)
	uniprot_df=uniprot_df.drop_duplicates(subset ="seq", keep = 'first', inplace = False)

	print (len(uniprot_df))
	# sys.exit(0)
	negative_seqs=[]
	while len(negative_seqs)<= N_seqs_per_boot*4:
		# uniprot_df_sampled=uniprot_df.sample(10000*4)
		uniprot_df_sampled = slicing_thpdb(uniprot_df,negative_seqs,window_size_range,N_number=100)
		uniprot_df_sampled=list(set(list(uniprot_df_sampled['seq']))-set(labeled_seqs)-set(negative_seqs))
		negative_seqs.extend(uniprot_df_sampled)
		print ('len(negative_seqs)',len(negative_seqs))
	negative_seqs=negative_seqs[:N_seqs_per_boot*4]
	uniprot_df_sampled=pd.DataFrame({'seq':negative_seqs,'IC_50':[0]*len(negative_seqs)})
	
	# print (thp_dfs[0])
	# sys.exit(0)
	labeled_dfs=combine_all_csv_files([uniprot_df_sampled]+label_dfs)

	# ###  Apply filter ####
	# ### 1. seq <= 50 
	# ### 2. replace all lower case letter with 'x'


	print ('total labeled data number',len(labeled_dfs))
	labeled_dfs=replace_synthetic_with_x(labeled_dfs)
	# print (labeled_dfs)
	labeled_dfs=labeled_dfs.drop_duplicates(subset ="seq", keep = 'first', inplace = False)
	
	# ft_df=reverse_str_in_csv(ft_df)
	# ft_df=add_space(ft_df);
	print ('total labeled data number',len(labeled_dfs))
	labeled_dfs.to_csv(Out_folder+'labeled_data_warm_up.csv',index=0)

	sys.exit(0)



	#### unbelled data ####
	unlabeled_dfs=combine_all_csv_files([db_df,uniprot_df]+thp_dfs)
	unlabeled_dfs=unlabeled_dfs.drop(columns=['IC_50'])
	print ('total unlabeled data number',len(unlabeled_dfs))
	unlabeled_dfs=unlabeled_dfs.drop_duplicates(subset ="seq", keep = 'first', inplace = False)
	unlabeled_dfs.to_csv(Out_folder+'unlabled_data.csv',index=0)
	print ('total unlabeled data number',len(unlabeled_dfs))




