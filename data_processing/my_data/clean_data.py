import pandas as pd 
import numpy as np 


letter_pool=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']


def check_unique_letter(seqs_to_check):
	letters=[]
	for seq in seqs_to_check: 
		letters.extend(list(set(seq)))
	unique_letters=set(letters)
	print (unique_letters)



def clean_unlabeled_data():
	df=pd.read_csv('unlabeled_revise.csv',keep_default_na=False)
	seqs=df['seq']
	check_unique_letter(seqs)
	good_seqs=[]
	lens=[]
	save=True
	print ('N data before filtering',len(seqs))
	### remove sequences with artificiall amino acid 
	for seq in seqs:
		good_seq=[]
		for STR in seq:
			if STR not in letter_pool:
				save=False  
				break
			else:
				good_seq.append(STR)

		if save==True:
			if len(good_seq)>=16:  ### filter out seq whose length < 16
				# good_seq.reverse()
				good_seq=''.join(good_seq)
				good_seqs.append(good_seq)
		save=True 
	check_unique_letter(good_seqs)
	print ('N data after filtering',len(good_seqs))

	# new_df={'text':good_seqs,'ic_50':['None']*len(good_seqs)}
	# print (good_seqs)
	# sys.exit(0)
	new_df={'seq':good_seqs}
	new_df=pd.DataFrame(new_df)
	new_df.to_csv('unlabeled_revised_2.csv')


def clean_labeled_data():
	df=pd.read_csv('labeled_data.csv',keep_default_na=False)
	df = df.reset_index()
	seqs=df['seq']
	check_unique_letter(seqs)
	good_seqs=[]
	ic_50s=[]
	lens=[]
	save=True
	print ('N data before filtering',len(seqs))
	### remove sequences with artificial amino acid 
	for index, row in df.iterrows():
		seq=row['seq']
		ic_50=row['IC_50']
		good_seq=[] 
		for STR in seq:
			if STR not in letter_pool:
				save=False  
				break
			else:
				good_seq.append(STR)

		if save==True:
			if len(good_seq)>=16:  ### filter out seq whose length < 16
				# good_seq.reverse()
				good_seq=' '.join(good_seq)
				good_seqs.append(good_seq)
				ic_50s.append(ic_50)
		save=True 
	check_unique_letter(good_seqs)
	print ('N data after filtering',len(good_seqs))

	new_df={'text':good_seqs,'ic_50':ic_50s}

	new_df=pd.DataFrame(new_df)
	new_df.to_csv('labeled_new_2.csv')

clean_unlabeled_data()
# clean_labeled_data()