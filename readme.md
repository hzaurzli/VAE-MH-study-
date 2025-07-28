** System requirements **
1. Operation system: Ubuntu 16 
2. Python dependencies: See environment.yml 
3. Hardware: GPU >= Nvidia 1080

** Installtion guide **
1. Instructions: 
	a. Download Anaconda 
	b. Create anaconda environment: conda env create -f environment_partial.yaml
	c. Activate the environment: conda activate test_env 

** Run sampling with provided models locally **
1. In this setup, the VAE models are provided 
2. cd sampling_revised
3. python run.py (note you can setup the settings in the file) 

** Run sampling with provided models on cloud **
1. We use the code peer-review service provided by Code Ocean for the public accessibility of the code and the data 
2. Go to Code Ocean: Peer review link shall be provided via the journal before publication. Public link will be provided here after the publication.  
3. Click 'Reproducible run' 

** train VAE model for peptide condtion locally**
1. cd vae_condition 
2. set configuration in VAE_condiguration.py 
3. python train_vae.py


** train VAE model for peptide extension locally**
1. cd vae_extension 
2. Set configuration in VAE_condiguration.py 
3. python train_vae.py

** Data availability ** 
1. VAE training data (unlabelled) are stored in 'data_processing/my_data'
2. qcz model training data (labelled) are stored in 'sampling_revised/labelled_data'

** Raw data processing (Optional) **
1. We have provided the processed data along with the code. If you want to start from the scracth, please do the following: 
2. Place the "raw_data" folder under data_processing/my_data/. "raw_data" is provided here: https://drive.google.com/drive/folders/1GRaZ1MttvQ2iy1kN-76cwZxYGZo98hER?usp=sharing
3. run "my_create_datasets.py" to create unlabeled_data and labeled_data 
4. run "clean_data.py" to further process the unlabeled_data and the labeled_data. This produces "labeled_new_2.csv" and "unlabeled_new_2.csv".
5. Place the "labeled_new_2.csv" file into 'sampling_revised/labelled_data'. 


