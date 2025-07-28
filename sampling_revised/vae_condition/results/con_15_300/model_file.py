import torch
import torch.nn as nn
import VAE_configuration as Params
import numpy as np 


class WordDropout(nn.Module):
    def __init__(self, p_word_dropout):
        super(WordDropout, self).__init__()
        self.p = p_word_dropout

    def forward(self, x):
        """
        Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
        """
        data = x.clone().detach()

        # Sample masks: elems with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p, size=tuple(data.size()))
                .astype('uint8')
        ).to(x.device)

        mask = mask.bool()
        # Set to <unk>
        data[mask] = Params.dataset_params['token']['unk']

        return data




class GRUEncoder(nn.Module):
    """
    Encoder is GRU with FC layers connected to last hidden unit
    """
    def __init__(self, 
                 emb_dim,
                 h_dim,
                 z_dim,
                 biGRU,
                 layers,
                 p_dropout):
        super(GRUEncoder, self).__init__()
        self.rnn = nn.GRU(input_size=emb_dim, 
                          hidden_size=h_dim, 
                          num_layers=layers,
                          dropout=p_dropout,
                          bidirectional=biGRU,
                          batch_first=True)
        # Bidirectional GRU has 2*hidden_state
        self.biGRU_factor = 2 if biGRU else 1
        self.biGRU = biGRU
        # Reparametrization
        self.q_mu = nn.Linear(self.biGRU_factor*h_dim, z_dim)
        self.q_logvar = nn.Linear(self.biGRU_factor*h_dim, z_dim)

    def forward(self, x):
        """
        Inputs is embeddings of: mbsize x seq_len x emb_dim
        """
        _, h = self.rnn(x, None)
        if self.biGRU:
            # Concatenates features from Forward and Backward
            # Uses the highest layer representation
            h = torch.cat((h[-2,:,:], 
                           h[-1,:,:]), 1)
        # Forward to latent
        h = h.view(-1, h.shape[-1])
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)
        return mu, logvar



class GRUDecoder(nn.Module):
    """
    Decoder is GRU with FC layers connected to last hidden unit
    """

    def __init__(self,
                 embedding,
                 emb_dim,
                 output_dim,
                 # input_zc_dim,
                 h_dim,
                 p_word_dropout,
                 p_out_dropout,
                 skip_connetions):
        super(GRUDecoder, self).__init__()
        # Reference to word embedding
        self.emb = embedding
        self.rnn = nn.GRU(emb_dim, h_dim,
                          batch_first=True)
        # self.init_hidden_fc = nn.Linear(input_zc_dim, h_dim) # TODO use this for initial hidden state?
        self.fc = nn.Sequential(
            nn.Dropout(p_out_dropout),
            nn.Linear(h_dim, output_dim))
        self.word_dropout = WordDropout(p_word_dropout)

        self.skip_connetions = skip_connetions
        if self.skip_connetions:
            self.skip_weight_x = nn.Linear(h_dim, h_dim, bias=False)
            self.skip_weight_z = nn.Linear(h_dim, h_dim, bias=False)



    def forward(self, x, z):
        mbsize, seq_len = x.shape
        # mbsize x (z_dim + c_dim)  -- required shape (num_layers * num_directions, batch, hidden_size)
        init_h = z
        dec_inputs = self.emb(self.word_dropout(x))
        expanded_init_h = init_h.unsqueeze(1).expand(-1, seq_len, -1)
        dec_inputs = torch.cat([dec_inputs, expanded_init_h], 2)
        rnn_out, _ = self.rnn(dec_inputs, init_h.unsqueeze(0))

        # apply skip connection
        if self.skip_connetions:
            rnn_out = self.skip_weight_x(rnn_out) + self.skip_weight_z(expanded_init_h)
        y = self.fc(rnn_out)
        return y,rnn_out




class RNN_VAE(nn.Module):

    def __init__(self,
                 n_vocab,
                 max_seq_len,
                 z_dim,
                 emb_dim,
                 PAD_IDX,
                 device,
                 GRUEncoder_params,
                 GRUDecoder_params):


        super(RNN_VAE, self).__init__()
        # self.MAX_SEQ_LEN = max_seq_len
        # self.n_vocab = n_vocab
        # self.z_dim = z_dim
        self.device = Params.device

        self.n_vocab=n_vocab
        self.emb_dim=emb_dim
        self.PAD_IDX=PAD_IDX
        self.word_emb = nn.Embedding(self.n_vocab, self.emb_dim, PAD_IDX).to(self.device)   ## embedding layer 
        
        self.z_dim=z_dim
        self.GRUEncoder_params=GRUEncoder_params
        self.encoder = GRUEncoder(**self.GRUEncoder_params,emb_dim=self.emb_dim,z_dim=self.z_dim).to(self.device)


        self.output_dim=self.n_vocab
        self.h_dim_decoder=self.z_dim
        self.decoder_input_dim=self.z_dim+self.emb_dim
        self.decoder = GRUDecoder(embedding=self.word_emb,emb_dim=self.decoder_input_dim,
            output_dim=self.output_dim,h_dim=self.h_dim_decoder,**GRUDecoder_params).to(self.device)



    def forward_encoder(self, inputs):
        ''' 
        Inputs is batch of sentences: seq_len x mbsize
               or batch of soft sentences: seq_len x mbsize x n_vocab.
        ''' 
        # print (inputs)
        inputs = self.word_emb(inputs)
        mu, logvar= self.encoder(inputs)
        return mu, logvar


    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
        """
        eps = torch.randn(mu.size(0), self.z_dim).to(self.device)
        return mu + torch.exp(logvar / 2) * eps

    def sample_z_prior(self, mbsize):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = torch.randn(mbsize, self.z_dim).to(Params.device)
        return z


    def forward_decoder(self, inputs, z):
        """
        Inputs are indices: seq_len x mbsize
        """
        return self.decoder(inputs, z)


    def forward(self, sequences):
        # print ('sequences',sequences)
        # sys.exit(0)
        mbsize = sequences.size(0)
        mu, logvar = self.forward_encoder(sequences)
        z = mu 
        assert mu.size(0) == logvar.size(0) == mbsize       
        dec_logits,embeddings = self.forward_decoder(sequences, z)


        return (mu, logvar), z, dec_logits


    def get_embedding(self,sequences):
        with torch.no_grad():
            mbsize = sequences.size(0)
            mu, logvar = self.forward_encoder(sequences)
            z = mu 
            assert mu.size(0) == logvar.size(0) == mbsize       
            dec_logits,embeddings = self.forward_decoder(sequences, z)

        return embeddings


    def sampling_sequences(self,encoding): 
        with torch.no_grad():
            mbsize = encoding.size(0)
            z = encoding 
            assert z.size(0) == mbsize
            sequences=torch.ones(mbsize,1,device=encoding.device)*Params.token['start']
            sequences=sequences.to(torch.int64)
            for pos in range(Params.dataset_params['max_seq_len']+1):   ## +2 for start and end token      
                dec_logits,embeddings = self.forward_decoder(sequences, z)
                selects=torch.argmax(dec_logits[:,-1:,:],dim=2)
                sequences=torch.cat((sequences,selects),1)
            # print ('sequences',sequences)
            # sys.exit(0)
            sequences=sequences #[:,1:] ### Exclude the random init sequence, which is selected as start token in this case
            return sequences

    def sampling_sequences_shift(self,encoding): 
        with torch.no_grad():
            mbsize = encoding.size(0)
            z = encoding 
            assert z.size(0) == mbsize
            sequences=torch.ones(mbsize,1,device=encoding.device)*Params.token['start']
            sequences=sequences.to(torch.int64)
            for pos in range(Params.extension_dataset_params['max_seq_len']+2):   ## +2 for start and end token      
                dec_logits,embeddings = self.forward_decoder(sequences, z)
                selects=torch.argmax(dec_logits[:,-1:,:],dim=2)
                sequences=torch.cat((sequences,selects),1)
            sequences=sequences[:,1:] ### Exclude the random init sequence, which is selected as start token in this case
            return sequences

    def sampling_embedding(self,encoding):
        with torch.no_grad():
            mbsize = encoding.size(0)
            z = encoding 
            assert z.size(0) == mbsize
            sequences=torch.ones(mbsize,1,device=encoding.device)*Params.token['start']
            sequences=sequences.to(torch.int64)
            for pos in range(Params.dataset_params['max_seq_len']+2):   ## +2 for start and end token  
                if pos==0:    
                    dec_logits,embeddings = self.forward_decoder(sequences, z)
                else: 
                    dec_logits,embeddings_cur = self.forward_decoder(sequences, z)
                    embeddings=torch.cat((embeddings,embeddings_cur),0)
                selects=torch.argmax(dec_logits[:,-1:,:],dim=2)
                sequences=torch.cat((sequences,selects),1)
            sequences=sequences[:,1:] ### Exclude the random init sequence, which is selected as start token in this case
            
            return sequences,embeddings




