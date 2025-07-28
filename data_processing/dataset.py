import random
import csv
from collections import defaultdict, OrderedDict
import copy
import os
import io
import pandas as pd
import torch
import torchtext as tt
import codecs
# from models.model import UNK_IDX, PAD_IDX, START_IDX, EOS_IDX
import numpy as np

UNK_IDX = 0
PAD_IDX = 1
START_IDX = 2
EOS_IDX = 3

class AttributeField(tt.data.RawField):
    def __init__(self, name, mappingdict):
        self.name = name
        self.mappingdict = mappingdict
        self.itos = {ix: attr_label for attr_label, ix in self.mappingdict.items()}
        self.is_target = True

    def preprocess(self, x):
        # print ('x',x)
        # print ('self.mappingdict',self.mappingdict)
        
        if not x in self.mappingdict:
            raise KeyError('Key {} not in mappingdict (keys: {}) for attribute {}'.format(
                x, ','.join(self.mappingdict.keys()), self.name))
        return self.mappingdict[x]

    def process(self, x, device, *args, **kwargs):
        return torch.LongTensor(x).to(device)


class ReadOnlyVocab(object):
    def __init__(self, vocab_path):
        self.itos = []
        self.stoi = {}
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            for line in f:
                # print ('line',line)
                lsp = line.split()
                word = " ".join(lsp[:-1])
                ix = int(lsp[-1])
                assert len(self.itos) == ix, 'assuming vocab is in order'
                self.itos.append(word)
                self.stoi[word] = int(ix)

    def __len__(self):
        return len(self.itos)


class PandaDataset(tt.data.Dataset):
    """ make a torchtext Dataset from a pandas DataFrame. """

    def __init__(self, sourcedf, fields):
        # sys.exit(0)
        self.sourcedf = sourcedf.fillna('na')
        examples = self.sourcedf.Example.values  # column in dataframe
        # sys.exit(0)
        if 'sample_weights' in self.sourcedf.columns:
            self.sample_weights = torch.from_numpy(self.sourcedf['sample_weights'].values)
        # print ('examples',fields.values())
        # NOTE converting fields OrderedDict -> list for torchtext constructor
        super(PandaDataset, self).__init__(examples, fields.values())


class WeightedRandomIterator(tt.data.Iterator):
    """ iterator that uses tt.dataset and sample_weights,
    infinitely iterates with batches sampled according to multinomial with replacement"""

    def __init__(self, dataset, sample_weights, batch_size, device):
        super(WeightedRandomIterator, self).__init__(dataset, batch_size, device=device, repeat=True)
        self.sample_weights = sample_weights

    def init_epoch(self):
        if self._restored_from_state:
            raise NotImplementedError

    def __iter__(self):
        self.init_epoch()
        while True:
            indices = torch.multinomial(self.sample_weights, self.batch_size, replacement=True)
            minibatch = [self.dataset[i] for i in indices]
            yield tt.data.Batch(minibatch, self.dataset, self.device)


class MultiCsvReader:
    """ Class to read multiple csv files into single pandas dataframe.
    Provides functions to subset and create weighted samplers/iterators on the resulting
    dataset."""

    def __init__(self, path, csv_files, max_seq_len, fields,
                 csv_reader_params={}):
        self.fields = fields
        # Read across csv files; construct lookup. Then reduce to list.
        self.data = defaultdict(dict)  # eg: {pep_str: {text: "M A C ...", idx: 0, amp: amp_pos, tox: tox_neg}, ... }
        # print ('self.data.values()',self.data.values())
        for csv_file in csv_files: ### add all rows of all csv files to self.data
            # print ('csv_file:',csv_file)
            
            # TODO pd.read_csv and df.combine_first(df) could be more efficient.
            #  http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.combine_first.html
            fn = os.path.join(path, csv_file)
            # print('Load csv file', fn)
            # sys.exit(0)
            self._parse_csv(fn, csv_reader_params)
        # sys.exit(0)
        # Done with construction, make list of Example s from dicst of dicts
        # print('Construct torchtext Example objects')
        # print ('self.data.values()',self.data.values())
        # sys.exit(0) 


        for data_dict in self.data.values():
            # print ('data_dict', data_dict)
            # sys.exit(0)
            self.make_and_insert_example(data_dict)
        # Make pandas dataframe
        print('Make and filter pandas dataframe')
        # print (self.data)
   
        self.df = pd.DataFrame([self.data[k] for k in sorted(self.data.keys())])
        # self.df.to_csv('test.csv')
        # sys.exit(0)
        self.df['lens'] = self.df.text.apply(lambda x: len(x.strip().split()))
        self.df = self.df[self.df.lens <= max_seq_len]
        # self.df.to_csv('test.csv')
        # sys.exit(0)
        
        # print('df len: ', len(self.df))
        # print('df columns: ', self.df.columns)
        # print('df columns: ', np.sum(self.df['IC_50'].apply(lambda x:x=='1')))
        # sys.exit(0)


    def _parse_csv(self, fn, csv_reader_params):
        """ deviating from torchtext, just care about python 3 here """
        # print ('aaaaa',fn)
        # print ('self.data',self.data)
        # sys.exit(0)
        # print ('fn',fn)

        # print ('aaaaa', **csv_reader_params)
        # sys.exit(0)
        with io.open(fn, encoding="utf8") as f:
            # print ('csv read param',csv_reader_params['amp'])
            # sys.exit(0)
            reader = csv.DictReader(f, **csv_reader_params)
            # print ('fn',fn)
            # print ('reader',reader)
            # sys.exit(0)
            # print ('self.data',self.data)
            # sys.exit(0)
            for entry in reader:
                # print ('entry',entry)
                # sys.exit(0)
                k = entry['text'].strip()
                # print ('reader',reader)
                # print ('k',k)
                # print ('entry',entry)
                # print ('self.data',self.data[k])
                # sys.exit(0)
                # print (self.data)
                self._insert_csv_entry(k, entry)
                # print (self.data)
                 # print ('self.data',self.data)
                # sys.exit(0)
            # sys.exit(0)
    def _insert_csv_entry(self, key, entry):
        # print (key)
        # print ('self.data[key]',self.data[key])
        # sys.exit(0)
        exdic = self.data[key]
        exdic.update(entry)

    def make_example(self, data_dict):
        """ create Example from dict with possible missing values.
        Follows @classmethod constructors of class Example"""
        cls = tt.data.Example
        fields = self.fields
        ex = cls()
        for key, vals in fields.items():
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    # print ('data_dict',data_dict)
                    # print ('name', name)
                    # print ('field', field)
                    # print ('key content',data_dict.get(key, 'na'))
                    setattr(ex, name, field.preprocess(data_dict.get(key, 'na')))
        return ex

    def make_and_insert_example(self, data_dict):
        """ makes Example from data_dict then adds as data_dict['Example'] (modifies dict) """
        # print ('data_dict',data_dict)
        data_dict['Example'] = self.make_example(data_dict)

    def compute_splits(self, ratios, random_seed):
        assert len(ratios) == 3 and sum(ratios) == 1.0, 'provide train/val/test split ratio'
        LEN = len(self.df) ## Tong: total number of the data 
        # print (self.df)
        # print (LEN)
        # sys.exit(0)
        rix = self.df.index.tolist()
        random.Random(random_seed).shuffle(rix)
        a, b = int(ratios[0] * LEN), int(ratios[1] * LEN)
        print ('train N',a)
        print ('test N',b)
        print ('eval N',LEN-a-b)
        
        trainix, valix, testix = rix[:a], rix[a:a + b], rix[a + b:]
        # print ('ratios',ratios)
        # print ('len(trainix)',len(trainix))
        # sys.exit(0)
        self.df.loc[self.df.index.isin(trainix), 'split'] = 'train'
        self.df.loc[self.df.index.isin(valix), 'split'] = 'val'
        self.df.loc[self.df.index.isin(testix), 'split'] = 'test'
        print('TRAIN SPLIT CHECK first 10 train seqs (note: ordered alphabetically)')
        print(self.df.text[self.df.split == 'train'][:10])

    def get_subset(self, *colspecifiers):
        """ Returns subset wrapped in PandaDataset, colspecifiers see get_subset_df() """
        ss = self.get_subset_df(*colspecifiers)
        # print ('self.df',self.df)
        # print ('self.df',self.fields)
        # print ('ss',self.fields)
        # sys.exit(0)
        # print ('ss',ss)
        # search_space(ss)
        # sys.exit(0)
        return PandaDataset(ss, self.fields)

    def get_subset_df(self, *colspecifiers):
        """ list of column specifiers, which is a mini language to mask/include
        only data points that have attributes present/absent/specific values.
        supported operators:
        col=v1,v2 # col is one of the specifieds vals v1 or v2
        col       # col must be present (not null)
        ^col      # col must be absent (null)
        """
        mask = True
        # print ('colspecifiers',colspecifiers)
        # sys.exit(0)
        i=0
        for cs in colspecifiers:
            # print ('colspecifiers',colspecifiers)
            # print ('cs',cs)
            # i+=1
            # print (i)
            # sys.exit(0)  
            # print ('cs',cs)
            # self.df.to_csv('test.csv')
            # sys.exit(0)
            mask = mask & self.get_mask(self.df, cs)
            # print ('self.df',self.df)
            # print ('cs',cs)
            # print ('mask',sum(mask))
        # sys.exit(0)
        return self.df[mask].copy()

    def get_mask(self, df, colspecifier):
        if '=' in colspecifier:
            k, allowed_vals = colspecifier.split('=')
            
            #### temp modified ####
            # key=allowed_vals.split(',')
            # if 'amp_pos' in allowed_vals.split(',')[0]:
            #     key=['amp_pos']
            # elif 'tox_pos' in allowed_vals.split(',')[0]:
            #     key=['tox_pos']
            # mask = df[k].isin(key) ### should not be 
            #### ends #### 
            # print ('k',k)
            # print ('allowed_vals',allowed_vals)       
            # sys.exit(0)
            # print ('df',df)
            # print ('df',df[k])
            # sys.exit(0)
            mask = df[k].isin(allowed_vals.split(','))  ### original 
            # print ('mask',mask)
            # sys.exit(0)

        else:
            if colspecifier[0] == '^':
                k = colspecifier[1:]
                mask = df[k].isna()
            else:
                k = colspecifier
                # print ('colspecifier',colspecifier)
                mask = df[k].notna()
        # print ('mask',mask)
        return mask

    def df_add_sample_weights(self, df, sample_prob_factors={}, sample_weights={}):
        """ takes a df, and based on specifiers for column values,
        add a new column "sample_prob" for weighted sampling. Either through
        * sample probability factors: upsample X times wrt base rate
        * sample weights: specify exact fractions per column specifier.
        """
        if sample_prob_factors:
            # print ('df',df)
            df.loc[:, 'sample_weights'] = 1.0
            # sys.exit(0)
            for colspecifier, factor in sample_prob_factors.items():
                # print ('colspecifier',colspecifier)
                # print ('factor',factor)
                # sys.exit(0)
                # if colspecifier=='sol' or 'anticancer':
                #     continue 
                mask = self.get_mask(df, colspecifier)
                assert mask.sum() > 0, 'empty mask for colspecifier {}'.format(colspecifier)
                mask = mask & (df['sample_weights'] < factor)  # max(factor, existing)
                df.loc[mask, 'sample_weights'] = factor
            df.loc[:, 'sample_weights'] /= df.sample_weights.sum()
            # df.to_csv('test.csv')
            # print (df['sample_weights'])
            # sys.exit(0)
        elif sample_weights:
            raise NotImplementedError
            df.loc[:, 'sample_weights'] = 0.0
            # TODO per colspecifier, set sample_weight = class_weight / num_samples
        else:
            df.loc[:, 'sample_weights'] = 1.0 / len(df)

    def get_subset_iterators(self, iteratorspecs, mbsize, device):
        iterators, subsets = {}, {}
        for name, spec in iteratorspecs.items():
            # print('Make subset & iterator', name)
            # print ('spec',spec.pop('subset'))
            # sys.exit(0)
            # print ('name',name)
            # print ('spec',spec)
            # sys.exit(0)
            spec = copy.deepcopy(spec)
            # self.df.to_csv('test.csv')
            # sys.exit(0)
            ss = self.get_subset_df(*spec.pop('subset'))
            # sys.exit(0)
         # print ('ss',ss)
            # sys.exit(0)
            weighted_random_sample = spec.pop('weighted_random_sample', False)
            repeat_iterator = spec.pop('repeat', True)
            
            ### Tong: changed for conditional only ####
            # weighted_random_sample=False
            # repeat_iterator = False
            ### Tong:ends ###

            # print ('weighted_random_sample',weighted_random_sample)
            # print ('repeat_iterator',repeat_iterator)
            # sys.exit(0)


            if weighted_random_sample:
                # print ('ss',ss)
                # sys.exit(0)
                self.df_add_sample_weights(ss, **spec)
                # ss.to_csv(name+'.csv')
                # sys.exit(0)
            # ss.to_csv('test.csv')
            # print  ('sssss',ss.text)
            # sys.exit(0)
            ds = PandaDataset(ss, self.fields)
            # ss.to_csv('test.csv')
            # print  ('sssss',ds.text)
            # sys.exit(0)
            if weighted_random_sample:
                assert repeat_iterator, 'WeightedRandomIterator samples infinitely with replacement'
                iterator = WeightedRandomIterator(ds, ds.sample_weights, mbsize, device=device)
            else:
                iterator = tt.data.Iterator(ds, mbsize, shuffle=True, repeat=repeat_iterator, sort=False, device=device)
            # print ('ds',ds)
            subsets[name] = ds
            iterators[name] = iterator
            # print ('ds ',ds)
            # sys.exit(0)
        # sys.exit(0)
        return iterators, subsets


### Tong: helper funcs to chekc ####

def search_letter_space(Str):
    # Str_result=''
    for i,c in enumerate(Str):
        if c=='e':
            print ('im here')
            print (Str)
        # Str_result+=c+' '
    return Str


def search_space(df):
    ### add space for each seq  
    func=lambda x: search_letter_space(x)
    new_seq=df['text'].apply(func)
    return 


def get_all_data_iterators(self, iteratorspecs, mbsize, device):
    iterators, subsets = {}, {}
    for name, spec in iteratorspecs.items():
        # print('Make subset & iterator', name)
        # print ('spec',spec.pop('subset'))
        # sys.exit(0)
        print ('name',name)
        print ('spec',spec)
        # sys.exit(0)
        spec = copy.deepcopy(spec)
        # self.df.to_csv('test.csv')
        # sys.exit(0)
        ss = self.get_subset_df(*spec.pop('subset'))
        # sys.exit(0)
     # print ('ss',ss)
        # sys.exit(0)

        ### Tong: only for evaluation ###
        # weighted_random_sample = False
        # repeat_iterator = True
        # print ('weighted_random_sample',weighted_random_sample)
        # print ('repeat_iterator',repeat_iterator)
        # sys.exit(0)


        if weighted_random_sample:
            # print ('ss',ss)
            # sys.exit(0)
            self.df_add_sample_weights(ss, **spec)
            # ss.to_csv(name+'.csv')
            # sys.exit(0)
        # ss.to_csv('test.csv')
        # print  ('sssss',ss.text)
        # sys.exit(0)
        ds = PandaDataset(ss, self.fields)
        # ss.to_csv('test.csv')
        # print  ('sssss',ds.text)
        # sys.exit(0)
        if weighted_random_sample:
            assert repeat_iterator, 'WeightedRandomIterator samples infinitely with replacement'
            iterator = WeightedRandomIterator(ds, ds.sample_weights, mbsize, device=device)
        else:
            iterator = tt.data.Iterator(ds, mbsize, shuffle=True, repeat=repeat_iterator, sort=False, device=device)
        # print ('ds',ds)
        subsets[name] = ds
        iterators[name] = iterator
        # print ('ds ',ds)
        # sys.exit(0)
    # sys.exit(0)
    return iterators, subsets



### ends ###


class AttributeDataLoader:
    """ Reads csv, tsv. Combines multiple csv per attribute through MultiCsvReader.
    Splits in train/valid/test, and sets up batched iterators for each of them.
    Exposes `next_batch(iterator_name)`
    """

    def __init__(self, mbsize=32, max_seq_len=15, data_path=None,
                 data_format='csv', lower=False,
                 emb_dim=50, glove_cache=None,
                 attributes=[], csv_files=[],
                 split_seed=1238, iteratorspecs={},
                 fixed_vocab_path='',
                 device=torch.device('cuda')):

        print('Loading Dataset...')

        self.device = device
        self.TEXT = tt.data.Field(init_token='<start>', eos_token='<eos>', sequential=True,
                                  lower=lower, tokenize=self.tokenizer, fix_length=max_seq_len,
                                  batch_first=True)
        # TODO fix_length=None)
        self.ATTRIBUTES = [AttributeField(k, mapping) for k, mapping in attributes]
        self.num_attrs = len(self.ATTRIBUTES)
        # Create "fields" dict for 'Example's
        self.fields = OrderedDict()
        self.fields['text'] = ('text', self.TEXT)
        for a in self.ATTRIBUTES:
            # print ('a',a.name)
            # print (a)
            self.fields[a.name] = (a.name, a)
        # sys.exit(0)
        # print('fields :: ', self.fields)
        # sys.exit(0)
        # Only take sentences with length <= max_seq_len
        
        filt = lambda ex: len(ex.text) <= max_seq_len
        self.max_seq_len = max_seq_len
        # print('self.field',self.fields)
        # sys.exit(0)

        self.dataset = MultiCsvReader(data_path, csv_files, max_seq_len, self.fields)
        self.dataset.compute_splits([0.8, 0.1, 0.1], random_seed=split_seed)
        self.iterators, self.subsets = self.dataset.get_subset_iterators(iteratorspecs, mbsize, device)
        # for k, it in self.iterators.items():
        #     print ('aaaaaaaaaaaa',k)
        #     print ('iter',iter(it))
        #     sys.exit(0)
        self.iterators_ = {k: iter(it) for k, it in self.iterators.items()}
        # inputs = self.next_batch('train_vae')
        # print ('inputs',inputs.text)
        # inputs = self.next_batch('train_vae')
        # print ('inputs',inputs.IC_50)
        # sys.exit(0)
        # a=self.dataset.get_subset('split=train')
        # print (self.dataset.get_subset('split=train').df)
        # print (self.fields)
        # search_space(self.dataset.df)
        # sys.exit(0) 

        if fixed_vocab_path:
            self.TEXT.vocab = ReadOnlyVocab(fixed_vocab_path)
        else:
            self.TEXT.build_vocab(self.dataset.get_subset('split=train'), self.fields)
        # sys.exit(0)
        # inputs = self.next_batch('train_vae')
        # print ('inputs',inputs.text)
        # print ('inputs',inputs.IC_50)
        # sys.exit(0)

        self.n_vocab = len(self.TEXT.vocab)
        for ix, tok in zip([UNK_IDX, PAD_IDX, START_IDX, EOS_IDX], ['<unk>', '<pad>', '<start>', '<eos>']):
            assert self.TEXT.vocab.itos[ix] == tok

    def print_stats(self):
        print('Vocab size:', self.n_vocab)
        print('Vocab (first 50):', ' / '.join(self.TEXT.vocab.itos[:50]))
        for iname, iterator in self.iterators.items():
            print('iterator {:10s}: subset data size: {:7d}. iterator size: {}'.format( \
                iname, len(self.subsets[iname]), len(iterator)))

    def get_vocab_vectors(self):
        return self.TEXT.vocab.vectors

    def tokenizer(self, text):  # create a tokenizer function
        return [tok for tok in str.split(text)]

    def next_batch(self, iterator_name):
        # print ('iterator_name',iterator_name)
        return next(self.iterators_[iterator_name])

    def get_generator(self, iterator_name):
        # print ('iterator_name',iterator_name)
        return self.iterators_[iterator_name]


    def get_all_data(self, iterator_name):
        return next(self.iterators_[iterator_name])


    def idx2sentences(self, idxs, print_special_tokens=True):
        """ recursively descend into n-dim tensor or list and return same nesting """
        if not isinstance(idxs[0], list) and (isinstance(idxs[0], (int, float)) or idxs[0].dim() == 0):
            # 1D, no more nesting
            return self.idx2sentence(idxs, print_special_tokens)
        else:
            return [self.idx2sentences(s, print_special_tokens) for s in idxs]

    def idx2sentence(self, idxs, print_special_tokens=True):
        assert isinstance(idxs, list) or idxs.dim() == 1, 'expecting single sentence here'
        if not print_special_tokens:
            idxs = [i for i in idxs if i not in [UNK_IDX, PAD_IDX, START_IDX, EOS_IDX]]  # filter out
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2attr(self, idx, attr=None):
        if not attr:
            attr = self.fields.keys()[1]  # first attr
        return self.fields[attr][1].itos[idx]

    def idx2label(self, idx):
        # TEMP function, just take first attributes
        return self.idx2attr(idx, self.ATTRIBUTES[0].name)
