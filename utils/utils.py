from msilib import sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys


def preprocessing(data):

    data = data[['seq_1','seq_2','label']]
    data['concat_protein'] = data['seq_1'] +" " + data['seq_2']
    print(data.head(20))
    return data

def train_test_split_function(data):
    X  = data['concat_protein']
    y = data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    return X_train, X_test, y_train, y_test


def preprocessing_for_bert(protein_sequences):
    input_ids = []
    attention_masks = []
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
    for protein_sequence in tqdm(protein_sequences):
        id = tokenizer.encode_plus(protein_sequence,max_length = 1002,pad_to_max_length = True, truncation=False)
        input_id = id.get('input_ids')
        attention_mask = id.get('attention_mask')
        input_ids.append(input_id)
        attention_masks.append(attention_mask)                         
    
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    
    return input_ids, attention_masks
    



def prot_to_seq_string(data):
    id =[]
    sequence = []
    name_to_seq_dict = {}
    from Bio import SeqIO
    for seq_record in SeqIO.parse("9606.protein.sequences.v11.5.fa", "fasta"):
        id.append(seq_record.id)
        x = repr(seq_record.seq)[5:-2]
        l = ""
        for i in x:
            l = l + i + " "
        l = l.rstrip()
        sequence.append(l)
    name_to_seq_dict = dict(zip(id, sequence))
    data['seq_1'] = data.apply (lambda row: name_to_seq_dict[row['protein1']], axis=1)
    data['seq_2'] = data.apply (lambda row: name_to_seq_dict[row['protein2']], axis=1)
    return data