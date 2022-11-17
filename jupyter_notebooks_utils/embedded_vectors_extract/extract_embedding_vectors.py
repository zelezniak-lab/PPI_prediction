import argparse
import torch
from transformers import BertModel, BertTokenizer, AlbertModel, AlbertTokenizer, XLNetModel, XLNetTokenizer
import re
import os
import requests
from tqdm.auto import tqdm


parser = argparse.ArgumentParser(description='Extract embedding vectors')

parser.add_argument('--bert_model',
                      type =str,
                      default = "prot_bert_bfd",
                      help = "Choose bert model")
parser.add_argument('--input_path_pickle',
                     type = str,
                     default = "seq.pkl",
                     help = "path of pickle file containing sequences")

parser.add_argument('--output_path_pickle',
                     type = str,
                     default = "embed.pkl *",
                     help = "path of pickle file containing embedding vectors")




def return_embedding_vectors_from_dict(sequences,protein_name, model_name):
    """ 
    Return embedding vectors from Bert model

    Args:
        sequences: dict having sequences and protein name
        model_name (_type_): name of bert model
    """

    tokenizer = BertTokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False)
    model = BertModel.from_pretrained(f"Rostlab/{model_name}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
    embedding = embedding.cpu().numpy()
    embedded_vectors_dct = {}
    for i in tqdm(range(len(embedding))):
        embedded_vectors_dct[protein_name[i]] = embedding[i][0]
    
    return embedded_vectors_dct 










