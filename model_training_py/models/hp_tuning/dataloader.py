from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import timeit

import os
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from matplotlib.pyplot import figure


class Data(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)



def load_pickle(pickle_path,num_of_pickle = 7):
    """_load pickle file_

    Args:
        path (str, optional): _description_. Defaults to "/home/anwer/Desktop/sum_pickle_not_mean/".
        num_of_pickle (int, optional): _description_. Defaults to 7.

    Returns:
        _type_: _description_
    """

    dc = {}
    l = 0 
    for i in os.listdir(pickle_path):
        path = os.path.join(pickle_path,i)
        with open(pickle_path,'rb') as handle:
            d = pickle.load(handle)
        l = l+1
    
        dc.update(d)
        if l == num_of_pickle:
            break
        
    return dc


def load_csv(path_train,path_test):
    """_summary_

    Args:
        path_train (str, optional): _description_. Defaults to "../../../csv_files_new_ppi/training_and_test_set/combined_prot_a_prot_b//combined_train_only_name.csv".
        path_test (str, optional): _description_. Defaults to "../../../csv_files_new_ppi/training_and_test_set/combined_prot_a_prot_b//combined_test_only_name.csv".

    Returns:
        _type_: _description_
    """

    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)
    return train, test


def return_embed(dct,prot_name):
    """_return embedding_vectors_

    Args:
        dct (_type_): dict which map protein_name to embedding vectors
        prot_name (_type_): protein_name
    """

    try:
        return dct[prot_name]
    except:
        return np.nan




def load_data(path_train = "../../../csv_files_new_ppi/training_and_test_set/combined_prot_a_prot_b//combined_train_only_name.csv"
        ,path_test = "../../../csv_files_new_ppi/training_and_test_set/combined_prot_a_prot_b//combined_test_only_name.csv",pickle_path ="/home/anwer/Desktop/sum_pickle_not_mean/"):
    """_summary_

    Args:
        path_train (str, optional): _description_. Defaults to "../../../csv_files_new_ppi/training_and_test_set/combined_prot_a_prot_b//combined_train_only_name.csv".
        path_test (str, optional): _description_. Defaults to "../../../csv_files_new_ppi/training_and_test_set/combined_prot_a_prot_b//combined_test_only_name.csv".
        pickle_path (str, optional): _description_. Defaults to "/home/anwer/Desktop/sum_pickle_not_mean/".
    """

    train,test = load_csv(path_train,path_test)
    dct_prot_to_embed = load_pickle(pickle_path=pickle_path)
    train['embed_vec'] = train["Protein_name"].apply(return_embed)
    test['embed_vec'] = test["Protein_name"].apply(return_embed)
    train = train.dropna()
    test = test.dropna()
    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)
    val = test.iloc[50000:]
    test = test.iloc[0:50000]
    val_neg = val[val.label == 0][0:5000]
    val_pos = val[val.label == 1][0:5000]
    val = pd.concat([val_neg,val_pos])

    return train,test,val



def dataloader()







