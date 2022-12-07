import pandas as pd
from tqdm import tqdm
import pickle
import argparse
from not_mean_extract_embedding_vectors import return_embedding_vectors_from_dict
parser = argparse.ArgumentParser(description='main file')


parser.add_argument('--number',
                     type = str,
                     help = "path of pickle file containing embedding vectors")
args = parser.parse_args()
print(args.number)
with open('/mimer/NOBACKUP/groups/snic2022-6-127/split_combined_prot.pickle', 'rb') as handle:
    dc= pickle.load(handle)
dc = dc[f"var_{args.number}"]
prot_seq = []
prot_name = []
for i,j in dc.items():
    prot_seq.append(str(j))
    prot_name.append(str(i))
try:
    with open(f'/mimer/NOBACKUP/groups/snic2022-6-127/PPI_prediction/pickle/not_mean_embedded_vectors_bert_bfd_combined_{args.number}.pickle', 'rb') as handle:
        dc= pickle.load(handle)
except:
    dc = {}
embedded_vectors = {}
l = 0
for i in tqdm(range(0,len(prot_seq),320)):
    l = l + 1
    print(i+320)
    print(args.number)
    vectors = return_embedding_vectors_from_dict(prot_seq[i:i+320],prot_name[i:i+320])
    dc.update(vectors)
    if l%20 == 0:
        
        try:
            with open(f'/mimer/NOBACKUP/groups/snic2022-6-127/PPI_prediction/pickle/not_mean_embedded_vectors_bert_bfd_combined_{args.number}.pickle', 'rb') as handle:
                x= pickle.load(handle)
                x.update(dc)
            with open(f'/mimer/NOBACKUP/groups/snic2022-6-127/PPI_prediction/pickle/not_mean_embedded_vectors_bert_bfd_combined_{args.number}.pickle', 'wb') as handle:
                pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
            dc.clear()
            x.clear()
            x = {}
            dc = {}
        except: 
            with open(f'/mimer/NOBACKUP/groups/snic2022-6-127/PPI_prediction/pickle/not_mean_embedded_vectors_bert_bfd_combined_{args.number}.pickle', 'wb') as handle:
                pickle.dump(dc, handle, protocol=pickle.HIGHEST_PROTOCOL)
            dc.clear()
            dc = {}
            
            
with open(f'/mimer/NOBACKUP/groups/snic2022-6-127/PPI_prediction/pickle/not_mean_embedded_vectors_bert_bfd_combined_{args.number}.pickle', 'rb') as handle:
    x= pickle.load(handle)
    x.update(dc)
    with open(f'/mimer/NOBACKUP/groups/snic2022-6-127/PPI_prediction/pickle/not_mean_embedded_vectors_bert_bfd_combined_{args.number}.pickle', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
