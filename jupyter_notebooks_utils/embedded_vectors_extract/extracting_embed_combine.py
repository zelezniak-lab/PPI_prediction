import pandas as pd
from tqdm import tqdm
import pickle
from extract_embedding_vectors import return_embedding_vectors_from_dict
with open('/cephyr/users/anwer/PPI_prediction/pickle/prot_to_seq_pickle/combined_prot.pickle', 'rb') as handle:
    dc= pickle.load(handle)
prot_seq = []
prot_name = []
for i,j in dc.items():
    prot_seq.append(str(j))
    prot_name.append(str(i))
try:
    with open('/mimer/NOBACKUP/groups/snic2022-6-127/embedded_vectors_bert_bfd_mean_combined.pickle', 'rb') as handle:
        dc= pickle.load(handle)
except:
    dc = {}
embedded_vectors = {}
for i in tqdm(range(0,len(prot_seq),40)):
    print(i+40)
    
    vectors = return_embedding_vectors_from_dict(prot_seq[i:i+40],prot_name[i:i+40])
    dc.update(vectors)
    with open('/mimer/NOBACKUP/groups/snic2022-6-127/embedded_vectors_bert_bfd_mean_combined.pickle', 'wb') as handle:
        pickle.dump(dc, handle, protocol=pickle.HIGHEST_PROTOCOL)