import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.optim as optim

class BertClassifier(nn.Module):
    
    def __init__(self, embed_dim = 1024):
        super(BertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim,512)
        self.fc2 = nn.Linear(512,256)
        
        self.fc3 = nn.Linear(512,64)
        self.fc4 = nn.Linear(64,32)
        
        self.fc5 = nn.Linear(32,16)
        self.fc6 = nn.Linear(16,8)
        self.fc7 = nn.Linear(8,1)

    
    def forward(self, input_ids,attention_mask):
        
        embedding = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        output = torch.tensor(embedding[0][:, 0, :])
        # print(output)
        output_1 = self.relu(self.fc1(output))
        output_2 = self.relu(self.fc3(output_1))
        output_3 = self.relu(self.fc4(output_2))
        output_4 = self.relu(self.fc5(output_3))
        output_5 = self.relu(self.fc6(output_4))
        output = self.fc7(output_5)

        return output




def initilize_model():
    """Instantiate a CNN model and an optimizer."""

    model = BertClassifier()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Send model to `device` (GPU/CPU)
    model.to(device)
    model= nn.DataParallel(model,device_ids = [0])

    return model
