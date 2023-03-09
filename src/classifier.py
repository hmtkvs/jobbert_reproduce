import random
import pandas as pd
import numpy as np
from transformers import BertModel, AutoTokenizer
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Jobbert_neg_sampling(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(Jobbert_neg_sampling, self).__init__()

        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)

        # # Initialize the gating layer
        self.gating = nn.Linear(self.bert_model.config.hidden_size, 1)

        # # Initialize the layers   
        self.mlp_layers = nn.Sequential(
                        nn.Linear(self.bert_model.config.hidden_size, 768),
                        nn.ReLU(),
                        nn.Linear(768, 300)
                        )
        
        self.criteron = nn.NLLLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.weight.shape[1] == 1:
                nn.init.uniform_(module.weight, -math.sqrt(1/768), math.sqrt(1/768))
                nn.init.uniform_(module.bias, -1, 1)
            else:
                nn.init.uniform_(module.weight, -math.sqrt(1/768), math.sqrt(1/768))
                nn.init.uniform_(module.bias, -math.sqrt(1/300), math.sqrt(1/300))


    def forward(self, data):
        #print('data.keys(): ', data.keys())
        t = data['input_job_title']
        u_j = self.embeddings_context(data['positive_skill'])
        u_k = self.embeddings_context(data['negative_skills'])

        # # Get the BERT embeddings
        bert_output = self.bert_model(t)['last_hidden_state']
        # # Get the gating scores
        x = self.gating(bert_output).sigmoid()
        # # Get the averaged sum of the gated embeddings
        det = x.sum(dim=1)
        gating_scores = x / det.unsqueeze(dim=-1)
        # # Get the input embeddings
        input_embs = gating_scores * bert_output
        input_embs = self.mlp_layers(input_embs)

        # print('input_embs.shape: ', input_embs.shape)  
        # print('u_j.shape: ', u_j.shape)
        # print('u_k.shape: ', u_k.shape)

        pos_matx = torch.mul(u_j, input_embs)

        aux_embs = input_embs.unsqueeze(2).expand(-1, -1, 5, -1)
        neg_matx = (u_k * aux_embs).sum(dim=2).neg()

        J_T = F.logsigmoid(pos_matx) + F.logsigmoid(neg_matx)
        J_T = J_T.sum(dim=1)
        # print('J_T.shape: ', J_T.shape)
        # print(u_j.view(-1, 300).shape)

        target = torch.tensor(u_j.view(-1, 300).long())
        target = torch.argmax(target ,axis=1)

        loss = self.criteron(J_T, target)

        return loss
    

# model = Jobbert_neg_sampling(embedding_size=300, vocab_size=num_of_unique_skills)
# data_row = next(iter(train_loader))
# model(data_row)