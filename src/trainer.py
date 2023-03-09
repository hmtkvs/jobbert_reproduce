import random
import pandas as pd
import numpy as np
from transformers import BertModel, AutoTokenizer
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from dataset import JobTitleDataset
from classifier import Jobbert_neg_sampling


# read the data frame with two columns: title, positive_skill
df = pd.read_csv("../jobbert/v1401_1_4_train.csv")
df[['InputConcat']] = df[['InputConcat']].astype(str)
df_split = df['InputConcat'].str.split('<h>', expand=True)
dfIC = pd.concat([df, df_split], axis=1)
df_ = dfIC[[0, 2]]
df_raw = df_.rename(columns={0:'job_title',2:'positive_skill'})
df_raw = df_raw[~df_raw.apply(lambda x: x.str.strip() == '', axis=1).any(axis=1)]
df_raw.sample()

training_pairs, validation_pairs = np.split(df_raw.sample(frac=0.025, random_state=42), [int(1000)])
print(training_pairs.shape, validation_pairs.shape)

# Training Parameters
batch_size = 8
num_epochs = 5
# Initialize the data loader
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

ds_train = JobTitleDataset(training_pairs, tokenizer, max_token_len=16, K=5)
ds_val = JobTitleDataset(validation_pairs, tokenizer, max_token_len=16, K=5)

train_loader  = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)
val_loader  = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True)

# get training parameters
num_of_unique_skills = ds_train.num_of_unique_skills

# Initialize the model
model = Jobbert_neg_sampling(embedding_size=300, vocab_size=num_of_unique_skills)


# Initialize the optimizers
optimizer_gating_mlp = optim.SGD(model.mlp_layers.parameters(), lr=0.05)
optimizer_context_matrix = optim.Adagrad(model.embeddings_context.parameters(), lr=0.01)

num_epochs = 2
loss_func = nn.NLLLoss()

# define your training loop
for epoch in range(num_epochs):
    total_loss = 0
    for i, input in enumerate(train_loader):
        # clear gradients
        optimizer_gating_mlp.zero_grad()
        optimizer_context_matrix.zero_grad()

        # forward pass
        loss = model(input)
        #print('J_T.shape: ', J_T.shape)
        #loss = negative_log_likelihood_loss(J_T, target)
        #loss = loss_func(J_T, target)

        # backward pass
        loss.backward()

        # update weights
        optimizer_gating_mlp.step()
        optimizer_context_matrix.step()

        total_loss += loss.item()

        #target = J_T.argmax(dim=1)

    print(f"Epoch {epoch}: loss={total_loss}")