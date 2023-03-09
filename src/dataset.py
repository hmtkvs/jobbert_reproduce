import random
import pandas as pd
import numpy as np
from transformers import BertModel, AutoTokenizer
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



class JobTitleDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_token_len, K):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.K = K
        self.training_pairs = self._prepare_training_data(data_frame)
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, index):
        job_title, positive_skill = self.training_pairs[index]['job_title'], self.training_pairs[index]['positive_skill']

        job_title_encoding = self.tokenizer.encode_plus(
          job_title,
          add_special_tokens=False,
          max_length=self.max_token_len,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        positive_skill_IDs = self.skill_to_id[positive_skill]

        random_negatives = self.skill_frequencies.sample(n=self.K, weights=self.skill_frequencies.values).index.tolist()
        negative_skills_IDs = [self.skill_to_id[item] for item in random_negatives]

        return dict(
          input_job_title=job_title_encoding["input_ids"].flatten(),
          positive_skill=torch.tensor(positive_skill_IDs).unsqueeze(0),
          negative_skills=torch.tensor(negative_skills_IDs).unsqueeze(0)
                  )

    def _prepare_training_data(self, df):
        all_skills_series = df['positive_skill'].str.strip().str.split(', ').explode()
        all_skills_series = all_skills_series.apply(lambda x: x.strip())
        all_unique_skills = all_skills_series.unique().tolist()
        self.skill_frequencies = all_skills_series.value_counts(normalize=True) ** 0.75
        self.num_of_unique_skills = len(all_unique_skills)
        self.skill_to_id = {skill: i for i, skill in enumerate(all_unique_skills)}

        df_ = df.merge(all_skills_series, left_index=True, right_index=True)
        df_new = df_.rename(columns={'positive_skill_y':'positive_skill'})
        df_new.drop(columns=['positive_skill_x'], inplace=True)
        df_new = df_new.sample(frac=1).reset_index(drop=True)
        data_dict = df_new[['job_title', 'positive_skill']].to_dict('records')
        return data_dict