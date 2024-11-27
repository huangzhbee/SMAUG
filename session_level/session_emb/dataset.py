import ast
import pickle
import json
import os
import tqdm
import numpy as np
import pandas as pd
import datetime
from torch.utils.data import DataLoader, Dataset


class Shanghai_Dataset(Dataset):
    def __init__(self, use_index_list=None, seed=1):
        np.random.seed(seed)  # seed for ground truth choice
        max_seq_length = 100  # max sequence length

        self.app_seq_embs = []
        self.seq_idxs = []

        data = pd.read_csv('../../data/shanghai/stay point/app_usage_trace_30min_session.txt', sep='\t', header=0)
        data['session'] = data['session'].apply(ast.literal_eval)

        with open("../../data/shanghai/appemb.pk", "rb") as f:
            _, vectors = pickle.load(f)

        location_emb = pd.read_csv('../data/shanghai/Basestation/cluster_embedding.txt', sep='\t', names=['idx', 'emb'])

        for _, row in tqdm.tqdm(data.iterrows(), total=len(data)):
            seq = (row['session'] + [0] * max_seq_length)[:max_seq_length]
            seq_emb = np.array([vectors[a] if a != 0 else ([0] * 16) for a in seq])
            self.app_seq_embs.append(seq_emb)

        self.app_seq_embs = np.array(self.app_seq_embs)
        self.seq_idxs = np.array(data['session_idx']).astype("int64")

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.seq_idxs))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "app_seq_emb": self.app_seq_embs[index],
            "seq_idx": self.seq_idxs[index]
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, batch_size=32):
    # only to obtain total length of dataset
    dataset = Shanghai_Dataset(seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    np.random.seed(seed)
    np.random.shuffle(indlist)
    num_train = (int)(len(dataset) * 0.8)
    train_index = indlist[:num_train]
    valid_index = indlist[num_train:]

    train_dataset = Shanghai_Dataset(use_index_list=train_index, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Shanghai_Dataset(use_index_list=valid_index, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, all_loader
