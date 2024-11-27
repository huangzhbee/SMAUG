import ast
import pickle
import json
import os
import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def extract_half_hour(x):
    hour = x.hour
    min = x.minute
    t = 2 * hour
    if min > 30:
        t = t + 1
    return t


def parse_id(data, max_seq_length):
    if len(data) == 0:
        return None
    else:
        locs = []
        time_bins = []
        seqs = []
        session_ids = []
        for _, row in data.iterrows():
            locs.append(row['location'])
            time_bins.append(row['time_bin'])
            session_ids.append(row['session_idx'])
            seq = (row['session'] + [0] * max_seq_length)[:max_seq_length]
            seqs.append(seq)

        return locs, time_bins, seqs, session_ids


def get_idlist():
    with open('../data/shanghai/user2id.json', 'r') as f:
        patient_id = json.load(f)
    return patient_id


class Shanghai_Dataset(Dataset):
    def __init__(self, use_index_list=None, seed=0):
        np.random.seed(seed)  # seed for ground truth choice
        max_seq_length = 30  # max sequence length

        self.Locations = []
        self.Time_bins = []
        self.App_seqs = []
        self.Session_ids = []

        path = ("shanghai_appinfo.pk")

        if os.path.isfile(path) == False:  # if datasetfile is none, create
            data = pd.read_csv('../data/shanghai/stay point/app_usage_trace_30min_session.txt', sep='\t', header=0)
            data['session'] = data['session'].apply(ast.literal_eval)
            data['start_time'] = pd.to_datetime(data['start_time'], format='%Y-%m-%d %H:%M:%S')
            data['time_bin'] = data['start_time'].apply(lambda x: extract_half_hour(x))


            idlist = get_idlist()
            for id_ in tqdm.tqdm(idlist.keys()):
                try:
                    user_data = data[data['user'] == int(id_)]
                    output = parse_id(user_data, max_seq_length)
                    if output is not None:
                        locs, time_bins, seqs, session_ids = output

                        self.Locations += locs
                        self.Time_bins += time_bins
                        self.App_seqs += seqs
                        self.Session_ids += session_ids

                except Exception as e:
                    print(id_, e)
                    continue

            self.Locations = np.array(self.Locations)
            self.Time_bins = np.array(self.Time_bins)
            self.App_seqs = np.array(self.App_seqs)
            self.Session_ids = np.array(self.Session_ids)

            self.Locations = self.Locations.astype("int64")
            self.Time_bins = self.Time_bins.astype("int64")
            self.App_seqs = self.App_seqs.astype("int64")
            self.Session_ids = self.Session_ids.astype("int64")

            with open(path, "wb") as f:
                pickle.dump(
                    [self.Locations, self.Time_bins, self.App_seqs, self.Session_ids],
                    f
                )
        else:  # load datasetfile
            with open(path, "rb") as f:
                self.Locations, self.Time_bins, self.App_seqs, self.Session_ids = pickle.load(
                    f
                )
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.Locations))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]

        s = {
            "timepoints": self.Time_bins[index],
            "locations": self.Locations[index],
            "app_seq": self.App_seqs[index],
            "session_id": self.Session_ids[index]
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16):
    # only to obtain total length of dataset
    dataset = Shanghai_Dataset(seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = int(nfold * 0.2 * len(dataset))
    end = int((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = int(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_dataset = Shanghai_Dataset(use_index_list=train_index, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = Shanghai_Dataset(use_index_list=valid_index, seed=seed)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = Shanghai_Dataset(use_index_list=test_index, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
