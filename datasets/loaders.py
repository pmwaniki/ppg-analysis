from torch.utils.data import Dataset
# import joblib
import numpy as np
import torch
from settings import segment_length, segment_slide
import _pickle as pickle

def load_file(filename):
    with open(filename,'rb') as f:
        data=pickle.load(f)
    return data


def normalize(x, mean=[0.3, 0.6], std=[0.09, 0.13]):
    mean_tensor = x.new_empty(x.size()).fill_(0)
    mean_tensor.index_fill_(1, torch.tensor(0, device=x.device), mean[0])
    mean_tensor.index_fill_(1, torch.tensor(1, device=x.device), mean[1])

    std_tensor = x.new_empty(x.size()).fill_(0)
    std_tensor.index_fill_(1, torch.tensor(0, device=x.device), std[0])
    std_tensor.index_fill_(1, torch.tensor(1, device=x.device), std[1])

    output = x - mean_tensor
    output = torch.div(output, std_tensor)
    return output


class TriageDataset(Dataset):
    def __init__(self, data, labels=None, stft_fun=None, transforms=None, aug_raw=[], normalize=False, sample_by=None):
        self.data = data
        self.aug_raw = aug_raw
        self.normalize = normalize
        self.labels = labels
        self.stft_fun = stft_fun
        self.transforms = transforms
        self.sample_by = sample_by
        if self.sample_by:
            self.unique_ids = self.data[self.sample_by].unique()
        # self.client = MongoClient('mongodb://%s:%s@%s' % (Mongo.user.value, Mongo.password.value,Mongo.host.value),
        #                           connect=False)

    def __len__(self):
        if self.sample_by:
            return len(self.unique_ids)
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.sample_by:
            rows = self.data.loc[self.data[self.sample_by] == self.unique_ids[item], :].reset_index()
            sampled_row=torch.randperm(rows.shape[0],).numpy()[0]
            row=rows.iloc[sampled_row, :]
        else:
            row = self.data.iloc[item, :]
        # ppg = joblib.load(row['filename'])
        ppg = load_file(row['filename'])

        red, infrared = np.array(ppg["red"]), np.array(ppg["infrared"])

        for aug in self.aug_raw:
            red, infrared = aug(red, infrared)

        if self.stft_fun:
            red_stft = self.stft_fun(red)
            infrared_stft = self.stft_fun(infrared)
            x_stft = np.stack([red_stft, infrared_stft])
            x_stft = torch.tensor(x_stft)
        if self.normalize:
            red = (red - 0.3) / 0.09
            infrared = (infrared - 0.6) / 0.13
        x = np.stack([red, infrared])
        x = torch.tensor(x)

        if self.transforms:
            x_stft = self.transforms(x_stft)
        if self.labels is None:
            if self.stft_fun is None:
                return x
            return x, x_stft
        else:
            y = row[self.labels]
            if self.stft_fun is None:
                return x, torch.tensor([y, ])
            return x, x_stft, torch.tensor([y, ])


class TriagePairs(Dataset):
    def __init__(self, data, id_var='id', position_var='position', stft_fun=None, transforms=None, aug_raw=[],
                 overlap=False, normalize=False, pretext='sample'):
        super().__init__()
        self.data = data
        self.id_var = id_var
        self.position_var = position_var
        self.ids = data[id_var].unique()
        self.stft_fun = stft_fun
        self.transforms = transforms
        self.raw_aug = aug_raw
        self.normalize = normalize
        self.pretext = pretext

        if overlap:
            self.position_distance = 0
        else:
            self.position_distance = int(segment_length / segment_slide)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        rows = self.data.loc[self.data[self.id_var] == self.ids[item], :].reset_index()
        sampled_row = torch.randperm(rows.shape[0], ).numpy()[0]
        sample_position = rows.iloc[sampled_row][self.position_var]
        if self.pretext == "augment":
            sample_position2 = sample_position
        elif self.pretext == "sample":
            rows2 = rows.loc[(rows[self.position_var] < (sample_position - self.position_distance)) | (
                        rows[self.position_var] > (sample_position + self.position_distance))].reset_index()
            sampled_row2 = torch.randperm(rows2.shape[0] ).numpy()[0]
            sample_position2 = rows2.iloc[sampled_row2][self.position_var]
        else:
            raise NotImplementedError(f"Pretext task {self.pretext} not implemented")
        sample_rows = rows.loc[rows[self.position_var].isin([sample_position, sample_position2])]

        # ppg1 = joblib.load(sample_rows.iloc[0]['filename'])
        ppg1 = load_file(sample_rows.iloc[0]['filename'])
        if self.pretext == "augment":
            ppg2 = ppg1
        elif self.pretext == "sample":
            # ppg2 = joblib.load(sample_rows.iloc[1]['filename'])
            ppg2 = load_file(sample_rows.iloc[1]['filename'])

        red1, infrared1 = ppg1["red"], ppg1["infrared"]
        red2, infrared2 = ppg2["red"], ppg2["infrared"]

        for aug in self.raw_aug:
            red1, infrared1 = aug(red1, infrared1)
            red2, infrared2 = aug(red2, infrared2)

        if self.normalize:
            red1 = (np.array(red1) - 0.3) / 0.09
            infrared1 = (np.array(infrared1) - 0.6) / 0.13
            red2 = (np.array(red2) - 0.3) / 0.09
            infrared2 = (np.array(infrared2) - 0.6) / 0.13

        x1 = torch.tensor(np.stack([red1, infrared1]))
        x2 = torch.tensor(np.stack([red2, infrared2]))

        if self.stft_fun:
            stft_red1 = self.stft_fun(red1)
            stft_infrared1 = self.stft_fun(infrared1)
            stft_red2 = self.stft_fun(red2)
            stft_infrared2 = self.stft_fun(infrared2)
            stft_x1 = torch.tensor(np.stack([stft_red1, stft_infrared1]))
            stft_x2 = torch.tensor(np.stack([stft_red2, stft_infrared2]))

            if self.transforms:
                stft_x1 = self.transforms(stft_x1)
                stft_x2 = self.transforms(stft_x2)
            return x1, stft_x1, x2, stft_x2
        else:
            return x1, x2
